#!/usr/bin/env python3
"""
Hugging Face Datasets Server API — 纯 HTTP 获取数据集 Sample（前 5–10 行）。

用于数据探查/验证，严禁使用 snapshot_download 或 hf_hub_download 下载物理文件。
API 文档: https://huggingface.co/docs/datasets-server
"""
from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

BASE_URL = "https://datasets-server.huggingface.co"
DEFAULT_ROW_LENGTH = 5
DEFAULT_RETRIES = 2
DEFAULT_DELAY_SEC = 1.0


def _get_json(url: str, timeout_s: int = 30) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[str]]:
    """GET JSON. Returns (data, status_code, error_msg)."""
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "DataBot-DataSearcher/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status, None
    except urllib.error.HTTPError as e:
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, e.code, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:
        return None, None, str(e)


def fetch_splits(
    repo_id: str,
    timeout_s: int = 30,
    retries: int = DEFAULT_RETRIES,
    delay_sec: float = DEFAULT_DELAY_SEC,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    请求 splits API，解析第一个可用的 config 和 split。
    Returns (config, split, error). 若失败返回 (None, None, error)，404 表示 Viewer_Disabled。
    """
    rid_enc = urllib.parse.quote(str(repo_id), safe="")
    url = f"{BASE_URL}/splits?dataset={rid_enc}"
    last_err = None
    for attempt in range(1, retries + 2):
        if attempt > 1:
            time.sleep(delay_sec * attempt)
        data, status, err = _get_json(url, timeout_s=timeout_s)
        if status == 404:
            return None, None, "Viewer_Disabled"
        if status in (429, 501, 500):
            last_err = err or f"HTTP {status}"
            continue
        if data is None:
            last_err = err
            continue
        splits = data.get("splits") or []
        for s in splits:
            if isinstance(s, dict):
                cfg = s.get("config")
                sp = s.get("split")
                if cfg and sp:
                    return str(cfg), str(sp), None
        last_err = "No available config/split"
        break
    return None, None, last_err or "Unknown error"


def _is_retryable_err(err: Optional[str], status: Optional[int]) -> bool:
    """501/500 瞬时错误可重试；401/404 不重试。"""
    if status in (501, 500):
        return True
    if err and ("timed out" in err.lower() or "Connection" in err or "Remote" in err):
        return True
    return False


def _is_exceeds_size_err(err: Optional[str]) -> bool:
    return err is not None and "exceeds the supported size" in err


def fetch_rows(
    repo_id: str,
    config: str,
    split: str,
    offset: int = 0,
    length: int = DEFAULT_ROW_LENGTH,
    timeout_s: int = 60,
    retries: int = DEFAULT_RETRIES,
    delay_sec: float = DEFAULT_DELAY_SEC,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], Optional[int]]:
    """
    请求 rows API，提取 rows 列表。config/split 含 #、+ 等会做 URL 编码。
    Returns (rows, error, num_rows_total). rows 为 [{"row_idx": int, "row": {...}}, ...] 的列表。
    num_rows_total 来自 API 响应，用于随机采样时计算 offset。
    """
    cfg_enc = urllib.parse.quote(str(config), safe="")
    sp_enc = urllib.parse.quote(str(split), safe="")
    rid_enc = urllib.parse.quote(str(repo_id), safe="")
    url = f"{BASE_URL}/rows?dataset={rid_enc}&config={cfg_enc}&split={sp_enc}&offset={offset}&length={length}"
    last_err = None
    last_status = None
    for attempt in range(1, retries + 2):
        if attempt > 1:
            time.sleep(delay_sec * attempt)
        data, status, err = _get_json(url, timeout_s=timeout_s)
        last_status = status
        last_err = err
        if status == 429:
            continue
        if data is None:
            if _is_retryable_err(err, status) and attempt <= retries:
                continue
            return None, err, None
        rows = data.get("rows")
        if isinstance(rows, list):
            num_total = data.get("num_rows_total")
            if isinstance(num_total, (int, float)):
                num_total = int(num_total)
            else:
                num_total = None
            return rows, None, num_total
        if _is_retryable_err(err, status) and attempt <= retries:
            continue
        last_err = "Response missing 'rows'"
        break
    return None, last_err, None


def fetch_sample_via_api(
    repo_id: str,
    length: int = DEFAULT_ROW_LENGTH,
    timeout_s: int = 60,
    retries: int = DEFAULT_RETRIES,
    delay_sec: float = DEFAULT_DELAY_SEC,
    random_sample: bool = False,
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    完整流程：splits -> rows。获取数据集 Sample（纯 JSON）。
    Returns (rows, error). 若 splits 404 则 error="Viewer_Disabled"，不兜底下载物理文件。
    若 rows 返回 "exceeds size"，自动用 length=1 重试。

    random_sample: 若 True，从 [0, num_rows_total) 随机抽取 length 个离散行索引，逐条请求；
    否则取前 N 行。
    """
    config, split, err = fetch_splits(
        repo_id, timeout_s=timeout_s, retries=retries, delay_sec=delay_sec
    )
    if err:
        return None, err
    time.sleep(delay_sec)

    if random_sample:
        # 先请求 1 行获取 num_rows_total
        _, err_probe, num_total = fetch_rows(
            repo_id, config, split,
            offset=0, length=1,
            timeout_s=timeout_s, retries=retries, delay_sec=delay_sec,
        )
        if err_probe:
            return None, err_probe
        if num_total is None or num_total < 1:
            # 无法获取总数，退化为取前 N 行
            rows, err, _ = fetch_rows(
                repo_id, config, split,
                offset=0, length=length,
                timeout_s=timeout_s, retries=retries, delay_sec=delay_sec,
            )
            return rows, err
        # 从完整范围内随机抽取 length 个不重复索引
        n = min(length, num_total)
        indices = random.sample(range(num_total), n)
        collected: List[Dict[str, Any]] = []
        for idx in indices:
            time.sleep(delay_sec)
            chunk, err_chunk, _ = fetch_rows(
                repo_id, config, split,
                offset=idx, length=1,
                timeout_s=timeout_s, retries=retries, delay_sec=delay_sec,
            )
            if err_chunk:
                return None, err_chunk
            if chunk:
                collected.extend(chunk)
        return collected, None

    rows, err, _ = fetch_rows(
        repo_id, config, split,
        offset=0, length=length,
        timeout_s=timeout_s, retries=retries, delay_sec=delay_sec,
    )
    if err and _is_exceeds_size_err(err) and length > 1:
        time.sleep(delay_sec)
        rows, err, _ = fetch_rows(
            repo_id, config, split,
            offset=0, length=1,
            timeout_s=timeout_s, retries=retries, delay_sec=delay_sec,
        )
    return rows, err
