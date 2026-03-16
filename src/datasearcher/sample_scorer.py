#!/usr/bin/env python3
"""
Sample Scorer — 对已下载的 slice samples 进行打分排序。

读取 samples/ 目录下的 *_sample.json，结合 recall_pool 的 metadata 计算分数，
输出带 score 的排序列表。纯本地计算，不发起网络请求。

计分规则:
1. 基础分: row_count * 1.0（样本行数）
2. 平台热度: HF downloads/likes 各上限 5 分，渐近逼近（5*x/(x+k)）；GH stars 线性
3. Intent 匹配: 每个匹配 token +3.0（若提供 intent_text）
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 渐近函数：score = cap * x / (x + k)，x->∞ 时趋近 cap
_ASYMPTOTIC_CAP = 5.0
_LIKES_K = 500.0   # likes 半饱和点，约 500 likes 得 2.5 分
_DOWNLOADS_K = 2000.0   # downloads 半饱和点，约 2k 得 2.5 分


def _asymptotic_score(x: float, k: float, cap: float = _ASYMPTOTIC_CAP) -> float:
    """渐近逼近 cap，非线性：cap * x / (x + k)"""
    if x <= 0:
        return 0.0
    return cap * x / (x + k)


def _tokenize(text: str) -> List[str]:
    return [x for x in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if x]


def _score_sample(
    row_count: int,
    meta: Dict[str, Any],
    intent_text: str = "",
) -> float:
    """
    对单个 sample 打分。

    计分规则:
    1. 基础分: row_count * 1.0
    2. 平台热度: HF downloads/likes 各上限 5 分（渐近）；GH stars 线性
    3. Intent 匹配: 每个匹配 token +3.0
    """
    score = float(row_count) * 1.0

    source_type = str(meta.get("source_type", "")).lower()
    vm = meta.get("verified_meta") or {}
    if source_type == "huggingface":
        downloads = float(vm.get("downloads") or 0)
        likes = float(vm.get("likes") or 0)
        score += _asymptotic_score(downloads, _DOWNLOADS_K)
        score += _asymptotic_score(likes, _LIKES_K)
    elif source_type == "github":
        score += float(vm.get("stars") or 0) * 0.01

    if intent_text:
        text = " ".join(
            [
                str(meta.get("dataset_name", "")),
                str(meta.get("repo_id", "")),
                str(meta.get("brief_intro", "")),
            ]
        ).lower()
        intent_tokens = set(_tokenize(intent_text))
        for token in intent_tokens:
            if token and token in text:
                score += 3.0

    return round(score, 4)


def _sanitize_to_repo_id(filename: str) -> str:
    """xxx__yyy_sample.json -> xxx/yyy"""
    base = filename.replace("_sample.json", "").replace(".json", "")
    return base.replace("__", "/", 1) if "__" in base else base


def load_recall_pool_map(recall_pool_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load recall_pool.jsonl -> {repo_id: item}"""
    m: Dict[str, Dict[str, Any]] = {}
    if not recall_pool_path.exists():
        return m
    with recall_pool_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = str(obj.get("repo_id", "")).strip()
                if rid:
                    m[rid] = obj
            except json.JSONDecodeError:
                continue
    return m


def load_llm_scores(llm_scored_path: Path) -> Dict[str, Tuple[float, str]]:
    """Load sample_llm_scored.jsonl -> {repo_id: (llm_score, llm_reason)}"""
    m: Dict[str, Tuple[float, str]] = {}
    if not llm_scored_path.exists():
        return m
    with llm_scored_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = str(obj.get("repo_id", "")).strip()
                if rid:
                    score = float(obj.get("llm_score", 0))
                    reason = str(obj.get("llm_reason", ""))
                    m[rid] = (score, reason)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return m


# LLM 分数权重：作为最重要因子，0-10 分映射到 0-50 分
LLM_SCORE_WEIGHT = 5.0


def score_samples(
    samples_dir: Path,
    recall_pool_path: Path,
    output_path: Path,
    intent_text: str = "",
    llm_scored_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    扫描 samples_dir 下所有 *_sample.json，打分并排序。
    若提供 llm_scored_path 且存在，则合并 llm_score 作为最高权重。
    输出到 output_path（JSONL，每行一个带 score 的 item）。
    """
    recall_map = load_recall_pool_map(recall_pool_path)
    llm_map = load_llm_scores(llm_scored_path or Path()) if llm_scored_path else {}

    results: List[Dict[str, Any]] = []
    for fp in sorted(samples_dir.glob("*_sample.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        dataset = data.get("dataset", "")
        rows = data.get("rows") or []
        row_count = len(rows)

        repo_id = dataset or _sanitize_to_repo_id(fp.name)
        meta = recall_map.get(repo_id) or {
            "repo_id": repo_id,
            "dataset_name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
            "source_type": "huggingface",
            "brief_intro": "",
            "verified_meta": {},
        }

        meta_score = _score_sample(row_count, meta, intent_text)

        item = dict(meta)
        item["row_count"] = row_count
        item["sample_path"] = str(fp)

        if repo_id in llm_map:
            llm_score, llm_reason = llm_map[repo_id]
            item["llm_score"] = llm_score
            item["llm_reason"] = llm_reason
            item["score"] = round(llm_score * LLM_SCORE_WEIGHT + meta_score, 4)
        else:
            item["score"] = round(meta_score, 4)

        results.append(item)

    results.sort(key=lambda x: float(x.get("score", 0)), reverse=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return results
