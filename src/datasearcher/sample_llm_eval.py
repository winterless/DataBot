#!/usr/bin/env python3
"""
Sample LLM Eval — 调用 API 模型对每个 sample 的第一条数据打分。

复用 searcher 的 API 配置（aliyun/local），逐条评估并实时输出到 terminal。
支持中断后增量续跑（已打分的不重复调用）。
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_CONFIG_PATH = "configs/datasearcher_api.json"
DEFAULT_PROMPT_PATH = "configs/prompts/sample_eval_prompt.txt"
DEFAULT_API_KEY_FILE = ".secrets/alicloud_api_key.txt"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
FIRST_ROW_TRUNCATE_CHARS = 30000  # 16k tokens 下约 3w 字符，留足 prompt 空间


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [LLMEval] {msg}", file=sys.stderr, flush=True)


def _api_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    return f"{base}/v1{path}" if not base.endswith("/v1") else f"{base}{path}"


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_completion(
    base_url: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    timeout_s: int = 120,
    retries: int = 2,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """Simple chat completion, returns content string."""
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
    }
    last_err = None
    for attempt in range(1, retries + 2):
        try:
            resp = _post_json(_api_url(base_url, "/chat/completions"), payload, timeout_s, headers=headers)
            choice = (resp.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content", "")
            return content.strip()
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            last_err = str(e)
            if attempt <= retries:
                time.sleep(attempt)
                continue
            raise RuntimeError(f"Chat API failed: {last_err}") from e
    raise RuntimeError(last_err or "Chat failed")


def _extract_first_row_content(sample_data: Dict[str, Any], truncate: int = FIRST_ROW_TRUNCATE_CHARS) -> str:
    """Extract first row as JSON string, truncated."""
    rows = sample_data.get("rows") or []
    if not rows:
        return "{}"
    row_obj = rows[0]
    row = row_obj.get("row", row_obj) if isinstance(row_obj, dict) else row_obj
    raw = json.dumps(row, ensure_ascii=False, indent=0)
    if len(raw) > truncate:
        raw = raw[:truncate] + "\n...(truncated)"
    return raw


def _parse_score_response(content: str) -> Tuple[float, str]:
    """Parse JSON from model response. Returns (score, reason)."""
    content = content.strip()
    # 提取第一个完整 JSON 对象
    start = content.find("{")
    if start < 0:
        return (0.0, "parse_failed")
    depth = 0
    for i, c in enumerate(content[start:], start=start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(content[start : i + 1])
                    score = float(obj.get("score", 0))
                    score = max(0, min(10, score))
                    reason = str(obj.get("reason", ""))[:80]
                    return (score, reason)
                except (json.JSONDecodeError, TypeError, ValueError):
                    break
    return (0.0, "parse_failed")


def load_already_scored(repo_ids: set, output_path: Path) -> set:
    """Load repo_ids that are already in output file (for resume)."""
    if not output_path.exists():
        return set()
    seen = set()
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = obj.get("repo_id", "")
                if rid and rid in repo_ids:
                    seen.add(rid)
            except json.JSONDecodeError:
                continue
    return seen


def run_llm_eval(
    samples_dir: Path,
    recall_pool_path: Path,
    output_path: Path,
    prompt_path: Path,
    config_path: Path,
    provider: str = "aliyun",
    api_key: str = "",
    base_url: str = "",
    model: str = "",
    timeout_s: int = 120,
    retries: int = 2,
    max_items: int = 0,
    resume: bool = True,
) -> int:
    """
    对 samples 逐条调用 LLM 打分，实时输出到 terminal，增量写入 output_path。
    返回成功打分的数量。
    """
    cfg = {}
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

    prov_cfg = (cfg.get("providers") or {}).get(provider) or {}
    resolved_base = base_url or prov_cfg.get("base_url") or os.getenv("ALIYUN_BASE_URL") or DEFAULT_BASE_URL
    resolved_model = model or prov_cfg.get("model") or os.getenv("ALIYUN_MODEL") or DEFAULT_MODEL

    if provider == "aliyun":
        resolved_key = (
            api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("ALIYUN_API_KEY")
            or (Path(DEFAULT_API_KEY_FILE).read_text().strip() if Path(DEFAULT_API_KEY_FILE).exists() else "")
        )
        if not resolved_key:
            raise ValueError("Aliyun requires API key: DASHSCOPE_API_KEY or --api-key")
        headers = {"Authorization": f"Bearer {resolved_key}"}
    else:
        headers = {}

    prompt_template = prompt_path.read_text(encoding="utf-8")

    recall_map: Dict[str, Dict[str, Any]] = {}
    if recall_pool_path.exists():
        with recall_pool_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rid = str(obj.get("repo_id", "")).strip()
                    if rid:
                        recall_map[rid] = obj
                except json.JSONDecodeError:
                    continue

    sample_files = sorted(samples_dir.glob("*_sample.json"))
    if max_items > 0:
        sample_files = sample_files[:max_items]

    def _sanitize_to_repo_id(name: str) -> str:
        base = name.replace("_sample.json", "").replace(".json", "")
        return base.replace("__", "/", 1) if "__" in base else base

    all_repo_ids = set()
    for fp in sample_files:
        rid = _sanitize_to_repo_id(fp.name)
        all_repo_ids.add(rid)

    already_done = load_already_scored(all_repo_ids, output_path) if resume else set()
    if already_done:
        _log(f"Resume: skipping {len(already_done)} already scored")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if (resume and output_path.exists()) else "w"
    out_file = output_path.open(mode, encoding="utf-8")

    ok_count = 0
    fail_count = 0
    total = len(sample_files)

    try:
        for idx, fp in enumerate(sample_files, start=1):
            repo_id = _sanitize_to_repo_id(fp.name)
            if resume and repo_id in already_done:
                continue

            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception as e:
                _log(f"[{idx}/{total}] SKIP {repo_id}: load error {e}")
                fail_count += 1
                continue

            dataset_id = data.get("dataset", repo_id)
            first_row = _extract_first_row_content(data)

            user_content = prompt_template.replace("{dataset_id}", dataset_id).replace("{first_row_content}", first_row)

            messages = [{"role": "user", "content": user_content}]

            try:
                content = chat_completion(
                    resolved_base,
                    resolved_model,
                    messages,
                    timeout_s=timeout_s,
                    retries=retries,
                    headers=headers,
                )
            except Exception as e:
                _log(f"[{idx}/{total}] FAIL {repo_id}: {e}")
                fail_count += 1
                continue

            score, reason = _parse_score_response(content)
            meta = recall_map.get(repo_id) or {
                "repo_id": repo_id,
                "dataset_name": repo_id.split("/")[-1] if "/" in repo_id else repo_id,
                "source_type": "huggingface",
                "brief_intro": "",
                "verified_meta": {},
            }

            item = dict(meta)
            item["llm_score"] = score
            item["llm_reason"] = reason
            item["sample_path"] = str(fp)
            item["row_count"] = len(data.get("rows") or [])

            out_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_file.flush()

            ok_count += 1
            print(f"[{idx}/{total}] {repo_id} -> llm_score={score:.1f} | {reason}", flush=True)

            time.sleep(0.5)

    finally:
        out_file.close()

    _log(f"Done: ok={ok_count}, fail={fail_count}")
    return ok_count


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based sample quality scoring")
    parser.add_argument("--samples-dir", default="out/datasearcher/samples")
    parser.add_argument("--recall-pool", default="out/datasearcher/recall_pool.jsonl")
    parser.add_argument("--output", default="out/datasearcher/sample_llm_scored.jsonl")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", default="aliyun")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true", help="Do not skip already scored")
    args = parser.parse_args()

    run_llm_eval(
        samples_dir=Path(args.samples_dir),
        recall_pool_path=Path(args.recall_pool),
        output_path=Path(args.output),
        prompt_path=Path(args.prompt),
        config_path=Path(args.config),
        provider=args.provider,
        api_key=args.api_key,
        base_url=args.base_url or None,
        model=args.model or None,
        timeout_s=args.timeout,
        retries=args.retries,
        max_items=args.max_items,
        resume=not args.no_resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
