#!/usr/bin/env python3
"""
DataSearcher Client - Branch A only (Function Calling + local API tools).

Flow:
1) LLM parses intent and decides tool calls.
2) Local Python tools call HF/GitHub official APIs.
3) Selector enforces source_policy (HF/GitHub ratio) and returns final datasets.

Branch B is intentionally left empty for now.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _log(msg: str) -> None:
    """Print execution status to stderr (does not pollute JSON stdout)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [DataSearcher] {msg}", file=sys.stderr)

try:
    from .function_tools import execute_tool_call, get_tool_schemas
    from .source_selector import select_candidates_two_layer, _ratio_counts
    from .api_clients.huggingface_api import (
        search_datasets_recent,
        list_datasets_by_author,
        search_datasets_by_tags,
    )
    from .api_clients.github_api import (
        search_repositories_time_sweep,
        search_repositories,
        search_code_for_data_repos,
    )
    from .readme_extractor import (
        extract_linked_repos,
        fetch_github_readme,
        fetch_hf_dataset_readme,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from function_tools import execute_tool_call, get_tool_schemas
    from source_selector import select_candidates_two_layer, _ratio_counts
    from api_clients.huggingface_api import (
        search_datasets_recent,
        list_datasets_by_author,
        search_datasets_by_tags,
    )
    from api_clients.github_api import (
        search_repositories_time_sweep,
        search_repositories,
        search_code_for_data_repos,
    )
    from readme_extractor import (
        extract_linked_repos,
        fetch_github_readme,
        fetch_hf_dataset_readme,
    )

DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen-plus"
DEFAULT_CONFIG_PATH = "configs/datasearcher_api.json"
DEFAULT_SOURCE_POLICY = {"huggingface": 6, "github": 4}
DEFAULT_OUTPUT_PATH = "out/datasearcher/sample.json"
DEFAULT_RECALL_POOL_OUTPUT_PATH = "out/datasearcher/recall_pool.jsonl"
DEFAULT_LLM_TRACE_OUTPUT_PATH = "out/datasearcher/llm_trace.json"
DEFAULT_ALIYUN_API_KEY_FILE = ".secrets/alicloud_api_key.txt"


def _response_envelope_success(data: Dict[str, Any]) -> Dict[str, Any]:
    return {"status": "SUCCESS", "error": None, "data": data}


def _response_envelope_failed(
    code: str, message: str, retryable: bool, data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "status": "FAILED",
        "error": {"code": code, "message": message, "retryable": retryable},
        "data": data,
    }


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _api_url(base_url: str, path: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/v1"):
        return f"{normalized}{path}"
    return f"{normalized}/v1{path}"


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def _load_text_file(path: str) -> str:
    txt_path = Path(path)
    if not txt_path.exists():
        raise ValueError(f"Prompt file not found: {path}")
    return txt_path.read_text(encoding="utf-8")


def _load_api_key_from_file(path: str) -> str:
    key_path = Path(path)
    if not key_path.exists():
        return ""
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _render_prompt_template(
    prompt_template: str,
    source_policy: Dict[str, int],
    layer_cfg: Optional[Dict[str, int]] = None,
) -> str:
    rendered = (
        prompt_template.replace("{hf_count}", str(int(source_policy.get("huggingface", 6))))
        .replace("{gh_count}", str(int(source_policy.get("github", 4))))
    )
    if layer_cfg:
        rendered = (
            rendered.replace("{recall_pool_size}", str(int(layer_cfg.get("recall_pool_size", 120))))
            .replace("{download_size}", str(int(layer_cfg.get("download_size", 10))))
        )
    return rendered.strip()


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_layer_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    layer_cfg = cfg.get("selection") if isinstance(cfg.get("selection"), dict) else {}
    out: Dict[str, Any] = {
        "recall_pool_size": max(1, int(layer_cfg.get("recall_pool_size", 120))),
        "download_size": max(1, int(layer_cfg.get("download_size", 10))),
    }
    pref = layer_cfg.get("preferred_size")
    if isinstance(pref, dict):
        out["preferred_size"] = pref
    ts = cfg.get("time_sweep")
    if isinstance(ts, dict):
        out["sweep_limits"] = {
            "hf_limit": int(ts.get("hf_limit", 100)),
            "gh_limit": int(ts.get("gh_limit", 150)),
        }
    ds = cfg.get("deep_scan")
    if isinstance(ds, dict):
        out["deep_scan"] = {
            "enabled": bool(ds.get("enabled", True)),
            "max_pages": int(ds.get("max_pages", 5)),
            "readme_sample_size": int(ds.get("readme_sample_size", 30)),
            "code_search_limit": int(ds.get("code_search_limit", 50)),
        }
    return out


def _post_json(
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
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


def _get_json(url: str, timeout_s: int, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_model_id(base_url: str, timeout_s: int, headers: Optional[Dict[str, str]] = None) -> str:
    data = _get_json(_api_url(base_url, "/models"), timeout_s, headers=headers)
    items = data.get("data", [])
    if not items:
        raise ValueError("No model returned from /v1/models")
    model_id = items[0].get("id", "")
    if not model_id:
        raise ValueError("Model id missing in /v1/models response")
    return model_id


def chat_with_retry(
    base_url: str,
    model_id: str,
    messages: List[Dict[str, Any]],
    timeout_s: int,
    retries: int,
    headers: Optional[Dict[str, str]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.0,
    }
    if tools:
        payload["tools"] = tools
    if tool_choice:
        payload["tool_choice"] = tool_choice

    last_error: Optional[str] = None
    for attempt in range(1, retries + 2):
        try:
            return _post_json(_api_url(base_url, "/chat/completions"), payload, timeout_s, headers=headers)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {e.code}: {body or str(e)}"
                if attempt <= retries:
                    time.sleep(attempt)
                    continue
                raise RuntimeError(last_error)
            raise RuntimeError(f"HTTP {e.code}: {body or str(e)}")
        except urllib.error.URLError as e:
            last_error = f"URLError: {e}"
            if attempt <= retries:
                time.sleep(attempt)
                continue
            raise RuntimeError(last_error)
        except TimeoutError as e:
            last_error = f"TimeoutError: {e}"
            if attempt <= retries:
                time.sleep(attempt)
                continue
            raise RuntimeError(last_error)
        except Exception as e:
            raise RuntimeError(str(e))
    raise RuntimeError(last_error or "Unknown chat failure")


def _extract_tool_calls(chat_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    choice = (chat_resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []
    return tool_calls if isinstance(tool_calls, list) else []


def _build_intent_system_prompt(source_policy: Dict[str, int]) -> str:
    hf_w = int(source_policy.get("huggingface", 6))
    gh_w = int(source_policy.get("github", 4))
    parts = []
    if hf_w > 0:
        parts.append(f"huggingface={hf_w}")
    if gh_w > 0:
        parts.append(f"github={gh_w}")
    ratio_str = ", ".join(parts) if parts else "huggingface=6, github=4"
    return (
        "You are DataSearcher semantic router. "
        "You MUST discover real data sources only via tool calls. "
        "Never fabricate URL/repo_id. "
        f"Target ratio: {ratio_str}. "
        "Call tools first, then stop."
    )


def _build_user_prompt(raw_prompt: str, source_policy: Dict[str, int]) -> str:
    hf_w = int(source_policy.get("huggingface", 6))
    gh_w = int(source_policy.get("github", 4))
    if hf_w > 0 and gh_w > 0:
        line = f"请分别调用 HuggingFace 与 GitHub 工具检索候选，目标配比 HF={hf_w}, GH={gh_w}。"
    elif hf_w > 0:
        line = f"请调用 HuggingFace 工具检索候选，尽量扩大召回覆盖面（配比 HF={hf_w}）。"
    elif gh_w > 0:
        line = f"请调用 GitHub 工具检索候选，尽量扩大召回覆盖面（配比 GH={gh_w}）。"
    else:
        line = "请调用 HuggingFace 与 GitHub 工具检索候选。"
    return f"需求: {raw_prompt}\n{line}"


def _parse_tool_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}
    return {}


def _candidate_key(item: Dict[str, Any]) -> str:
    return f"{str(item.get('source_type', '')).strip().lower()}::{str(item.get('repo_id', '')).strip()}"


def _merge_unique_candidates(target: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> int:
    existing_keys = {_candidate_key(x) for x in target if isinstance(x, dict)}
    added = 0
    for row in incoming:
        if not isinstance(row, dict):
            continue
        key = _candidate_key(row)
        if key.endswith("::") or key in existing_keys:
            continue
        target.append(row)
        existing_keys.add(key)
        added += 1
    return added


def _build_hf_fallback_queries(initial_query: str, raw_prompt: str) -> List[str]:
    stopwords = {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "your",
        "data",
        "dataset",
        "datasets",
        "training",
        "please",
    }
    base = f"{initial_query} {raw_prompt}".lower()
    tokens = [t for t in re.split(r"[^a-z0-9_]+", base) if len(t) >= 3 and t not in stopwords]
    unique_tokens: List[str] = []
    for token in tokens:
        if token not in unique_tokens:
            unique_tokens.append(token)

    fallback: List[str] = []
    for token in unique_tokens[:8]:
        fallback.append(token)

    # 通用 fallback 关键词，当 prompt 分词不足时补充
    generic_seeds = [
        "agent", "agentic", "instruction", "reasoning", "code",
        "function calling", "trajectory", "tool", "llm",
    ]
    for seed in generic_seeds:
        if seed not in fallback:
            fallback.append(seed)

    normalized_initial = initial_query.strip().lower()
    return [q for q in fallback if q and q.strip().lower() != normalized_initial]


def _build_gh_fallback_queries(initial_query: str, raw_prompt: str) -> List[str]:
    seeds = _build_hf_fallback_queries(initial_query, raw_prompt)
    out: List[str] = []
    for seed in seeds:
        out.append(f"{seed} language:python")
        out.append(f"{seed} llm")
    return out[:12]


_CODEC_SEARCH_STOPWORDS = frozenset({
    "the", "and", "for", "with", "data", "dataset", "datasets",
    "training", "please", "code", "repo", "repository",
})


def _extract_domain_keywords(hf_query: str, gh_query: str) -> List[str]:
    """Extract domain keywords for code search (agent, function call, etc.).
    Excludes generic terms like dataset, data, training that add no specificity.
    """
    base = f"{hf_query} {gh_query}".lower()
    tokens = [t for t in re.split(r"[^a-z0-9_-]+", base) if 2 <= len(t) <= 30]
    seen: set = set()
    out: List[str] = []
    for t in tokens:
        if t not in seen and t not in _CODEC_SEARCH_STOPWORDS:
            seen.add(t)
            out.append(t)
    return out[:5]


def _stub_from_repo_id(source_type: str, repo_id: str) -> Dict[str, Any]:
    """Create minimal candidate stub from repo_id (for README-extracted links)."""
    name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    if source_type == "huggingface":
        return {
            "dataset_name": name,
            "repo_id": repo_id,
            "source_type": "huggingface",
            "source_url": f"https://huggingface.co/datasets/{repo_id}",
            "license": "unknown",
            "downloads": None,
            "likes": None,
            "size": None,
            "size_human": "unknown",
            "size_mb": None,
            "last_modified": "",
            "description": "",
        }
    return {
        "dataset_name": name,
        "repo_id": repo_id,
        "source_type": "github",
        "source_url": f"https://github.com/{repo_id}",
        "license": "unknown",
        "stars": None,
        "size": None,
        "size_human": "unknown",
        "size_mb": None,
        "updated_at": "",
        "description": "",
    }


def _run_deep_scan(
    hf_rows: List[Dict[str, Any]],
    gh_rows: List[Dict[str, Any]],
    timeout_s: int,
    hf_primary_query: str,
    gh_primary_query: str,
    deep_scan_cfg: Optional[Dict[str, Any]],
    source_policy: Optional[Dict[str, int]] = None,
) -> None:
    """Deep scan: code search, HF tags, pagination, README link extraction."""
    if not deep_scan_cfg or not deep_scan_cfg.get("enabled"):
        return
    policy = source_policy or {}
    hf_on = int(policy.get("huggingface", 6)) > 0
    gh_on = int(policy.get("github", 4)) > 0

    cfg = deep_scan_cfg
    max_pages = int(cfg.get("max_pages", 5))
    readme_sample = int(cfg.get("readme_sample_size", 30))
    code_search_limit = int(cfg.get("code_search_limit", 50))

    if gh_on:
        try:
            domain_kw = _extract_domain_keywords(hf_primary_query, gh_primary_query)
            added = _merge_unique_candidates(
                gh_rows,
                search_code_for_data_repos(code_search_limit, timeout_s, domain_keywords=domain_kw),
            )
            if added:
                _log(f"Deep scan: code search (domain={domain_kw or 'generic'}): +{added}")
        except Exception as e:
            _log(f"WARN: Code search failed: {e}")

    if hf_on:
        try:
            added = _merge_unique_candidates(hf_rows, search_datasets_by_tags(limit_per_tag=50, timeout_s=timeout_s))
            if added:
                _log(f"Deep scan: HF semantic tags: +{added}")
        except Exception as e:
            _log(f"WARN: HF tag search failed: {e}")

    if gh_on and gh_primary_query.strip() and max_pages > 1:
        try:
            extra = search_repositories(
                gh_primary_query,
                limit=500,
                sort="stars",
                order="desc",
                timeout_s=timeout_s,
                max_pages=max_pages,
                full_field=True,
            )
            added = _merge_unique_candidates(gh_rows, extra)
            if added:
                _log(f"Deep scan: GH pagination (max_pages={max_pages}): +{added}")
        except Exception as e:
            _log(f"WARN: GH pagination failed: {e}")

    all_content: List[Tuple[str, str, str]] = []
    for item in gh_rows[:readme_sample]:
        rid = str(item.get("repo_id", "")).strip()
        if "/" in rid:
            parts = rid.split("/", 1)
            if len(parts) == 2:
                content = fetch_github_readme(parts[0], parts[1], timeout_s)
                if content:
                    all_content.append(("github", rid, content))
    for item in hf_rows[:readme_sample]:
        rid = str(item.get("repo_id", "")).strip()
        if "/" in rid:
            content = fetch_hf_dataset_readme(rid, timeout_s)
            if content:
                all_content.append(("huggingface", rid, content))

    linked: List[Tuple[str, str]] = []
    for _st, _rid, content in all_content:
        for src, repo_id in extract_linked_repos(content):
            if src and repo_id:
                linked.append((src, repo_id))

    if linked:
        stubs = [_stub_from_repo_id(src, rid) for src, rid in linked]
        added_hf = _merge_unique_candidates(hf_rows, [s for s in stubs if s.get("source_type") == "huggingface"]) if hf_on else 0
        added_gh = _merge_unique_candidates(gh_rows, [s for s in stubs if s.get("source_type") == "github"]) if gh_on else 0
        if added_hf or added_gh:
            _log(f"Deep scan: README links (HF +{added_hf}, GH +{added_gh})")


def _run_time_sweep_and_seed_orgs(
    hf_rows: List[Dict[str, Any]],
    gh_rows: List[Dict[str, Any]],
    timeout_s: int,
    dynamic_seed_orgs: List[str],
    suggested_created_after: Optional[str],
    hf_primary_query: str,
    gh_primary_query: str,
    raw_prompt: str,
    sweep_limits: Optional[Dict[str, int]] = None,
    source_policy: Optional[Dict[str, int]] = None,
) -> None:
    """Merge time-sweep and seed-org results (LLM-driven params only) into hf_rows/gh_rows."""
    policy = source_policy or {}
    hf_on = int(policy.get("huggingface", 6)) > 0
    gh_on = int(policy.get("github", 4)) > 0

    limits = sweep_limits or {}
    hf_limit = int(limits.get("hf_limit", 100))
    gh_limit = int(limits.get("gh_limit", 150))

    if gh_on and suggested_created_after:
        base_q = gh_primary_query.strip() or " ".join(re.split(r"[^a-zA-Z0-9_-]+", raw_prompt)[:3])
        if not base_q:
            base_q = "dataset"
        try:
            added = _merge_unique_candidates(
                gh_rows,
                search_repositories_time_sweep(base_q, suggested_created_after, gh_limit, timeout_s),
            )
            if added:
                _log(f"Time-sweep GH (sort=updated, created:>{suggested_created_after}): +{added}")
        except Exception as e:
            _log(f"WARN: GH time-sweep failed: {e}")

    if hf_on:
        hf_sweep_queries = [q for q in [hf_primary_query.strip()] if q]
        if not hf_sweep_queries and raw_prompt:
            tokens = [t for t in re.split(r"[^a-zA-Z0-9_-]+", raw_prompt.lower()) if len(t) >= 3][:3]
            hf_sweep_queries = tokens or ["dataset"]
        if suggested_created_after and hf_sweep_queries:
            try:
                for q in hf_sweep_queries[:2]:
                    added = _merge_unique_candidates(hf_rows, search_datasets_recent(q, hf_limit, timeout_s))
                    if added:
                        _log(f"Time-sweep HF (sort=createdAt, q={q}): +{added}")
            except Exception as e:
                _log(f"WARN: HF time-sweep failed: {e}")
        for org in dynamic_seed_orgs:
            org = str(org).strip()
            if not org:
                continue
            try:
                added = _merge_unique_candidates(hf_rows, list_datasets_by_author(org, 100, timeout_s))
                if added:
                    _log(f"Seed org {org}: +{added}")
            except Exception as e:
                _log(f"WARN: Seed org {org} failed: {e}")


def _collect_candidates_from_tools(
    tool_calls: List[Dict[str, Any]],
    timeout_s: int,
    source_policy: Dict[str, int],
    raw_prompt: str,
    recall_pool_size: int,
    sweep_limits: Optional[Dict[str, int]] = None,
    deep_scan_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    hf_need, gh_need = _ratio_counts(recall_pool_size, source_policy)
    hf_rows: List[Dict[str, Any]] = []
    gh_rows: List[Dict[str, Any]] = []
    saw_hf_call = False
    saw_gh_call = False
    hf_primary_query = ""
    gh_primary_query = ""
    hf_fallback_queries: List[str] = []
    gh_fallback_queries: List[str] = []

    dynamic_seed_orgs: List[str] = []
    suggested_created_after: Optional[str] = None

    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        args = fn.get("arguments", "{}")
        parsed_args = _parse_tool_arguments(args)

        if name == "set_discovery_parameters":
            try:
                result = execute_tool_call(name, parsed_args, timeout_s=timeout_s)
                dynamic_seed_orgs = result.get("dynamic_seed_orgs") or []
                suggested_created_after = result.get("suggested_created_after")
            except Exception:
                pass
            continue

        if name == "search_huggingface_datasets":
            parsed_args["limit"] = max(int(parsed_args.get("limit", 0) or 0), min(max(hf_need, 20), 100))
        elif name == "search_github_repositories":
            parsed_args["limit"] = max(int(parsed_args.get("limit", 0) or 0), min(max(gh_need, 20), 100))
            parsed_args["sort"] = str(parsed_args.get("sort", "stars") or "stars")
            parsed_args["order"] = str(parsed_args.get("order", "desc") or "desc")
            if deep_scan_cfg and deep_scan_cfg.get("enabled"):
                parsed_args["max_pages"] = int(deep_scan_cfg.get("max_pages", 5))
        try:
            result = execute_tool_call(name, parsed_args, timeout_s=timeout_s)
        except Exception:
            continue
        source_rows = result.get("candidates", [])
        if not isinstance(source_rows, list):
            continue
        if name == "search_huggingface_datasets":
            saw_hf_call = True
            hf_primary_query = str(parsed_args.get("query", "")).strip()
            _merge_unique_candidates(hf_rows, [x for x in source_rows if isinstance(x, dict)])
        elif name == "search_github_repositories":
            saw_gh_call = True
            gh_primary_query = str(parsed_args.get("query", "")).strip()
            _merge_unique_candidates(gh_rows, [x for x in source_rows if isinstance(x, dict)])

    # HF fallback: when model query is too narrow or model didn't call HF tool.
    if hf_need > 0 and len(hf_rows) < hf_need:
        fallback_seed_query = hf_primary_query if saw_hf_call else raw_prompt
        fallback_queries = _build_hf_fallback_queries(fallback_seed_query, raw_prompt)
        for q in fallback_queries:
            if len(hf_rows) >= max(hf_need, 20):
                break
            try:
                fallback_result = execute_tool_call(
                    "search_huggingface_datasets",
                    {"query": q, "limit": max(hf_need, 20)},
                    timeout_s=timeout_s,
                )
            except Exception:
                continue
            rows = fallback_result.get("candidates", [])
            if not isinstance(rows, list):
                continue
            added = _merge_unique_candidates(hf_rows, [x for x in rows if isinstance(x, dict)])
            if added > 0:
                hf_fallback_queries.append(q)

    # GH fallback for larger recall pool.
    if gh_need > 0 and len(gh_rows) < gh_need:
        fallback_seed_query = gh_primary_query if saw_gh_call else raw_prompt
        fallback_queries = _build_gh_fallback_queries(fallback_seed_query, raw_prompt)
        for q in fallback_queries:
            if len(gh_rows) >= max(gh_need, 20):
                break
            try:
                fallback_result = execute_tool_call(
                    "search_github_repositories",
                    {"query": q, "limit": max(gh_need, 20), "sort": "stars", "order": "desc"},
                    timeout_s=timeout_s,
                )
            except Exception:
                continue
            rows = fallback_result.get("candidates", [])
            if not isinstance(rows, list):
                continue
            added = _merge_unique_candidates(gh_rows, [x for x in rows if isinstance(x, dict)])
            if added > 0:
                gh_fallback_queries.append(q)

    _run_deep_scan(
        hf_rows, gh_rows, timeout_s,
        hf_primary_query=hf_primary_query,
        gh_primary_query=gh_primary_query,
        deep_scan_cfg=deep_scan_cfg,
        source_policy=source_policy,
    )

    _run_time_sweep_and_seed_orgs(
        hf_rows, gh_rows, timeout_s,
        dynamic_seed_orgs=dynamic_seed_orgs,
        suggested_created_after=suggested_created_after,
        hf_primary_query=hf_primary_query,
        gh_primary_query=gh_primary_query,
        raw_prompt=raw_prompt,
        sweep_limits=sweep_limits,
        source_policy=source_policy,
    )

    return {
        "huggingface": hf_rows,
        "github": gh_rows,
        "hf_primary_query": hf_primary_query,
        "hf_fallback_queries": hf_fallback_queries,
        "gh_primary_query": gh_primary_query,
        "gh_fallback_queries": gh_fallback_queries,
        "dynamic_seed_orgs": dynamic_seed_orgs,
        "suggested_created_after": suggested_created_after,
    }


def _load_source_policy(cfg: Dict[str, Any]) -> Dict[str, int]:
    """Load source_policy: inline in cfg is primary; optional source_policy_file extends/overrides."""
    policy: Dict[str, Any] = {}
    policy_file = str(cfg.get("source_policy_file", "")).strip()
    if policy_file:
        path = Path(policy_file)
        if path.exists():
            policy = dict(json.loads(path.read_text(encoding="utf-8")))
    inline = cfg.get("source_policy") if isinstance(cfg.get("source_policy"), dict) else {}
    policy.update(inline)  # inline takes precedence
    return {
        "huggingface": int(policy.get("huggingface", DEFAULT_SOURCE_POLICY["huggingface"])),
        "github": int(policy.get("github", DEFAULT_SOURCE_POLICY["github"])),
    }


def run_datasearcher_branch_a(
    provider: str,
    host: str,
    port: int,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout_s: int,
    retries: int,
    source_policy: Dict[str, int],
    recall_pool_size: int,
    download_size: int,
    recall_pool_output: str,
    preferred_size: Optional[Dict[str, Any]] = None,
    llm_trace_output: Optional[str] = None,
    sweep_limits: Optional[Dict[str, int]] = None,
    deep_scan_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_provider = provider.strip().lower()
    headers: Dict[str, str] = {}

    if resolved_provider == "local":
        resolved_base_url = _normalize_base_url(base_url) if base_url else f"http://{host}:{port}"
        model_id = get_model_id(resolved_base_url, timeout_s)
    elif resolved_provider == "aliyun":
        resolved_base_url = _normalize_base_url(base_url or os.getenv("ALIYUN_BASE_URL") or DEFAULT_ALIYUN_BASE_URL)
        resolved_key = api_key or os.getenv("DASHSCOPE_API_KEY") or os.getenv("ALIYUN_API_KEY") or ""
        if not resolved_key:
            return _response_envelope_failed(
                "VALIDATION_ERROR",
                "Aliyun provider requires API key via --api-key or DASHSCOPE_API_KEY/ALIYUN_API_KEY.",
                retryable=False,
            )
        headers = {"Authorization": f"Bearer {resolved_key}"}
        model_id = model or os.getenv("ALIYUN_MODEL") or DEFAULT_ALIYUN_MODEL
    else:
        return _response_envelope_failed(
            "VALIDATION_ERROR",
            f"Unsupported provider: {provider}. Use 'local' or 'aliyun'.",
            retryable=False,
        )

    intent_messages = [
        {"role": "system", "content": _build_intent_system_prompt(source_policy)},
        {"role": "user", "content": _build_user_prompt(prompt, source_policy)},
    ]

    _log(f"Calling LLM ({resolved_provider}/{model_id}) for function-calling...")
    try:
        intent_resp = chat_with_retry(
            resolved_base_url,
            model_id,
            messages=intent_messages,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
            tools=get_tool_schemas(source_policy),
            tool_choice="auto",
        )
    except Exception as e:
        return _response_envelope_failed("TRANSIENT_ERROR", f"Function-calling stage failed: {e}", retryable=True)

    usage = intent_resp.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
    chat_url = _api_url(resolved_base_url, "/chat/completions")
    _log(f"LLM usage: url={chat_url}, prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}")

    if llm_trace_output:
        try:
            trace_obj = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prompt": {
                    "messages": intent_messages,
                    "tools": get_tool_schemas(source_policy),
                    "tool_choice": "auto",
                },
                "response": intent_resp,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
                "tool_calls": _extract_tool_calls(intent_resp),
            }
            trace_path = Path(llm_trace_output)
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            trace_path.write_text(json.dumps(trace_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            _log(f"Wrote LLM trace to {llm_trace_output}")
        except Exception as e:
            _log(f"WARN: llm_trace write failed: {e}")

    tool_calls = _extract_tool_calls(intent_resp)
    _log(f"Tool calls received: {len(tool_calls)}")
    if not tool_calls:
        # Branch B intentionally empty: no REQUIRE_NEW_TOOL emit yet.
        return _response_envelope_failed(
            "VALIDATION_ERROR",
            "No tool call emitted by model in Branch A. Branch B is intentionally not implemented.",
            retryable=False,
            data={"branch_b": {"implemented": False, "status": "TODO"}},
        )

    _log("Executing tool calls (HF/GitHub API)...")
    try:
        candidate_map = _collect_candidates_from_tools(
            tool_calls,
            timeout_s=timeout_s,
            source_policy=source_policy,
            raw_prompt=prompt,
            recall_pool_size=recall_pool_size,
            sweep_limits=sweep_limits,
            deep_scan_cfg=deep_scan_cfg,
        )
        hf_rows = candidate_map["huggingface"]
        gh_rows = candidate_map["github"]
        _log(f"Candidates collected: HF={len(hf_rows)}, GH={len(gh_rows)}")
        hf_primary_query = candidate_map.get("hf_primary_query", "")
        hf_fallback_queries = candidate_map.get("hf_fallback_queries", [])
        gh_primary_query = candidate_map.get("gh_primary_query", "")
        gh_fallback_queries = candidate_map.get("gh_fallback_queries", [])
        layer_result = select_candidates_two_layer(
            hf_candidates=hf_rows,
            gh_candidates=gh_rows,
            source_policy=source_policy,
            intent_text=prompt,
            recall_pool_size=recall_pool_size,
            download_size=download_size,
            preferred_size=preferred_size,
        )
        recall_pool = layer_result["recall_pool"]
        selected = layer_result["download_list"]
        notes = layer_result["notes"]
        _log(f"Selection done: recall_pool={len(recall_pool)}, download_list={len(selected)}")
    except Exception as e:
        return _response_envelope_failed("SYSTEM_ERROR", f"Tool execution stage failed: {e}", retryable=False)

    if not selected:
        return _response_envelope_failed(
            "VALIDATION_ERROR",
            "No valid dataset selected from live tool results.",
            retryable=False,
            data={
                "source_policy": source_policy,
                "tool_calls_count": len(tool_calls),
                "branch_b": {"implemented": False, "status": "TODO"},
            },
        )

    try:
        _write_jsonl(recall_pool_output, recall_pool)
        _log(f"Wrote recall_pool to {recall_pool_output} ({len(recall_pool)} items)")
    except Exception as e:
        notes.append(f"recall_pool写入失败: {e}")
        _log(f"WARN: recall_pool write failed: {e}")

    return _response_envelope_success(
        {
            "provider": resolved_provider,
            "base_url": resolved_base_url,
            "model_id": model_id,
            "source_policy": source_policy,
            "download_list": selected,
            "recall_pool": recall_pool,
            "selection": {
                "recall_pool_size": recall_pool_size,
                "download_size": download_size,
                "recall_pool_output": recall_pool_output,
                "preferred_size": preferred_size,
            },
            "semantic_router": {
                "mode": "branch_a_only",
                "branch_a": "enabled",
                "branch_b": {"implemented": False, "status": "TODO"},
                "notes": notes,
            },
            "trace": {
                "tool_calls_count": len(tool_calls),
                "hf_candidates": len(hf_rows),
                "gh_candidates": len(gh_rows),
                "hf_primary_query": hf_primary_query,
                "hf_fallback_queries": hf_fallback_queries,
                "gh_primary_query": gh_primary_query,
                "gh_fallback_queries": gh_fallback_queries,
                "dynamic_seed_orgs": candidate_map.get("dynamic_seed_orgs", []),
                "suggested_created_after": candidate_map.get("suggested_created_after"),
                "llm_usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            },
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="DataSearcher client (Branch A function-calling only)")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--provider", "--api", dest="provider", default="")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=-1)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--timeout", type=int, default=-1)
    parser.add_argument("--retries", type=int, default=-1)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--recall-pool-output", default=DEFAULT_RECALL_POOL_OUTPUT_PATH)
    args = parser.parse_args()

    try:
        cfg = _load_config(args.config)
    except Exception as e:
        envelope = _response_envelope_failed("VALIDATION_ERROR", str(e), retryable=False)
        print(json.dumps(envelope, ensure_ascii=False, indent=2))
        return 1

    providers_cfg = cfg.get("providers", {})
    local_cfg = providers_cfg.get("local", {})
    aliyun_cfg = providers_cfg.get("aliyun", {})
    req_cfg = cfg.get("request", {})
    source_policy = _load_source_policy(cfg)
    layer_cfg = _load_layer_config(cfg)

    provider = (args.provider or os.getenv("PROVIDER") or cfg.get("default_provider") or "local").strip().lower()
    host = args.host or os.getenv("HOST") or str(local_cfg.get("host", "127.0.0.1"))
    port = args.port if args.port >= 0 else int(os.getenv("PORT", str(local_cfg.get("port", 8000))))

    if provider == "aliyun":
        base_url = args.base_url or os.getenv("BASE_URL") or str(aliyun_cfg.get("base_url", DEFAULT_ALIYUN_BASE_URL))
        model = args.model or os.getenv("MODEL") or str(aliyun_cfg.get("model", DEFAULT_ALIYUN_MODEL))
        api_key_env = str(aliyun_cfg.get("api_key_env", "DASHSCOPE_API_KEY"))
        api_key = (
            args.api_key
            or os.getenv("API_KEY")
            or os.getenv(api_key_env)
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("ALIYUN_API_KEY")
            or _load_api_key_from_file(DEFAULT_ALIYUN_API_KEY_FILE)
            or ""
        )
    else:
        base_url = args.base_url or os.getenv("BASE_URL") or str(local_cfg.get("base_url", ""))
        model = args.model or os.getenv("MODEL") or ""
        api_key = args.api_key or os.getenv("API_KEY") or ""

    prompt = args.prompt or os.getenv("PROMPT") or str(cfg.get("prompt", "")).strip()
    if not prompt:
        prompt_file = str(cfg.get("prompt_file", "")).strip()
        if prompt_file:
            prompt = _render_prompt_template(_load_text_file(prompt_file), source_policy, layer_cfg)
    if not prompt:
        prompt = "agentic code/data training dataset discovery"

    timeout_s = args.timeout if args.timeout >= 0 else int(os.getenv("TIMEOUT", str(req_cfg.get("timeout", 20))))
    retries = args.retries if args.retries >= 0 else int(os.getenv("RETRIES", str(req_cfg.get("retries", 2))))
    recall_pool_size = int(os.getenv("RECALL_POOL_SIZE", str(layer_cfg["recall_pool_size"])))
    download_size = int(os.getenv("DOWNLOAD_SIZE", str(layer_cfg["download_size"])))
    recall_pool_output = args.recall_pool_output or str(cfg.get("recall_pool_output", DEFAULT_RECALL_POOL_OUTPUT_PATH))
    preferred_size = layer_cfg.get("preferred_size") if isinstance(layer_cfg.get("preferred_size"), dict) else None
    llm_trace_output = str(cfg.get("llm_trace_output", DEFAULT_LLM_TRACE_OUTPUT_PATH)).strip() or DEFAULT_LLM_TRACE_OUTPUT_PATH
    sweep_limits = layer_cfg.get("sweep_limits") if isinstance(layer_cfg.get("sweep_limits"), dict) else None
    deep_scan_cfg = layer_cfg.get("deep_scan") if isinstance(layer_cfg.get("deep_scan"), dict) else None

    _log(f"Starting: provider={provider}, recall_pool_size={recall_pool_size}, download_size={download_size}")
    if preferred_size:
        _log(f"preferred_size: min_mb={preferred_size.get('min_mb')}, max_mb={preferred_size.get('max_mb')}")
    _log(f"Prompt: {prompt[:80]}..." if len(prompt) > 80 else f"Prompt: {prompt}")

    try:
        envelope = run_datasearcher_branch_a(
            provider=provider,
            host=host,
            port=port,
            base_url=base_url,
            model=model,
            api_key=api_key,
            prompt=prompt,
            timeout_s=timeout_s,
            retries=retries,
            source_policy=source_policy,
            recall_pool_size=recall_pool_size,
            download_size=download_size,
            recall_pool_output=recall_pool_output,
            preferred_size=preferred_size,
            llm_trace_output=llm_trace_output,
            sweep_limits=sweep_limits,
            deep_scan_cfg=deep_scan_cfg,
        )
    except Exception as e:
        envelope = _response_envelope_failed("SYSTEM_ERROR", f"Unexpected failure: {e}", retryable=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"Wrote output to {args.output}")

    status = envelope.get("status", "UNKNOWN")
    _log(f"Done: status={status}")
    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0 if status == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
