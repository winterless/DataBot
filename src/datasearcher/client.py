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
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .function_tools import execute_tool_call, get_tool_schemas
    from .source_selector import select_candidates_by_policy
except ImportError:  # pragma: no cover - direct script execution fallback
    from function_tools import execute_tool_call, get_tool_schemas
    from source_selector import select_candidates_by_policy

DEFAULT_ALIYUN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ALIYUN_MODEL = "qwen-plus"
DEFAULT_CONFIG_PATH = "configs/datasearcher_api.json"
DEFAULT_SOURCE_POLICY = {"huggingface": 6, "github": 4}
DEFAULT_OUTPUT_PATH = "out/datasearcher/sample.json"
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


def _render_prompt_template(prompt_template: str, source_policy: Dict[str, int]) -> str:
    return (
        prompt_template.replace("{hf_count}", str(int(source_policy.get("huggingface", 6))))
        .replace("{gh_count}", str(int(source_policy.get("github", 4))))
        .strip()
    )


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
    hf_need = int(source_policy.get("huggingface", 6))
    gh_need = int(source_policy.get("github", 4))
    return (
        "You are DataSearcher semantic router. "
        "You MUST discover real data sources only via tool calls. "
        "Never fabricate URL/repo_id. "
        f"Target ratio: huggingface={hf_need}, github={gh_need}. "
        "Call tools first, then stop."
    )


def _build_user_prompt(raw_prompt: str, source_policy: Dict[str, int]) -> str:
    hf_need = int(source_policy.get("huggingface", 6))
    gh_need = int(source_policy.get("github", 4))
    return (
        f"需求: {raw_prompt}\n"
        f"请分别调用 HuggingFace 与 GitHub 工具检索候选，目标配比 HF={hf_need}, GH={gh_need}。"
    )


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

    # Static high-recall seeds for instruction/chat/coding datasets.
    for seed in ["instruction", "chat", "alpaca", "reasoning", "code", "llm"]:
        if seed not in fallback:
            fallback.append(seed)

    normalized_initial = initial_query.strip().lower()
    return [q for q in fallback if q and q.strip().lower() != normalized_initial]


def _collect_candidates_from_tools(
    tool_calls: List[Dict[str, Any]],
    timeout_s: int,
    source_policy: Dict[str, int],
    raw_prompt: str,
) -> Dict[str, Any]:
    hf_need = int(source_policy.get("huggingface", 6))
    hf_rows: List[Dict[str, Any]] = []
    gh_rows: List[Dict[str, Any]] = []
    saw_hf_call = False
    hf_primary_query = ""
    hf_fallback_queries: List[str] = []

    for call in tool_calls:
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = str(fn.get("name", "")).strip()
        args = fn.get("arguments", "{}")
        parsed_args = _parse_tool_arguments(args)
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
            _merge_unique_candidates(gh_rows, [x for x in source_rows if isinstance(x, dict)])

    # HF fallback: when model query is too narrow or model didn't call HF tool.
    if len(hf_rows) < hf_need:
        fallback_seed_query = hf_primary_query if saw_hf_call else raw_prompt
        fallback_queries = _build_hf_fallback_queries(fallback_seed_query, raw_prompt)
        for q in fallback_queries:
            if len(hf_rows) >= max(hf_need, 10):
                break
            try:
                fallback_result = execute_tool_call(
                    "search_huggingface_datasets",
                    {"query": q, "limit": max(hf_need * 2, 10)},
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

    return {
        "huggingface": hf_rows,
        "github": gh_rows,
        "hf_primary_query": hf_primary_query,
        "hf_fallback_queries": hf_fallback_queries,
    }


def _load_source_policy(cfg: Dict[str, Any]) -> Dict[str, int]:
    inline_policy = cfg.get("source_policy") if isinstance(cfg.get("source_policy"), dict) else {}
    policy_file = str(cfg.get("source_policy_file", "")).strip()

    policy: Dict[str, Any] = dict(inline_policy)
    if policy_file:
        path = Path(policy_file)
        if path.exists():
            policy = json.loads(path.read_text(encoding="utf-8"))
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

    try:
        intent_resp = chat_with_retry(
            resolved_base_url,
            model_id,
            messages=intent_messages,
            timeout_s=timeout_s,
            retries=retries,
            headers=headers,
            tools=get_tool_schemas(),
            tool_choice="auto",
        )
    except Exception as e:
        return _response_envelope_failed("TRANSIENT_ERROR", f"Function-calling stage failed: {e}", retryable=True)

    tool_calls = _extract_tool_calls(intent_resp)
    if not tool_calls:
        # Branch B intentionally empty: no REQUIRE_NEW_TOOL emit yet.
        return _response_envelope_failed(
            "VALIDATION_ERROR",
            "No tool call emitted by model in Branch A. Branch B is intentionally not implemented.",
            retryable=False,
            data={"branch_b": {"implemented": False, "status": "TODO"}},
        )

    try:
        candidate_map = _collect_candidates_from_tools(
            tool_calls,
            timeout_s=timeout_s,
            source_policy=source_policy,
            raw_prompt=prompt,
        )
        hf_rows = candidate_map["huggingface"]
        gh_rows = candidate_map["github"]
        hf_primary_query = candidate_map.get("hf_primary_query", "")
        hf_fallback_queries = candidate_map.get("hf_fallback_queries", [])
        selected, notes = select_candidates_by_policy(
            hf_candidates=hf_rows,
            gh_candidates=gh_rows,
            source_policy=source_policy,
            intent_text=prompt,
        )
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

    return _response_envelope_success(
        {
            "provider": resolved_provider,
            "base_url": resolved_base_url,
            "model_id": model_id,
            "source_policy": source_policy,
            "datasets": selected,
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
            prompt = _render_prompt_template(_load_text_file(prompt_file), source_policy)
    if not prompt:
        prompt = "agentic code/data training dataset discovery"

    timeout_s = args.timeout if args.timeout >= 0 else int(os.getenv("TIMEOUT", str(req_cfg.get("timeout", 20))))
    retries = args.retries if args.retries >= 0 else int(os.getenv("RETRIES", str(req_cfg.get("retries", 2))))

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
        )
    except Exception as e:
        envelope = _response_envelope_failed("SYSTEM_ERROR", f"Unexpected failure: {e}", retryable=False)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0 if envelope["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
