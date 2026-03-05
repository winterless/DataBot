#!/usr/bin/env python3
"""
DataSearcher Client — Sample 步骤（数据源检索与样本抽取）。

调用 LLM API（local vLLM 或阿里云 dashscope 等），根据 config 中的领域/语言/许可/规模约束，
让模型输出 10 条数据源（HuggingFace + GitHub 双源策略）。对模型返回内容做：
- 清洗 <think> 块、提取 JSON、规范化 download_url/download_command
- 校验 source_type/repo_id/source_url 及 HF·GitHub 存在性
- 输出统一 Response Envelope（status: SUCCESS|FAILED, error 含 code/message/retryable）

与 pipeline_architecture.md 对应：§3.1 DataSearcher，§5.5 统一响应契约，§6.1 错误分类。
"""
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


def _strip_think_block(text: str) -> str:
    # Remove model reasoning blocks like <think> ... </think>.
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def _build_default_prompt(source_policy: Dict[str, int]) -> str:
    hf_count = int(source_policy.get("huggingface", 6))
    gh_count = int(source_policy.get("github", 4))
    return (
        "请输出严格JSON。datasets数组共10条，"
        f"huggingface={hf_count}, github={gh_count}。"
    )


def _api_url(base_url: str, path: str) -> str:
    normalized = _normalize_base_url(base_url)
    if normalized.endswith("/v1"):
        return f"{normalized}{path}"
    return f"{normalized}/v1{path}"


def _load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"Failed to read config file '{path}': {e}") from e


def _load_text_file(path: str) -> str:
    txt_path = Path(path)
    if not txt_path.exists():
        raise ValueError(f"Prompt file not found: {path}")
    try:
        return txt_path.read_text(encoding="utf-8")
    except Exception as e:
        raise ValueError(f"Failed to read prompt file '{path}': {e}") from e


def _load_api_key_from_file(path: str) -> str:
    key_path = Path(path)
    if not key_path.exists():
        return ""
    try:
        return key_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _render_prompt_template(prompt_template: str, source_policy: Dict[str, int]) -> str:
    hf_count = int(source_policy.get("huggingface", 6))
    gh_count = int(source_policy.get("github", 4))
    rendered = prompt_template.replace("{hf_count}", str(hf_count))
    rendered = rendered.replace("{gh_count}", str(gh_count))
    return rendered


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    code_block_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model content.")
    return stripped[start : end + 1]


def _extract_hf_repo_id(hf_dataset_url: str) -> str:
    prefix = "https://huggingface.co/datasets/"
    if not hf_dataset_url.startswith(prefix):
        return ""
    repo = hf_dataset_url[len(prefix) :].strip("/")
    return repo


def _sanitize_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _canonical_hf_download_command(repo_id: str) -> str:
    local_name = _sanitize_local_name(repo_id)
    return (
        f"huggingface-cli download --repo-type dataset {repo_id} "
        f"--local-dir ./data/hf__{local_name}"
    )


def _extract_github_repo_id(source_url: str) -> str:
    prefix = "https://github.com/"
    if not source_url.startswith(prefix):
        return ""
    repo = source_url[len(prefix) :].strip("/")
    parts = repo.split("/")
    if len(parts) < 2:
        return ""
    return f"{parts[0]}/{parts[1]}"


def _canonical_github_download_command(repo_id: str) -> str:
    local_name = _sanitize_local_name(repo_id)
    return f"git clone https://github.com/{repo_id}.git ./data/gh__{local_name}"


def _hf_dataset_exists(repo_id: str, timeout_s: int = 8) -> bool:
    if not repo_id:
        return False
    try:
        _get_json(f"https://huggingface.co/api/datasets/{repo_id}", timeout_s=timeout_s)
        return True
    except Exception:
        return False


def _github_repo_exists(repo_id: str, timeout_s: int = 8) -> bool:
    if not repo_id or "/" not in repo_id:
        return False
    try:
        _get_json(
            f"https://api.github.com/repos/{repo_id}",
            timeout_s=timeout_s,
            headers={"User-Agent": "DataBot-DataSearcher"},
        )
        return True
    except Exception:
        return False


def _validate_structured_result(
    parsed: Dict[str, Any], verify_hf_exists: bool = True, source_policy: Optional[Dict[str, int]] = None
) -> List[str]:
    errors: List[str] = []
    datasets = parsed.get("datasets")
    if not isinstance(datasets, list):
        return ["Field 'datasets' must be a list."]

    if len(datasets) != 10:
        errors.append(f"Field 'datasets' must contain 10 items, got {len(datasets)}.")

    required_keys = [
        "dataset_name",
        "source_type",
        "repo_id",
        "source_url",
        "download_url",
        "download_command",
        "license",
        "reason",
    ]

    hf_count_actual = 0
    gh_count_actual = 0

    for idx, item in enumerate(datasets):
        if not isinstance(item, dict):
            errors.append(f"datasets[{idx}] must be an object.")
            continue
        for key in required_keys:
            val = item.get(key)
            if not isinstance(val, str) or not val.strip():
                errors.append(f"datasets[{idx}].{key} must be a non-empty string.")
        source_type = str(item.get("source_type", "")).strip().lower()
        repo_id = str(item.get("repo_id", "")).strip()
        source_url = str(item.get("source_url", "")).strip()

        if source_type == "huggingface":
            hf_count_actual += 1
            if not source_url.startswith("https://huggingface.co/datasets/"):
                errors.append(
                    f"datasets[{idx}].source_url must start with https://huggingface.co/datasets/."
                )
            extracted = _extract_hf_repo_id(source_url)
            if not extracted:
                errors.append(f"datasets[{idx}] has invalid huggingface source_url.")
            elif repo_id != extracted:
                errors.append(f"datasets[{idx}].repo_id must match source_url ({extracted}).")
            if verify_hf_exists and extracted and not _hf_dataset_exists(extracted):
                errors.append(f"datasets[{idx}] repo does not exist publicly: {source_url}")

            expected = _canonical_hf_download_command(repo_id or extracted)
            download_url_val = item.get("download_url", "")
            download_cmd_val = item.get("download_command", "")
            if isinstance(download_url_val, str) and download_url_val.strip() and "huggingface-cli download" not in download_url_val:
                errors.append(f"datasets[{idx}].download_url should be huggingface-cli command.")
            if isinstance(download_cmd_val, str) and download_cmd_val.strip() and "huggingface-cli download" not in download_cmd_val:
                errors.append(f"datasets[{idx}].download_command should be huggingface-cli command.")
            if not (isinstance(download_url_val, str) and download_url_val.strip()):
                errors.append(
                    f"datasets[{idx}].download_url must be non-empty full download command (e.g. {expected})."
                )
            if not (isinstance(download_cmd_val, str) and download_cmd_val.strip()):
                errors.append(
                    f"datasets[{idx}].download_command must be non-empty full download command (e.g. {expected})."
                )
        elif source_type == "github":
            gh_count_actual += 1
            if not source_url.startswith("https://github.com/"):
                errors.append(
                    f"datasets[{idx}].source_url must start with https://github.com/."
                )
            extracted = _extract_github_repo_id(source_url)
            if not extracted:
                errors.append(f"datasets[{idx}] has invalid github source_url.")
            elif repo_id != extracted:
                errors.append(f"datasets[{idx}].repo_id must match source_url ({extracted}).")
            if verify_hf_exists and extracted and not _github_repo_exists(extracted):
                errors.append(f"datasets[{idx}] repo does not exist publicly: {source_url}")

            expected = _canonical_github_download_command(repo_id or extracted)
            download_url_val = item.get("download_url", "")
            download_cmd_val = item.get("download_command", "")
            if isinstance(download_url_val, str) and download_url_val.strip() and "git clone " not in download_url_val:
                errors.append(f"datasets[{idx}].download_url should be git clone command.")
            if isinstance(download_cmd_val, str) and download_cmd_val.strip() and "git clone " not in download_cmd_val:
                errors.append(f"datasets[{idx}].download_command should be git clone command.")
            if not (isinstance(download_url_val, str) and download_url_val.strip()):
                errors.append(
                    f"datasets[{idx}].download_url must be non-empty full download command (e.g. {expected})."
                )
            if not (isinstance(download_cmd_val, str) and download_cmd_val.strip()):
                errors.append(
                    f"datasets[{idx}].download_command must be non-empty full download command (e.g. {expected})."
                )
        else:
            errors.append(f"datasets[{idx}].source_type must be 'huggingface' or 'github'.")

    policy = source_policy or DEFAULT_SOURCE_POLICY
    hf_expected = int(policy.get("huggingface", 6))
    gh_expected = int(policy.get("github", 4))
    if hf_count_actual != hf_expected:
        errors.append(f"source_policy mismatch: huggingface expected {hf_expected}, got {hf_count_actual}.")
    if gh_count_actual != gh_expected:
        errors.append(f"source_policy mismatch: github expected {gh_expected}, got {gh_count_actual}.")
    return errors


def _normalize_download_commands(parsed: Dict[str, Any]) -> Dict[str, Any]:
    datasets = parsed.get("datasets")
    if not isinstance(datasets, list):
        return parsed
    normalized: List[Dict[str, Any]] = []
    for item in datasets:
        if not isinstance(item, dict):
            normalized.append(item)
            continue
        source_type = str(item.get("source_type", "")).strip().lower()
        source_url = item.get("source_url", "")
        repo_id = str(item.get("repo_id", "")).strip()
        new_item = dict(item)
        if source_type == "huggingface":
            extracted = _extract_hf_repo_id(source_url) if isinstance(source_url, str) else ""
            rid = repo_id or extracted
            cmd = _canonical_hf_download_command(rid) if rid else ""
        elif source_type == "github":
            extracted = _extract_github_repo_id(source_url) if isinstance(source_url, str) else ""
            rid = repo_id or extracted
            cmd = _canonical_github_download_command(rid) if rid else ""
        else:
            cmd = ""
        if cmd:
            new_item["download_url"] = cmd
            new_item["download_command"] = cmd
        normalized.append(new_item)
    out = dict(parsed)
    out["datasets"] = normalized
    return out


def _split_valid_invalid_datasets(
    parsed: Dict[str, Any], verify_hf_exists: bool = True
) -> Dict[str, Any]:
    datasets = parsed.get("datasets")
    if not isinstance(datasets, list):
        return {"valid_datasets": [], "invalid_datasets": [{"index": -1, "errors": ["datasets must be list"]}]}

    required_keys = [
        "dataset_name",
        "source_type",
        "repo_id",
        "source_url",
        "download_url",
        "download_command",
        "license",
        "reason",
    ]
    valid: List[Dict[str, Any]] = []
    invalid: List[Dict[str, Any]] = []
    for idx, item in enumerate(datasets):
        item_errors: List[str] = []
        if not isinstance(item, dict):
            item_errors.append("item must be object")
        else:
            for key in required_keys:
                val = item.get(key)
                if not isinstance(val, str) or not val.strip():
                    item_errors.append(f"{key} must be non-empty string")
            source_type = str(item.get("source_type", "")).strip().lower()
            source_url = str(item.get("source_url", "")).strip()
            repo_id = str(item.get("repo_id", "")).strip()
            if source_type == "huggingface":
                extracted = _extract_hf_repo_id(source_url)
                if not extracted:
                    item_errors.append("invalid huggingface source_url")
                elif repo_id != extracted:
                    item_errors.append("repo_id mismatch for huggingface source_url")
                elif verify_hf_exists and not _hf_dataset_exists(extracted):
                    item_errors.append("huggingface source_url not publicly exists")
            elif source_type == "github":
                extracted = _extract_github_repo_id(source_url)
                if not extracted:
                    item_errors.append("invalid github source_url")
                elif repo_id != extracted:
                    item_errors.append("repo_id mismatch for github source_url")
                elif verify_hf_exists and not _github_repo_exists(extracted):
                    item_errors.append("github source_url not publicly exists")
            else:
                item_errors.append("source_type must be huggingface or github")

            download_url_val = item.get("download_url", "")
            download_cmd_val = item.get("download_command", "")
            if not (isinstance(download_url_val, str) and download_url_val.strip()):
                item_errors.append("download_url must be non-empty full download command")
            if not (isinstance(download_cmd_val, str) and download_cmd_val.strip()):
                item_errors.append("download_command must be non-empty full download command")
        if item_errors:
            invalid.append({"index": idx, "dataset": item, "errors": item_errors})
        else:
            normalized_item = dict(item)
            st = normalized_item.get("source_type", "")
            rid = normalized_item.get("repo_id", "")
            if st == "huggingface" and rid:
                cmd = _canonical_hf_download_command(rid)
                normalized_item["download_url"] = cmd
                normalized_item["download_command"] = cmd
            elif st == "github" and rid:
                cmd = _canonical_github_download_command(rid)
                normalized_item["download_url"] = cmd
                normalized_item["download_command"] = cmd
            valid.append(normalized_item)
    return {"valid_datasets": valid, "invalid_datasets": invalid}


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
        body = resp.read().decode("utf-8")
        return json.loads(body)


def _get_json(url: str, timeout_s: int, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers or {}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        return json.loads(body)


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
    prompt: str,
    timeout_s: int,
    retries: int,
    headers: Optional[Dict[str, str]] = None,
    backoff_base_s: float = 1.0,
) -> Dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    last_error: Optional[str] = None
    for attempt in range(1, retries + 2):
        try:
            return _post_json(
                _api_url(base_url, "/chat/completions"), payload, timeout_s, headers=headers
            )
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8")
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504):
                last_error = f"HTTP {e.code}: {body or str(e)}"
                if attempt <= retries:
                    time.sleep(backoff_base_s * attempt)
                    continue
                raise RuntimeError(last_error)
            raise RuntimeError(f"HTTP {e.code}: {body or str(e)}")
        except urllib.error.URLError as e:
            last_error = f"URLError: {e}"
            if attempt <= retries:
                time.sleep(backoff_base_s * attempt)
                continue
            raise RuntimeError(last_error)
        except TimeoutError as e:
            last_error = f"TimeoutError: {e}"
            if attempt <= retries:
                time.sleep(backoff_base_s * attempt)
                continue
            raise RuntimeError(last_error)
        except Exception as e:
            raise RuntimeError(str(e))

    raise RuntimeError(last_error or "Unknown chat failure")


def run_datasearcher(
    provider: str,
    host: str,
    port: int,
    base_url: str,
    model: str,
    api_key: str,
    prompt: str,
    timeout_s: int,
    retries: int,
    verify_hf_exists: bool,
    source_policy: Dict[str, int],
) -> Dict[str, Any]:
    resolved_provider = provider.strip().lower()
    headers: Dict[str, str] = {}

    if resolved_provider == "local":
        resolved_base_url = _normalize_base_url(base_url) if base_url else f"http://{host}:{port}"
        try:
            model_id = get_model_id(resolved_base_url, timeout_s)
        except Exception as e:
            return _response_envelope_failed(
                "TRANSIENT_ERROR", f"Failed /v1/models: {e}", retryable=True
            )
    elif resolved_provider == "aliyun":
        resolved_base_url = _normalize_base_url(
            base_url or os.getenv("ALIYUN_BASE_URL") or DEFAULT_ALIYUN_BASE_URL
        )
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

    try:
        chat_resp = chat_with_retry(
            resolved_base_url,
            model_id,
            prompt,
            timeout_s,
            retries,
            headers=headers,
        )
    except Exception as e:
        msg = str(e)
        if "HTTP 401" in msg or "HTTP 403" in msg:
            code = "VALIDATION_ERROR"
            retryable = False
        elif "HTTP 400" in msg:
            code = "VALIDATION_ERROR"
            retryable = False
        else:
            code = "TRANSIENT_ERROR"
            retryable = True
        return _response_envelope_failed(code, f"Failed /v1/chat/completions: {msg}", retryable)

    try:
        choice = (chat_resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        content = message.get("content", "")
        cleaned_content = _strip_think_block(content)
        if not cleaned_content:
            return _response_envelope_failed(
                "VALIDATION_ERROR",
                "Model returned empty content after cleaning.",
                retryable=False,
            )
        json_text = _extract_json_text(cleaned_content)
        parsed = json.loads(json_text)
        parsed = _normalize_download_commands(parsed)
        validation_errors = _validate_structured_result(
            parsed,
            verify_hf_exists=verify_hf_exists,
            source_policy=source_policy,
        )
        if validation_errors:
            split_result = _split_valid_invalid_datasets(parsed, verify_hf_exists=verify_hf_exists)
            return _response_envelope_failed(
                "VALIDATION_ERROR",
                "Structured output validation failed: " + " | ".join(validation_errors),
                retryable=False,
                data={
                    "provider": resolved_provider,
                    "base_url": resolved_base_url,
                    "model_id": model_id,
                    "source_policy": source_policy,
                    "valid_datasets": split_result["valid_datasets"],
                    "invalid_datasets": split_result["invalid_datasets"],
                    "raw_content": cleaned_content,
                },
            )
        return _response_envelope_success(
            {
                "provider": resolved_provider,
                "base_url": resolved_base_url,
                "model_id": model_id,
                "source_policy": source_policy,
                "content": cleaned_content,
                "datasets": parsed.get("datasets", []),
                "raw_response": chat_resp,
            }
        )
    except Exception as e:
        return _response_envelope_failed(
            "SYSTEM_ERROR", f"Response parsing failed: {e}", retryable=False
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="DataSearcher client (local vLLM + Aliyun)")
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
    parser.add_argument("--verify-hf-exists", default="")
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
    source_policy_cfg = cfg.get("source_policy", DEFAULT_SOURCE_POLICY)
    source_policy = {
        "huggingface": int(source_policy_cfg.get("huggingface", DEFAULT_SOURCE_POLICY["huggingface"])),
        "github": int(source_policy_cfg.get("github", DEFAULT_SOURCE_POLICY["github"])),
    }

    provider = (
        args.provider
        or os.getenv("PROVIDER")
        or cfg.get("default_provider")
        or "local"
    )
    provider = provider.strip().lower()

    host = args.host or os.getenv("HOST") or str(local_cfg.get("host", "127.0.0.1"))
    port = (
        args.port
        if args.port >= 0
        else int(os.getenv("PORT", str(local_cfg.get("port", 8000))))
    )
    if provider == "aliyun":
        base_url = (
            args.base_url
            or os.getenv("BASE_URL")
            or str(aliyun_cfg.get("base_url", DEFAULT_ALIYUN_BASE_URL))
        )
        model = args.model or os.getenv("MODEL") or str(
            aliyun_cfg.get("model", DEFAULT_ALIYUN_MODEL)
        )
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
        base_url = (
            args.base_url
            or os.getenv("BASE_URL")
            or str(local_cfg.get("base_url", ""))
        )
        model = args.model or os.getenv("MODEL") or ""
        api_key = args.api_key or os.getenv("API_KEY") or ""

    default_prompt = _build_default_prompt(source_policy)
    cfg_prompt = str(cfg.get("prompt", "")).strip()
    cfg_prompt_template = str(cfg.get("prompt_template", "")).strip()
    cfg_prompt_file = str(cfg.get("prompt_file", "")).strip()

    prompt = args.prompt or os.getenv("PROMPT") or cfg_prompt
    if not prompt:
        try:
            template = ""
            if cfg_prompt_file:
                template = _load_text_file(cfg_prompt_file).strip()
            elif cfg_prompt_template:
                template = cfg_prompt_template
            if template:
                prompt = _render_prompt_template(template, source_policy)
        except Exception as e:
            envelope = _response_envelope_failed("VALIDATION_ERROR", str(e), retryable=False)
            print(json.dumps(envelope, ensure_ascii=False, indent=2))
            return 1
    if not prompt:
        prompt = default_prompt
    timeout_s = (
        args.timeout
        if args.timeout >= 0
        else int(os.getenv("TIMEOUT", str(req_cfg.get("timeout", 20))))
    )
    retries = (
        args.retries
        if args.retries >= 0
        else int(os.getenv("RETRIES", str(req_cfg.get("retries", 2))))
    )
    verify_hf_exists_raw = (
        args.verify_hf_exists
        or os.getenv("VERIFY_HF_EXISTS")
        or str(req_cfg.get("verify_hf_exists", True))
    )
    verify_hf_exists = str(verify_hf_exists_raw).strip().lower() in ("1", "true", "yes", "on")

    envelope = run_datasearcher(
        provider=provider,
        host=host,
        port=port,
        base_url=base_url,
        model=model,
        api_key=api_key,
        prompt=prompt,
        timeout_s=timeout_s,
        retries=retries,
        verify_hf_exists=verify_hf_exists,
        source_policy=source_policy,
    )

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(envelope, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(envelope, ensure_ascii=False, indent=2))
    return 0 if envelope["status"] == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
