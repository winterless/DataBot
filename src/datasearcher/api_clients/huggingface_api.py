"""HuggingFace dataset API client."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Dict, List

HF_DATASET_SEARCH_ENDPOINT = "https://huggingface.co/api/datasets"


def _get_json(url: str, timeout_s: int = 20) -> Any:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "DataBot-DataSearcher",
        },
        method="GET",
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_hf_size(card_data: Dict[str, Any]) -> str:
    size_val = card_data.get("dataset_size")
    if isinstance(size_val, str) and size_val.strip():
        return size_val.strip()
    categories = card_data.get("size_categories")
    if isinstance(categories, list):
        names = [str(x).strip() for x in categories if str(x).strip()]
        if names:
            return ",".join(names[:3])
    return ""


def _extract_hf_size_mb(item: Dict[str, Any]) -> float | None:
    """Extract size in MB from HF item (full=true). Returns None if unavailable.
    Priority: dataset_info.download_size > dataset_info.dataset_size > dataset_info.size_in_bytes.
    Fallback: parse cardData.dataset_size / size_categories when human-readable (e.g. '100 MB').
    """
    try:
        ds_info = item.get("dataset_info") or {}
        if isinstance(ds_info, list) and ds_info and isinstance(ds_info[0], dict):
            ds_info = ds_info[0]
        if isinstance(ds_info, dict):
            bytes_val = (
                ds_info.get("download_size")
                or ds_info.get("dataset_size")
                or ds_info.get("size_in_bytes")
            )
            if bytes_val is not None:
                b = int(bytes_val)
                if b > 0:
                    return round(b / (1024 * 1024), 2)
        card = item.get("cardData") or {}
        if isinstance(card, dict):
            sh = card.get("dataset_size") or ""
            if isinstance(sh, str) and sh.strip():
                m = re.search(r"(\d+(?:\.\d+)?)\s*mb", sh.strip().lower())
                if m:
                    return round(float(m.group(1)), 2)
                m = re.search(r"(\d+(?:\.\d+)?)\s*kb", sh.strip().lower())
                if m:
                    return round(float(m.group(1)) / 1024, 2)
        return None
    except (TypeError, ValueError):
        return None


def _search_datasets_single(
    query: str,
    limit: int,
    timeout_s: int,
    sort: str = "downloads",
    direction: str = "-1",
    author: str | None = None,
) -> List[Dict[str, Any]]:
    """Single API call. HF /api/datasets returns 0 for multi-word search.
    Use author=org to list datasets by organization (ignores query).
    """
    params: Dict[str, str] = {
        "limit": str(max(1, min(limit, 100))),
        "full": "true",
        "sort": sort,
        "direction": direction,
    }
    if author:
        params["author"] = _safe_str(author)
    else:
        q = _safe_str(query)
        if not q:
            return []
        params["search"] = q
    url = f"{HF_DATASET_SEARCH_ENDPOINT}?{urllib.parse.urlencode(params)}"
    payload = _get_json(url, timeout_s=timeout_s)
    return payload if isinstance(payload, list) else []


def _tokenize_query(query: str) -> List[str]:
    """Extract searchable keywords (len>=3, not stopwords)."""
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "into", "your", "data", "dataset"}
    tokens = [t for t in re.split(r"[^a-zA-Z0-9_-]+", query.lower()) if len(t) >= 3 and t not in stopwords]
    seen = []
    for t in tokens:
        if t not in seen:
            seen.append(t)
    return seen


def search_datasets(query: str, limit: int = 20, timeout_s: int = 20) -> List[Dict[str, Any]]:
    """Search HF datasets. Splits multi-word query into keywords because HF API returns 0 for multi-word search."""
    q = _safe_str(query)
    if not q:
        return []

    size = max(1, min(int(limit), 100))
    payload = _search_datasets_single(q, size, timeout_s, sort="downloads", direction="-1")

    if not payload and " " in q:
        tokens = _tokenize_query(q)
        if tokens:
            seen_ids: set = set()
            out_raw: List[Dict[str, Any]] = []
            per_token = max(size // len(tokens), 20)
            for token in tokens[:5]:
                sub = _search_datasets_single(token, per_token, timeout_s, sort="downloads", direction="-1")
                for item in sub:
                    if isinstance(item, dict):
                        rid = _safe_str(item.get("id"))
                        if rid and rid not in seen_ids:
                            seen_ids.add(rid)
                            out_raw.append(item)
            payload = out_raw[:size]

    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = _safe_str(item.get("id"))
        if not repo_id:
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        license_name = _safe_str(card_data.get("license")) or _safe_str(item.get("license"))
        size_human = _extract_hf_size(card_data)
        size_mb = _extract_hf_size_mb(item)
        out.append(
            {
                "dataset_name": repo_id.split("/")[-1],
                "repo_id": repo_id,
                "source_type": "huggingface",
                "source_url": f"https://huggingface.co/datasets/{repo_id}",
                "license": license_name or "unknown",
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "size": None,
                "size_human": size_human or "unknown",
                "size_mb": round(size_mb, 2) if size_mb is not None else None,
                "last_modified": _safe_str(item.get("lastModified")),
                "description": _safe_str(item.get("description"))[:500],
            }
        )
    out.sort(key=lambda x: (float(x.get("downloads") or 0), float(x.get("likes") or 0)), reverse=True)
    return out


def search_datasets_recent(
    query: str = "dataset",
    limit: int = 100,
    timeout_s: int = 20,
) -> List[Dict[str, Any]]:
    """Time-sweep: fetch newest datasets by creation time (sort=createdAt, direction=-1)."""
    q = _safe_str(query) or "dataset"
    size = max(1, min(int(limit), 100))
    payload = _search_datasets_single(q, size, timeout_s, sort="createdAt", direction="-1")
    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = _safe_str(item.get("id"))
        if not repo_id:
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        license_name = _safe_str(card_data.get("license")) or _safe_str(item.get("license"))
        size_human = _extract_hf_size(card_data)
        size_mb = _extract_hf_size_mb(item)
        out.append({
            "dataset_name": repo_id.split("/")[-1],
            "repo_id": repo_id,
            "source_type": "huggingface",
            "source_url": f"https://huggingface.co/datasets/{repo_id}",
            "license": license_name or "unknown",
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "size": None,
            "size_human": size_human or "unknown",
            "size_mb": round(size_mb, 2) if size_mb is not None else None,
            "last_modified": _safe_str(item.get("lastModified")),
            "description": _safe_str(item.get("description"))[:500],
        })
    return out


def list_datasets_by_author(
    author: str,
    limit: int = 100,
    timeout_s: int = 20,
) -> List[Dict[str, Any]]:
    """Seed org scraper: list all datasets by organization/author."""
    org = _safe_str(author)
    if not org:
        return []
    size = max(1, min(int(limit), 100))
    payload = _search_datasets_single("", size, timeout_s, author=org)
    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = _safe_str(item.get("id"))
        if not repo_id:
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        license_name = _safe_str(card_data.get("license")) or _safe_str(item.get("license"))
        size_human = _extract_hf_size(card_data)
        size_mb = _extract_hf_size_mb(item)
        out.append({
            "dataset_name": repo_id.split("/")[-1],
            "repo_id": repo_id,
            "source_type": "huggingface",
            "source_url": f"https://huggingface.co/datasets/{repo_id}",
            "license": license_name or "unknown",
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "size": None,
            "size_human": size_human or "unknown",
            "size_mb": round(size_mb, 2) if size_mb is not None else None,
            "last_modified": _safe_str(item.get("lastModified")),
            "description": _safe_str(item.get("description"))[:500],
        })
    return out
