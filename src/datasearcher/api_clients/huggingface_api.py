"""HuggingFace dataset API client."""

from __future__ import annotations

import json
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


def search_datasets(query: str, limit: int = 20, timeout_s: int = 20) -> List[Dict[str, Any]]:
    q = _safe_str(query)
    if not q:
        return []

    size = max(1, min(int(limit), 100))
    params = urllib.parse.urlencode({"search": q, "limit": size, "full": "true"})
    url = f"{HF_DATASET_SEARCH_ENDPOINT}?{params}"
    payload = _get_json(url, timeout_s=timeout_s)
    if not isinstance(payload, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        repo_id = _safe_str(item.get("id"))
        if not repo_id:
            continue
        card_data = item.get("cardData") if isinstance(item.get("cardData"), dict) else {}
        license_name = _safe_str(card_data.get("license")) or _safe_str(item.get("license"))
        out.append(
            {
                "dataset_name": repo_id.split("/")[-1],
                "repo_id": repo_id,
                "source_type": "huggingface",
                "source_url": f"https://huggingface.co/datasets/{repo_id}",
                "license": license_name or "unknown",
                "downloads": item.get("downloads"),
                "likes": item.get("likes"),
                "last_modified": _safe_str(item.get("lastModified")),
                "description": _safe_str(item.get("description"))[:500],
            }
        )
    return out
