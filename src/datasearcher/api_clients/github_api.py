"""GitHub repository API client."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List

GITHUB_SEARCH_ENDPOINT = "https://api.github.com/search/repositories"


def _get_json(url: str, timeout_s: int = 20) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
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


def search_repositories(
    query: str,
    limit: int = 20,
    sort: str = "stars",
    order: str = "desc",
    timeout_s: int = 20,
) -> List[Dict[str, Any]]:
    q = _safe_str(query)
    if not q:
        return []

    size = max(1, min(int(limit), 100))
    params = urllib.parse.urlencode(
        {
            "q": q,
            "sort": _safe_str(sort) or "stars",
            "order": _safe_str(order) or "desc",
            "per_page": size,
            "page": 1,
        }
    )
    url = f"{GITHUB_SEARCH_ENDPOINT}?{params}"
    payload = _get_json(url, timeout_s=timeout_s)
    items = payload.get("items")
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        repo_id = _safe_str(item.get("full_name"))
        if not repo_id or "/" not in repo_id:
            continue
        license_obj = item.get("license") if isinstance(item.get("license"), dict) else {}
        license_name = _safe_str(license_obj.get("spdx_id")) or _safe_str(license_obj.get("name"))
        out.append(
            {
                "dataset_name": _safe_str(item.get("name")) or repo_id.split("/")[-1],
                "repo_id": repo_id,
                "source_type": "github",
                "source_url": _safe_str(item.get("html_url")) or f"https://github.com/{repo_id}",
                "license": license_name or "unknown",
                "stars": item.get("stargazers_count"),
                "size": item.get("size"),
                "size_human": f"{item.get('size')} KB" if item.get("size") is not None else "unknown",
                "updated_at": _safe_str(item.get("updated_at")),
                "description": _safe_str(item.get("description"))[:500],
            }
        )
    return out
