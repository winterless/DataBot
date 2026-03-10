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
    created_after: str | None = None,
) -> List[Dict[str, Any]]:
    q = _safe_str(query)
    if created_after:
        date_part = f" created:>{_safe_str(created_after)}"
        q = f"{q}{date_part}".strip() if q else f"created:>{_safe_str(created_after)}"
    if not q:
        return []

    size = max(1, min(int(limit), 100))
    params = urllib.parse.urlencode(
        {
            "q": q,
            "sort": _safe_str(sort) or "stars",
            "order": _safe_str(order) or "desc",
            "per_page": min(size, 100),
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
        size_kb = item.get("size")
        try:
            if size_kb is not None:
                size_mb = round(float(size_kb) / 1024, 2)
                if size_mb < 0:
                    size_mb = None
            else:
                size_mb = None
        except (TypeError, ValueError):
            size_mb = None
        out.append(
            {
                "dataset_name": _safe_str(item.get("name")) or repo_id.split("/")[-1],
                "repo_id": repo_id,
                "source_type": "github",
                "source_url": _safe_str(item.get("html_url")) or f"https://github.com/{repo_id}",
                "license": license_name or "unknown",
                "stars": item.get("stargazers_count"),
                "size": size_kb,
                "size_human": f"{size_kb} KB" if size_kb is not None else "unknown",
                "size_mb": size_mb,
                "updated_at": _safe_str(item.get("updated_at")),
                "description": _safe_str(item.get("description"))[:500],
            }
        )
    return out


def search_repositories_time_sweep(
    base_query: str = "dataset",
    created_after: str = "2024-06-01",
    limit: int = 150,
    timeout_s: int = 20,
) -> List[Dict[str, Any]]:
    """Time-sweep: fetch newest repos by update time (sort=updated) within date range."""
    all_items: List[Dict[str, Any]] = []
    per_page = min(100, limit)
    for page in range(1, 1 + max(1, (limit + 99) // 100)):
        q = f"{_safe_str(base_query)} created:>{_safe_str(created_after)}".strip()
        params = urllib.parse.urlencode({
            "q": q,
            "sort": "updated",
            "order": "desc",
            "per_page": per_page,
            "page": page,
        })
        url = f"{GITHUB_SEARCH_ENDPOINT}?{params}"
        payload = _get_json(url, timeout_s=timeout_s)
        items = payload.get("items")
        if not isinstance(items, list):
            break
        for item in items:
            if not isinstance(item, dict):
                continue
            repo_id = _safe_str(item.get("full_name"))
            if not repo_id or "/" not in repo_id:
                continue
            license_obj = item.get("license") if isinstance(item.get("license"), dict) else {}
            license_name = _safe_str(license_obj.get("spdx_id")) or _safe_str(license_obj.get("name"))
            size_kb = item.get("size")
            try:
                size_mb = round(float(size_kb) / 1024, 2) if size_kb is not None else None
                if size_mb is not None and size_mb < 0:
                    size_mb = None
            except (TypeError, ValueError):
                size_mb = None
            all_items.append({
                "dataset_name": _safe_str(item.get("name")) or repo_id.split("/")[-1],
                "repo_id": repo_id,
                "source_type": "github",
                "source_url": _safe_str(item.get("html_url")) or f"https://github.com/{repo_id}",
                "license": license_name or "unknown",
                "stars": item.get("stargazers_count"),
                "size": size_kb,
                "size_human": f"{size_kb} KB" if size_kb is not None else "unknown",
                "size_mb": size_mb,
                "updated_at": _safe_str(item.get("updated_at")),
                "description": _safe_str(item.get("description"))[:500],
            })
        if len(items) < per_page:
            break
        if len(all_items) >= limit:
            break
    return all_items[:limit]
