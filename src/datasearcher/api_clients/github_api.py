"""GitHub repository API client."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, List

GITHUB_SEARCH_ENDPOINT = "https://api.github.com/search/repositories"
GITHUB_CODE_SEARCH_ENDPOINT = "https://api.github.com/search/code"
IN_QUALIFIER = " in:name,description,readme"


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


def _ensure_in_qualifier(q: str) -> str:
    """Append in:name,description,readme if not already present."""
    if not q:
        return q
    lower = q.lower()
    if " in:name" in lower or " in:description" in lower or " in:readme" in lower:
        return q
    return f"{q}{IN_QUALIFIER}".strip()


def _normalize_gh_item(item: Dict[str, Any]) -> Dict[str, Any] | None:
    """Convert raw GitHub API item to unified candidate dict."""
    if not isinstance(item, dict):
        return None
    repo_id = _safe_str(item.get("full_name"))
    if not repo_id or "/" not in repo_id:
        return None
    license_obj = item.get("license") if isinstance(item.get("license"), dict) else {}
    license_name = _safe_str(license_obj.get("spdx_id")) or _safe_str(license_obj.get("name"))
    size_kb = item.get("size")
    try:
        size_mb = round(float(size_kb) / 1024, 2) if size_kb is not None else None
        if size_mb is not None and size_mb < 0:
            size_mb = None
    except (TypeError, ValueError):
        size_mb = None
    return {
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


def search_repositories(
    query: str,
    limit: int = 20,
    sort: str = "stars",
    order: str = "desc",
    timeout_s: int = 20,
    created_after: str | None = None,
    max_pages: int = 1,
    full_field: bool = True,
) -> List[Dict[str, Any]]:
    q = _safe_str(query)
    if created_after:
        date_part = f" created:>{_safe_str(created_after)}"
        q = f"{q}{date_part}".strip() if q else f"created:>{_safe_str(created_after)}"
    if full_field:
        q = _ensure_in_qualifier(q)
    if not q:
        return []

    out: List[Dict[str, Any]] = []
    per_page = min(100, max(1, int(limit)))
    pages = min(max(1, int(max_pages)), 10)
    for page in range(1, pages + 1):
        params = urllib.parse.urlencode(
            {
                "q": q,
                "sort": _safe_str(sort) or "stars",
                "order": _safe_str(order) or "desc",
                "per_page": per_page,
                "page": page,
            }
        )
        url = f"{GITHUB_SEARCH_ENDPOINT}?{params}"
        payload = _get_json(url, timeout_s=timeout_s)
        items = payload.get("items")
        if not isinstance(items, list):
            break
        for item in items:
            row = _normalize_gh_item(item)
            if row:
                out.append(row)
        if len(items) < per_page:
            break
        if len(out) >= limit:
            break
    return out[:limit]


def _search_repositories_legacy_items(
    query: str,
    limit: int,
    sort: str,
    order: str,
    timeout_s: int,
    created_after: str | None,
) -> List[Dict[str, Any]]:
    """Internal: fetch raw items for backward compat. Used by time_sweep."""
    q = _safe_str(query)
    if created_after:
        date_part = f" created:>{_safe_str(created_after)}"
        q = f"{q}{date_part}".strip() if q else f"created:>{_safe_str(created_after)}"
    if not q:
        return []
    params = urllib.parse.urlencode({
        "q": q,
        "sort": _safe_str(sort) or "stars",
        "order": _safe_str(order) or "desc",
        "per_page": min(100, limit),
        "page": 1,
    })
    url = f"{GITHUB_SEARCH_ENDPOINT}?{params}"
    payload = _get_json(url, timeout_s=timeout_s)
    items = payload.get("items")
    return items if isinstance(items, list) else []


def search_code_for_data_repos(
    limit: int = 100,
    timeout_s: int = 20,
    domain_keywords: List[str] | None = None,
) -> List[Dict[str, Any]]:
    """Search repos containing data/ dir with jsonl or parquet, scoped to domain keywords.
    When domain_keywords provided (e.g. ['agent', 'function call']), queries become
    'agent path:data jsonl' etc. to target Agent/Function Call datasets.
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    keywords = [k for k in (domain_keywords or []) if k and len(str(k).strip()) >= 2]
    formats = ["jsonl", "parquet"]
    if keywords:
        base_queries = [f"{kw} path:data {fmt}" for kw in keywords[:3] for fmt in formats]
    else:
        base_queries = ["path:data jsonl", "path:data parquet"]
    for code_query in base_queries[:6]:
        try:
            params = urllib.parse.urlencode({
                "q": code_query,
                "per_page": 30,
                "page": 1,
            })
            url = f"{GITHUB_CODE_SEARCH_ENDPOINT}?{params}"
            payload = _get_json(url, timeout_s=timeout_s)
            items = payload.get("items")
            if not isinstance(items, list):
                continue
            for hit in items:
                repo = hit.get("repository") if isinstance(hit.get("repository"), dict) else {}
                full_name = _safe_str(repo.get("full_name"))
                if not full_name or "/" not in full_name or full_name in seen:
                    continue
                seen.add(full_name)
                out.append({
                    "dataset_name": full_name.split("/")[-1],
                    "repo_id": full_name,
                    "source_type": "github",
                    "source_url": f"https://github.com/{full_name}",
                    "license": "unknown",
                    "stars": repo.get("stargazers_count"),
                    "size": None,
                    "size_human": "unknown",
                    "size_mb": None,
                    "updated_at": "",
                    "description": _safe_str(repo.get("description"))[:500],
                })
                if len(out) >= limit:
                    return out
        except Exception:
            continue
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
