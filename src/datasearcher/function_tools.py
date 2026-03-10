"""Function-calling tools for live source retrieval."""

from __future__ import annotations

import json
from typing import Any, Dict, List

try:
    from .api_clients.github_api import search_repositories
    from .api_clients.huggingface_api import search_datasets
except ImportError:  # pragma: no cover - direct script execution fallback
    from api_clients.github_api import search_repositories
    from api_clients.huggingface_api import search_datasets


TOOL_SEARCH_HF = "search_huggingface_datasets"
TOOL_SEARCH_GH = "search_github_repositories"
TOOL_SET_DISCOVERY_PARAMS = "set_discovery_parameters"


def get_tool_schemas() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_HF,
                "description": "Search public HuggingFace datasets by query keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keywords. Do NOT wrap in double quotes; use plain keywords to match hyphenated names.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many candidates to retrieve.",
                            "minimum": 1,
                            "maximum": 100,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_GH,
                "description": "Search public GitHub repositories by query keywords.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "GitHub search keywords. Do NOT wrap in double quotes.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "How many candidates to retrieve.",
                            "minimum": 1,
                            "maximum": 100,
                        },
                        "sort": {
                            "type": "string",
                            "description": "GitHub sort field, e.g. stars, updated.",
                        },
                        "order": {
                            "type": "string",
                            "description": "Sort order: desc or asc.",
                        },
                        "max_pages": {
                            "type": "integer",
                            "description": "Pages to fetch (1-5). Use 5 for core keywords to increase recall.",
                            "minimum": 1,
                            "maximum": 5,
                        },
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_SET_DISCOVERY_PARAMS,
                "description": "Set dynamic discovery parameters based on the query domain. Call this (optionally) to enable org-based scan and time filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dynamic_seed_orgs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "3-5 authoritative orgs/authors in this domain for HuggingFace author-based queries. Empty if not applicable.",
                        },
                        "suggested_created_after": {
                            "type": ["string", "null"],
                            "description": "ISO date (YYYY-MM-DD) for GitHub created:> filter if the tech is recent; null if no time filter needed.",
                        },
                    },
                    "required": ["dynamic_seed_orgs", "suggested_created_after"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def _parse_arguments(arguments: Any) -> Dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        return json.loads(text)
    return {}


def _strip_query_quotes(q: str) -> str:
    """Remove surrounding double quotes to avoid strict phrase match; improves recall for hyphenated names."""
    s = str(q).strip()
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        return s[1:-1].strip()
    return s


def execute_tool_call(name: str, arguments: Any, timeout_s: int = 20) -> Dict[str, Any]:
    args = _parse_arguments(arguments)

    if name == TOOL_SET_DISCOVERY_PARAMS:
        orgs = args.get("dynamic_seed_orgs")
        if isinstance(orgs, list):
            orgs = [str(x).strip() for x in orgs if str(x).strip()]
        else:
            orgs = []
        created = args.get("suggested_created_after")
        if created is not None and not isinstance(created, str):
            created = None
        elif isinstance(created, str):
            created = created.strip() or None
        return {
            "tool_name": name,
            "dynamic_seed_orgs": orgs,
            "suggested_created_after": created,
        }

    query = _strip_query_quotes(str(args.get("query", "")).strip())
    limit = int(args.get("limit", 20)) if args.get("limit") is not None else 20

    if name == TOOL_SEARCH_HF:
        rows = search_datasets(query=query, limit=limit, timeout_s=timeout_s)
        return {"tool_name": name, "query": query, "count": len(rows), "candidates": rows}
    if name == TOOL_SEARCH_GH:
        sort = str(args.get("sort", "stars")).strip() or "stars"
        order = str(args.get("order", "desc")).strip() or "desc"
        max_pages = max(1, min(int(args.get("max_pages", 1)), 10))
        rows = search_repositories(
            query=query,
            limit=max(limit, 100) if max_pages > 1 else limit,
            sort=sort,
            order=order,
            timeout_s=timeout_s,
            max_pages=max_pages,
        )
        return {
            "tool_name": name,
            "query": query,
            "sort": sort,
            "order": order,
            "count": len(rows),
            "candidates": rows,
        }
    raise ValueError(f"Unsupported tool name: {name}")
