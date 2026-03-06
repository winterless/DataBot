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
                            "description": "Search keywords (domain/language/task/license hints).",
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
                            "description": "GitHub repository query syntax keywords.",
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
                    },
                    "required": ["query"],
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


def execute_tool_call(name: str, arguments: Any, timeout_s: int = 20) -> Dict[str, Any]:
    args = _parse_arguments(arguments)
    query = str(args.get("query", "")).strip()
    limit = int(args.get("limit", 20)) if args.get("limit") is not None else 20

    if name == TOOL_SEARCH_HF:
        rows = search_datasets(query=query, limit=limit, timeout_s=timeout_s)
        return {"tool_name": name, "query": query, "count": len(rows), "candidates": rows}
    if name == TOOL_SEARCH_GH:
        sort = str(args.get("sort", "stars")).strip() or "stars"
        order = str(args.get("order", "desc")).strip() or "desc"
        rows = search_repositories(
            query=query,
            limit=limit,
            sort=sort,
            order=order,
            timeout_s=timeout_s,
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
