"""README link extraction for second-order repo discovery."""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from typing import Any, Dict, List, Tuple

RE_HF_DATASET = re.compile(
    r"https?://(?:www\.)?huggingface\.co/datasets/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+)",
    re.IGNORECASE,
)
RE_GH_REPO = re.compile(
    r"https?://(?:www\.)?github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_.-]+?)(?:[/?#]|$)",
    re.IGNORECASE,
)


def extract_linked_repos(content: str) -> List[Tuple[str, str]]:
    """Extract HF dataset and GitHub repo links from text. Returns [(source_type, repo_id), ...]."""
    if not content or not isinstance(content, str):
        return []
    out: List[Tuple[str, str]] = []
    seen: set = set()
    for m in RE_HF_DATASET.finditer(content):
        rid = m.group(1).strip()
        if "/" in rid and rid not in seen:
            seen.add(rid)
            out.append(("huggingface", rid))
    for m in RE_GH_REPO.finditer(content):
        rid = m.group(1).strip().rstrip("/")
        if "/" in rid and rid not in seen:
            seen.add(rid)
            out.append(("github", rid))
    return out


def fetch_github_readme(owner: str, repo: str, timeout_s: int = 15) -> str:
    """Fetch README content from GitHub API. Returns raw text or empty string."""
    url = f"https://api.github.com/repos/{owner}/{repo}/readme"
    try:
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
            data = json.loads(resp.read().decode("utf-8"))
        enc = data.get("encoding", "")
        raw = data.get("content", "")
        if enc == "base64" and raw:
            return base64.b64decode(raw).decode("utf-8", errors="replace")
        return raw if isinstance(raw, str) else ""
    except Exception:
        return ""


def fetch_hf_dataset_readme(repo_id: str, timeout_s: int = 15) -> str:
    """Fetch README from HuggingFace dataset repo. Returns raw text or empty string."""
    url = f"https://huggingface.co/datasets/{repo_id}/raw/main/README.md"
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "DataBot-DataSearcher"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return ""
