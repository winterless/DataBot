"""Source policy based candidate selector."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _sanitize_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def canonical_hf_download_command(repo_id: str) -> str:
    local_name = _sanitize_local_name(repo_id)
    return (
        f"huggingface-cli download --repo-type dataset {repo_id} "
        f"--local-dir ./data/hf__{local_name}"
    )


def canonical_github_download_command(repo_id: str) -> str:
    local_name = _sanitize_local_name(repo_id)
    return f"git clone https://github.com/{repo_id}.git ./data/gh__{local_name}"


def _tokenize(text: str) -> List[str]:
    return [x for x in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if x]


def _score_candidate(item: Dict[str, Any], intent_text: str) -> float:
    text = " ".join(
        [
            str(item.get("dataset_name", "")),
            str(item.get("repo_id", "")),
            str(item.get("description", "")),
            str(item.get("license", "")),
        ]
    ).lower()
    score = 0.0

    intent_tokens = set(_tokenize(intent_text))
    for token in intent_tokens:
        if token and token in text:
            score += 3.0

    source_type = str(item.get("source_type", "")).lower()
    if source_type == "huggingface":
        score += float(item.get("downloads") or 0) * 0.00001
        score += float(item.get("likes") or 0) * 0.05
    if source_type == "github":
        score += float(item.get("stars") or 0) * 0.01
    return score


def _dedupe_by_repo(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in items:
        key = (str(item.get("source_type", "")).lower(), str(item.get("repo_id", "")).strip())
        if not key[1] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _normalize_selected_item(item: Dict[str, Any], reason: str) -> Dict[str, Any]:
    source_type = str(item.get("source_type", "")).lower().strip()
    repo_id = str(item.get("repo_id", "")).strip()
    if source_type == "huggingface":
        cmd = canonical_hf_download_command(repo_id)
    elif source_type == "github":
        cmd = canonical_github_download_command(repo_id)
    else:
        cmd = ""

    return {
        "dataset_name": str(item.get("dataset_name", "")).strip() or repo_id.split("/")[-1],
        "source_type": source_type,
        "repo_id": repo_id,
        "source_url": str(item.get("source_url", "")).strip(),
        "download_url": cmd,
        "download_command": cmd,
        "license": str(item.get("license", "")).strip() or "unknown",
        "reason": reason,
        "verified_meta": {
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "stars": item.get("stars"),
            "size": item.get("size"),
            "size_human": item.get("size_human"),
            "updated_at": item.get("updated_at"),
            "last_modified": item.get("last_modified"),
        },
    }


def _ratio_counts(total: int, source_policy: Dict[str, int]) -> Tuple[int, int]:
    if total <= 0:
        return 0, 0
    hf_weight = max(int(source_policy.get("huggingface", 6)), 0)
    gh_weight = max(int(source_policy.get("github", 4)), 0)
    weight_sum = hf_weight + gh_weight
    if weight_sum <= 0:
        return total, 0
    hf_target = int(round(total * hf_weight / weight_sum))
    hf_target = max(0, min(hf_target, total))
    gh_target = total - hf_target
    return hf_target, gh_target


def select_candidates_two_layer(
    hf_candidates: List[Dict[str, Any]],
    gh_candidates: List[Dict[str, Any]],
    source_policy: Dict[str, int],
    intent_text: str,
    recall_pool_size: int,
    download_size: int,
) -> Dict[str, Any]:
    notes: List[str] = []

    recall_pool_size = max(1, int(recall_pool_size))
    download_size = max(1, int(download_size))

    hf_pool = _dedupe_by_repo(hf_candidates)
    gh_pool = _dedupe_by_repo(gh_candidates)
    hf_ranked = sorted(hf_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)
    gh_ranked = sorted(gh_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)

    hf_recall_need, gh_recall_need = _ratio_counts(recall_pool_size, source_policy)
    hf_recall = hf_ranked[:hf_recall_need]
    gh_recall = gh_ranked[:gh_recall_need]
    recall_rows = hf_recall + gh_recall

    if len(hf_recall) < hf_recall_need:
        notes.append(f"recall层huggingface不足: 期望{hf_recall_need}, 实得{len(hf_recall)}")
    if len(gh_recall) < gh_recall_need:
        notes.append(f"recall层github不足: 期望{gh_recall_need}, 实得{len(gh_recall)}")

    hf_download_need, gh_download_need = _ratio_counts(download_size, source_policy)
    hf_download = hf_recall[:hf_download_need]
    gh_download = gh_recall[:gh_download_need]
    download_rows = hf_download + gh_download

    if len(hf_download) < hf_download_need:
        notes.append(f"download层huggingface不足: 期望{hf_download_need}, 实得{len(hf_download)}")
    if len(gh_download) < gh_download_need:
        notes.append(f"download层github不足: 期望{gh_download_need}, 实得{len(gh_download)}")

    return {
        "recall_pool": [
            _normalize_selected_item(item, "In recall pool.")
            for item in recall_rows
        ],
        "download_list": [
            _normalize_selected_item(item, "Selected for download by semantic relevance and source policy.")
            for item in download_rows
        ],
        "notes": notes,
    }


def select_candidates_by_policy(
    hf_candidates: List[Dict[str, Any]],
    gh_candidates: List[Dict[str, Any]],
    source_policy: Dict[str, int],
    intent_text: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    hf_need = int(source_policy.get("huggingface", 6))
    gh_need = int(source_policy.get("github", 4))
    notes: List[str] = []

    hf_pool = _dedupe_by_repo(hf_candidates)
    gh_pool = _dedupe_by_repo(gh_candidates)

    hf_ranked = sorted(hf_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)
    gh_ranked = sorted(gh_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)

    selected_hf = hf_ranked[:hf_need]
    selected_gh = gh_ranked[:gh_need]

    if len(selected_hf) < hf_need:
        notes.append(f"huggingface不足: 期望{hf_need}, 实得{len(selected_hf)}")
    if len(selected_gh) < gh_need:
        notes.append(f"github不足: 期望{gh_need}, 实得{len(selected_gh)}")

    out: List[Dict[str, Any]] = []
    out.extend(
        _normalize_selected_item(item, "Selected by source_policy and semantic relevance.")
        for item in selected_hf
    )
    out.extend(
        _normalize_selected_item(item, "Selected by source_policy and semantic relevance.")
        for item in selected_gh
    )
    return out, notes
