"""Source policy based candidate selector."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# size_human patterns that indicate "too large" or "unknown" - exclude when matched
DEFAULT_EXCLUDE_SIZE_HUMAN = (
    "n>1t",
    "unknown",
    "100b<n",
    "1b<n",
    "100m<n",
)


def _size_mb_from_kb(size_kb: Any) -> Optional[float]:
    """Convert size_kb to size_mb, or None if invalid."""
    if size_kb is None:
        return None
    try:
        return int(size_kb) / 1024.0
    except (TypeError, ValueError):
        return None


def _size_mb_from_size_human(size_human: Any) -> Optional[float]:
    """Parse 'X KB' from size_human, return size_mb or None."""
    if not size_human or not isinstance(size_human, str):
        return None
    m = re.search(r"(\d+)\s*kb", str(size_human).strip().lower())
    if m:
        return int(m.group(1)) / 1024.0
    return None


# HF size_human row-count patterns -> midpoint rows.
_SIZE_ROW_MAP = [
    (r"n\s*<\s*1k", 500),
    (r"1k\s*<\s*n\s*<\s*10k", 5_500),
    (r"10k\s*<\s*n\s*<\s*100k", 55_000),
    (r"100k\s*<\s*n\s*<\s*1m", 550_000),
    (r"1m\s*<\s*n\s*<\s*10m", 5_500_000),
    (r"10m\s*<\s*n\s*<\s*100m", 55_000_000),
    (r"100m\s*<\s*n\s*<\s*1b", 550_000_000),
    (r"1b\s*<\s*n\s*<\s*10b", 5_500_000_000),
    (r"10b\s*<\s*n\s*<\s*100b", 55_000_000_000),
    (r"100b\s*<\s*n\s*<\s*1t", 550_000_000_000),
    (r"n\s*>\s*1t", 1_500_000_000_000),
    (r"n\s*>\s*1m", 5_500_000),
    (r"(\d+)\s*k\b", None),  # 435K -> 435000
    (r"(\d+)\s*kb", None),   # 1485 KB -> treat as disk, not rows
]
BYTES_PER_ROW_EST = 512  # 0.5KB per row for MB conversion when rows known


def _parse_single_segment(sh: str) -> Tuple[Optional[int], Optional[float]]:
    """Parse one size_human segment -> (rows, mb). Returns (None, None) if unparseable."""
    rows: Optional[int] = None
    mb: Optional[float] = None
    if not sh:
        return (None, None)
    for pat, val in _SIZE_ROW_MAP:
        if val is not None and re.search(pat, sh, re.I):
            rows = val
            break
    if rows is None:
        m = re.search(r"(\d+)\s*k\b", sh)
        if m and "kb" not in sh:
            rows = int(m.group(1)) * 1000
    if rows is None and "kb" in sh:
        m = re.search(r"(\d+)\s*kb", sh)
        if m:
            mb = int(m.group(1)) / 1024.0
    if mb is None and rows is not None:
        mb = rows * BYTES_PER_ROW_EST / (1024 * 1024)
    return (rows, round(mb, 2) if mb is not None else None)


def _parse_size_to_comparable(
    size_human: Any,
    size_mb: Any,
    size_kb: Any,
) -> Tuple[Optional[int], Optional[float]]:
    """Parse size to (size_rows_equivalent, size_comparable_mb).
    size_rows_equivalent: int for row-based HF formats; None when unknown.
    size_comparable_mb: float, unified MB for comparison. None when unknown (no fake default).
    """
    mb: Optional[float] = None
    if size_mb is not None:
        try:
            mb = float(size_mb)
        except (TypeError, ValueError):
            pass
    if mb is None and size_kb is not None:
        mb = _size_mb_from_kb(size_kb)

    sh = str(size_human or "").strip().lower() if size_human else ""
    if not sh or "unknown" in sh:
        return (None, round(mb, 2) if mb is not None else None)

    segments = [s.strip() for s in re.split(r"[,;]", sh) if s.strip()]
    if not segments:
        return (None, round(mb, 2) if mb is not None else None)

    # Parse each segment; for comma-separated (e.g. "n<1K,1K<n<10K,10K<n<100K") take largest
    best_rows: Optional[int] = None
    best_mb: Optional[float] = None
    for seg in segments:
        r, m = _parse_single_segment(seg)
        if r is not None and (best_rows is None or r > best_rows):
            best_rows = r
            best_mb = m
        elif m is not None and best_rows is None and best_mb is None:
            best_mb = m  # disk size (KB) from one segment

    rows = best_rows
    if mb is None and best_mb is not None:
        mb = best_mb
    elif mb is None and rows is not None:
        mb = rows * BYTES_PER_ROW_EST / (1024 * 1024)

    return (rows, round(mb, 2) if mb is not None else None)


def _size_human_matches_exclude(size_human: Any, exclude: Tuple[str, ...]) -> bool:
    """Return True if size_human matches any exclude pattern (union check)."""
    if not size_human or not isinstance(size_human, str):
        return False
    lower = str(size_human).strip().lower()
    if not lower:
        return False
    for pat in exclude:
        if pat and pat in lower:
            return True
    for part in re.split(r"[,;]", lower):
        part = part.strip()
        for pat in exclude:
            if pat and pat in part:
                return True
    return False


def _in_preferred_size(
    item: Dict[str, Any],
    preferred: Optional[Dict[str, Any]],
) -> bool:
    """Return True if item passes preferred_size filter (or no filter).
    Uses union of size and size_human: exclude when EITHER indicates out-of-range.
    """
    if not preferred:
        return True
    min_mb = float(preferred.get("min_mb", 0))
    max_mb = float(preferred.get("max_mb", 0)) or 10**6
    if min_mb <= 0 and max_mb >= 10**6:
        return True

    exclude_patterns = tuple(
        str(p).strip().lower()
        for p in preferred.get("exclude_size_human") or DEFAULT_EXCLUDE_SIZE_HUMAN
        if str(p).strip()
    )

    source_type = str(item.get("source_type", "")).lower()
    size_mb = item.get("size_mb")
    size_kb = item.get("size")
    size_human = item.get("size_human")

    if size_mb is None and size_kb is not None:
        size_mb = _size_mb_from_kb(size_kb)
    if size_mb is None and size_human:
        size_mb = _size_mb_from_size_human(size_human)

    if size_mb is not None:
        if size_mb < min_mb or size_mb > max_mb:
            return False

    if _size_human_matches_exclude(size_human, exclude_patterns):
        return False

    if source_type == "huggingface" and size_mb is None:
        sh = str(size_human or "").strip().lower()
        if not sh or sh == "unknown":
            return False
    return True


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
    """
    对候选数据集/仓库打分，用于排序。分值越高越靠前。

    计分规则:
    1. Intent 匹配: intent_text 中每个 token 若出现在 (dataset_name, repo_id, description, license) 中，+3.0
    2. HuggingFace: downloads * 0.00001 + likes * 0.05
    3. GitHub: stars * 0.01
    """
    text = " ".join(
        [
            str(item.get("dataset_name", "")),
            str(item.get("repo_id", "")),
            str(item.get("description", "")),
            str(item.get("license", "")),
        ]
    ).lower()
    score = 0.0

    # 1. Intent 语义匹配: 每个匹配 token +3.0
    intent_tokens = set(_tokenize(intent_text))
    for token in intent_tokens:
        if token and token in text:
            score += 3.0

    # 2. 平台热度加权
    source_type = str(item.get("source_type", "")).lower()
    if source_type == "huggingface":
        score += float(item.get("downloads") or 0) * 0.00001  # 下载量
        score += float(item.get("likes") or 0) * 0.05        # 点赞数
    if source_type == "github":
        score += float(item.get("stars") or 0) * 0.01       # 星标数
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

    brief_intro = str(item.get("description", "")).strip()[:300] or None
    size_rows, size_comparable_mb = _parse_size_to_comparable(
        item.get("size_human"),
        item.get("size_mb"),
        item.get("size"),
    )

    return {
        "dataset_name": str(item.get("dataset_name", "")).strip() or repo_id.split("/")[-1],
        "source_type": source_type,
        "repo_id": repo_id,
        "source_url": str(item.get("source_url", "")).strip(),
        "download_url": cmd,
        "download_command": cmd,
        "license": str(item.get("license", "")).strip() or "unknown",
        "reason": reason,
        "brief_intro": brief_intro,
        "verified_meta": {
            "downloads": item.get("downloads"),
            "likes": item.get("likes"),
            "stars": item.get("stars"),
            "size": item.get("size"),
            "size_human": item.get("size_human"),
            "size_mb": item.get("size_mb"),
            "size_rows_equivalent": size_rows,
            "size_comparable_mb": size_comparable_mb,
            "updated_at": item.get("updated_at"),
            "last_modified": item.get("last_modified"),
        },
    }


def _ratio_counts(total: int, source_policy: Dict[str, int]) -> Tuple[int, int]:
    """Split total by source_policy weights (percentage). Weight=0 disables that source."""
    if total <= 0:
        return 0, 0
    hf_weight = max(int(source_policy.get("huggingface", 6)), 0)
    gh_weight = max(int(source_policy.get("github", 4)), 0)
    weight_sum = hf_weight + gh_weight
    if weight_sum <= 0:
        return 0, 0
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
    preferred_size: Optional[Dict[str, Any]] = None,
    slice_download_count: int = 0,
) -> Dict[str, Any]:
    notes: List[str] = []

    recall_pool_size = max(1, int(recall_pool_size))
    download_size = max(1, int(download_size))
    slice_download_count = max(0, int(slice_download_count))

    hf_pool = _dedupe_by_repo(hf_candidates)
    gh_pool = _dedupe_by_repo(gh_candidates)

    hf_ranked = sorted(hf_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)
    gh_ranked = sorted(gh_pool, key=lambda x: _score_candidate(x, intent_text), reverse=True)

    hf_recall_need, gh_recall_need = _ratio_counts(recall_pool_size, source_policy)
    hf_recall = hf_ranked[:hf_recall_need]
    gh_recall = gh_ranked[:gh_recall_need]
    hf_recall = sorted(hf_recall, key=lambda x: (float(x.get("downloads") or 0), float(x.get("likes") or 0)), reverse=True)
    gh_recall = sorted(gh_recall, key=lambda x: float(x.get("stars") or 0), reverse=True)
    recall_rows = hf_recall + gh_recall

    if len(hf_recall) < hf_recall_need:
        notes.append(f"recall层huggingface不足: 期望{hf_recall_need}, 实得{len(hf_recall)}")
    if len(gh_recall) < gh_recall_need:
        notes.append(f"recall层github不足: 期望{gh_recall_need}, 实得{len(gh_recall)}")

    if preferred_size:
        hf_recall_filtered = [x for x in hf_recall if _in_preferred_size(x, preferred_size)]
        gh_recall_filtered = [x for x in gh_recall if _in_preferred_size(x, preferred_size)]
        dropped_hf = len(hf_recall) - len(hf_recall_filtered)
        dropped_gh = len(gh_recall) - len(gh_recall_filtered)
        if dropped_hf or dropped_gh:
            notes.append(f"download层preferred_size过滤: HF剔除{dropped_hf}, GH剔除{dropped_gh}")
        hf_recall, gh_recall = hf_recall_filtered, gh_recall_filtered

    hf_download_need, gh_download_need = _ratio_counts(download_size, source_policy)
    hf_download = hf_recall[:hf_download_need]
    gh_download = gh_recall[:gh_download_need]
    download_rows = hf_download + gh_download

    if len(hf_download) < hf_download_need:
        notes.append(f"download层huggingface不足: 期望{hf_download_need}, 实得{len(hf_download)}")
    if len(gh_download) < gh_download_need:
        notes.append(f"download层github不足: 期望{gh_download_need}, 实得{len(gh_download)}")

    # Slice download: top N from recall by rank (recall_rows already sorted by downloads/likes or stars)
    slice_rows = recall_rows[:slice_download_count] if slice_download_count > 0 else []

    return {
        "recall_pool": [
            _normalize_selected_item(item, "In recall pool.")
            for item in recall_rows
        ],
        "download_list": [
            _normalize_selected_item(item, "Selected by source policy and size filter (extract).")
            for item in download_rows
        ],
        "slice_download_list": [
            _normalize_selected_item(item, "Slice download (whitelist-only, rank from recall).")
            for item in slice_rows
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
