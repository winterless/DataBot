"""
Agentic SFT adaptor for large pretty-printed JSON shards.

This module is designed for dataset directories like:
`data/hf__stepfun-ai__Step-3.5-Flash-SFT/json/general/chunk_*.json`

Key goals:
- stream parse huge top-level JSON arrays without loading whole files
- keep only SFT-friendly agentic trajectories
- optionally strip reasoning traces and redact basic PII
- emit normalized JSONL-ready records with lightweight metadata
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d\-\s().]{7,}\d)(?!\w)")
DATEISH_RE = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{1,2})?$")
SHELL_CODEBLOCK_RE = re.compile(r"```(?:bash|sh|shell)\n.+?```", re.DOTALL | re.IGNORECASE)


@dataclass
class AgenticFilterConfig:
    min_turns: Optional[int] = None
    max_turns: Optional[int] = None
    min_closed_tool_loops: int = 0
    max_tool_content_chars: Optional[int] = None
    max_sample_chars: Optional[int] = 32000
    require_final_assistant: bool = False
    require_declared_tools_or_system: bool = False
    strip_reasoning_content: bool = True
    redact_pii: bool = True


@dataclass
class AgenticStats:
    files_seen: int = 0
    objects_seen: int = 0
    kept: int = 0
    rejected: int = 0
    parse_errors: int = 0
    redacted_records: int = 0


@dataclass
class ParsedArrayObject:
    obj: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def iter_input_files(input_path: Path, glob_pattern: str = "chunk_*.json") -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return sorted(p for p in input_path.glob(glob_pattern) if p.is_file())


def iter_json_array_objects(path: Path, chunk_size: int = 1 << 20) -> Iterator[ParsedArrayObject]:
    """
    Stream objects from a file whose top-level structure is a JSON array.

    The implementation assumes array items are JSON objects and uses a small
    state machine to avoid materializing the whole file in memory.

    Invalid objects are reported as parse-error events so callers can count the
    failure and keep scanning the remainder of the shard.
    """
    in_array = False
    collecting = False
    depth = 0
    in_string = False
    escaped = False
    obj_chars: List[str] = []

    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            for ch in chunk:
                if not in_array:
                    if ch == "[":
                        in_array = True
                    continue

                if not collecting:
                    if ch == "{":
                        collecting = True
                        depth = 1
                        in_string = False
                        escaped = False
                        obj_chars = ["{"]
                    elif ch == "]":
                        return
                    continue

                obj_chars.append(ch)
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        payload = "".join(obj_chars)
                        collecting = False
                        obj_chars = []
                        try:
                            yield ParsedArrayObject(obj=json.loads(payload))
                        except json.JSONDecodeError as exc:
                            yield ParsedArrayObject(error=str(exc))

    if collecting and obj_chars:
        yield ParsedArrayObject(error="Unexpected EOF while parsing JSON object")


def summarize_signals(conversations: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    roles = [str(turn.get("role", "")).strip() for turn in conversations]
    tool_turns = 0
    tool_decl_turns = 0
    assistant_tool_call_turns = 0
    system_turns = 0
    tool_payload_max_chars = 0
    shell_observation_turns = 0
    assistant_shell_action_turns = 0

    for idx, turn in enumerate(conversations):
        role = str(turn.get("role", "")).strip()
        content = str(turn.get("content", ""))
        if role == "tool":
            tool_turns += 1
            tool_payload_max_chars = max(tool_payload_max_chars, len(content))
        if role == "system":
            system_turns += 1
            if _is_system_observation_text(content):
                shell_observation_turns += 1
        if isinstance(turn.get("tools"), list) and turn["tools"]:
            tool_decl_turns += 1
        if role == "assistant" and isinstance(turn.get("tool_calls"), list) and turn["tool_calls"]:
            assistant_tool_call_turns += 1
        next_turn = conversations[idx + 1] if idx + 1 < len(conversations) else None
        next_is_observation = bool(
            next_turn is not None
            and str(next_turn.get("role", "")).strip() == "system"
            and _is_system_observation_text(str(next_turn.get("content", "")))
        )
        if role == "assistant" and (_looks_like_shell_action_text(content) or next_is_observation):
            assistant_shell_action_turns += 1

    closed_tool_loops = count_closed_tool_loops(conversations)
    closed_shell_loops = count_closed_shell_loops(conversations)
    return {
        "turn_count": len(conversations),
        "roles": roles,
        "tool_turns": tool_turns,
        "tool_declaration_turns": tool_decl_turns,
        "assistant_tool_call_turns": assistant_tool_call_turns,
        "system_turns": system_turns,
        "closed_tool_loops": closed_tool_loops,
        "shell_observation_turns": shell_observation_turns,
        "assistant_shell_action_turns": assistant_shell_action_turns,
        "closed_shell_loops": closed_shell_loops,
        "max_tool_content_chars": tool_payload_max_chars,
    }


def _has_substantive_assistant_text(conversations: Sequence[Dict[str, Any]]) -> bool:
    for turn in reversed(conversations):
        if str(turn.get("role", "")).strip() != "assistant":
            continue
        content = str(turn.get("content", "")).strip()
        if content:
            return True
    return False


def is_intentional_no_call_candidate(
    conversations: Sequence[Dict[str, Any]],
    signals: Dict[str, Any],
) -> bool:
    """
    Heuristic for BFCL-like "tool available but do not call" samples.

    We treat a sample as a candidate when:
    - tools are declared in-context
    - no explicit tool or shell interaction actually happened
    - there is at least one assistant reply with substantive text
    """
    has_any_interaction = (
        signals.get("assistant_tool_call_turns", 0) > 0
        or signals.get("tool_turns", 0) > 0
        or signals.get("assistant_shell_action_turns", 0) > 0
        or signals.get("shell_observation_turns", 0) > 0
    )
    if has_any_interaction:
        return False
    if signals.get("tool_declaration_turns", 0) <= 0:
        return False
    return _has_substantive_assistant_text(conversations)


def estimate_sample_chars(
    conversations: Sequence[Dict[str, Any]],
    cfg: AgenticFilterConfig,
) -> int:
    """
    Estimate the final per-sample character footprint after normalization.

    This uses the post-normalization `conversations` JSON payload, which is a
    closer proxy to training-context length than raw source text.
    """
    normalized_conversations, _ = normalize_conversation(conversations, cfg)
    payload = {"conversations": normalized_conversations}
    return len(json.dumps(payload, ensure_ascii=False))


def count_closed_tool_loops(conversations: Sequence[Dict[str, Any]]) -> int:
    """
    Count assistant(tool_calls) -> tool -> assistant closed loops.

    If tool_call_id exists, it must match one of the requested tool call ids.
    """
    loops = 0
    for i, turn in enumerate(conversations[:-2]):
        role = str(turn.get("role", "")).strip()
        tool_calls = turn.get("tool_calls")
        if role != "assistant" or not isinstance(tool_calls, list) or not tool_calls:
            continue

        requested_ids = {
            str(tc.get("id", "")).strip()
            for tc in tool_calls
            if isinstance(tc, dict) and str(tc.get("id", "")).strip()
        }
        saw_matching_tool = False

        for next_turn in conversations[i + 1 :]:
            next_role = str(next_turn.get("role", "")).strip()
            if next_role == "tool":
                tool_call_id = str(next_turn.get("tool_call_id", "")).strip()
                if requested_ids:
                    if tool_call_id in requested_ids:
                        saw_matching_tool = True
                else:
                    saw_matching_tool = True
                continue

            if next_role == "assistant":
                if saw_matching_tool:
                    loops += 1
                break

            if next_role == "user" and saw_matching_tool:
                break

    return loops


def count_closed_shell_loops(conversations: Sequence[Dict[str, Any]]) -> int:
    """
    Count assistant(shell action) -> system(observation) -> assistant loops.
    """
    loops = 0
    for i, turn in enumerate(conversations[:-2]):
        if str(turn.get("role", "")).strip() != "assistant":
            continue
        next_turn = conversations[i + 1]
        if not (
            _looks_like_shell_action_text(str(turn.get("content", "")))
            or (
                str(next_turn.get("role", "")).strip() == "system"
                and _is_system_observation_text(str(next_turn.get("content", "")))
            )
        ):
            continue

        saw_observation = False
        for next_turn in conversations[i + 1 :]:
            next_role = str(next_turn.get("role", "")).strip()
            next_content = str(next_turn.get("content", ""))
            if next_role == "system" and _is_system_observation_text(next_content):
                saw_observation = True
                continue
            if next_role == "assistant":
                if saw_observation:
                    loops += 1
                break
            if next_role == "user" and saw_observation:
                break
    return loops


def _looks_like_shell_action_text(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    if SHELL_CODEBLOCK_RE.search(content):
        return True
    if stripped.startswith("<response>") and ("```bash" in content or "```sh" in content):
        return True
    if stripped.startswith("THOUGHT:") and ("```bash" in content or "```sh" in content):
        return True
    if stripped.startswith("$ "):
        return True
    return False


def _is_system_observation_text(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    markers = (
        "<returncode>",
        "<output>",
        "<warning>",
        "$ ",
        "[Current working directory:",
        "Environment initialized.",
    )
    return any(marker in stripped for marker in markers)


def detect_basic_pii(conversations: Sequence[Dict[str, Any]]) -> bool:
    for turn in conversations:
        for key in ("content", "reasoning_content"):
            value = turn.get(key)
            if not isinstance(value, str):
                continue
            if EMAIL_RE.search(value) or _contains_phone_like_pii(value):
                return True
    return False


def _contains_phone_like_pii(value: str) -> bool:
    for match in PHONE_RE.finditer(value):
        if _is_redactable_phone(match.group(0)):
            return True
    return False


def _redact_text(value: str) -> Tuple[str, bool]:
    updated = EMAIL_RE.sub("[REDACTED_EMAIL]", value)
    updated = PHONE_RE.sub(_phone_replacer, updated)
    return updated, updated != value


def _phone_replacer(match: re.Match[str]) -> str:
    if _is_redactable_phone(match.group(0)):
        return "[REDACTED_PHONE]"
    return match.group(0)


def _is_redactable_phone(raw: str) -> bool:
    candidate = raw.strip()
    digits = re.sub(r"\D", "", candidate)
    if ":" in candidate:
        return False
    if DATEISH_RE.fullmatch(candidate):
        return False
    return 10 <= len(digits) <= 15


def normalize_conversation(
    conversations: Sequence[Dict[str, Any]],
    cfg: AgenticFilterConfig,
) -> Tuple[List[Dict[str, Any]], bool]:
    normalized: List[Dict[str, Any]] = []
    did_redact = False

    for turn in conversations:
        new_turn = dict(turn)
        if cfg.strip_reasoning_content:
            new_turn.pop("reasoning_content", None)
        if cfg.redact_pii:
            for key in ("content", "reasoning_content"):
                value = new_turn.get(key)
                if isinstance(value, str) and value:
                    new_value, changed = _redact_text(value)
                    new_turn[key] = new_value
                    did_redact = did_redact or changed
        normalized.append(new_turn)

    return normalized, did_redact


def evaluate_agentic_record(
    obj: Dict[str, Any],
    cfg: AgenticFilterConfig,
) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    conversations = obj.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return False, ["missing_conversations"], None

    signals = summarize_signals(conversations)
    signals["sample_char_count"] = estimate_sample_chars(conversations, cfg)
    reasons: List[str] = []
    has_standard_tool_interaction = (
        signals["assistant_tool_call_turns"] > 0 or signals["tool_turns"] > 0
    )
    has_shell_interaction = (
        signals["assistant_shell_action_turns"] > 0
        and signals["shell_observation_turns"] > 0
    )
    intentional_no_call_candidate = is_intentional_no_call_candidate(conversations, signals)
    signals["intentional_no_call_candidate"] = intentional_no_call_candidate

    if cfg.min_turns is not None and signals["turn_count"] < cfg.min_turns:
        reasons.append("too_few_turns")
    if cfg.max_turns is not None and signals["turn_count"] > cfg.max_turns:
        reasons.append("too_many_turns")
    if (
        not has_standard_tool_interaction
        and not has_shell_interaction
        and not intentional_no_call_candidate
    ):
        reasons.append("no_tool_interaction")
    if cfg.min_closed_tool_loops > 0 and signals["closed_tool_loops"] < cfg.min_closed_tool_loops:
        reasons.append("no_closed_tool_loop")
    if cfg.require_final_assistant and signals["roles"][-1] != "assistant":
        reasons.append("final_turn_not_assistant")
    if (
        cfg.require_declared_tools_or_system
        and signals["tool_declaration_turns"] == 0
        and signals["system_turns"] == 0
    ):
        reasons.append("missing_system_or_tool_schema")
    if (
        cfg.max_tool_content_chars is not None
        and signals["max_tool_content_chars"] > cfg.max_tool_content_chars
    ):
        reasons.append("tool_payload_too_long")
    if cfg.max_sample_chars is not None and signals["sample_char_count"] > cfg.max_sample_chars:
        reasons.append("sample_too_long")

    if reasons:
        return False, reasons, signals
    return True, [], signals


def build_output_record(
    obj: Dict[str, Any],
    cfg: AgenticFilterConfig,
    *,
    source_file: str,
    sample_index: int,
    signals: Optional[Dict[str, Any]] = None,
    rejection_reasons: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    conversations = obj.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return None, False

    had_pii = detect_basic_pii(conversations)
    normalized_conversations, did_redact = normalize_conversation(conversations, cfg)
    meta = {
        **(signals or summarize_signals(conversations)),
        "had_basic_pii": had_pii,
        "reasoning_removed": cfg.strip_reasoning_content,
        "pii_redacted": did_redact,
    }
    record = {
        "source_file": source_file,
        "sample_index": sample_index,
        "agentic_meta": meta,
        "conversations": normalized_conversations,
    }
    if rejection_reasons:
        record["rejection_reasons"] = list(rejection_reasons)
    return record, did_redact


def adapt_agentic_record(
    obj: Dict[str, Any],
    cfg: AgenticFilterConfig,
    *,
    source_file: str,
    sample_index: int,
) -> Tuple[Optional[Dict[str, Any]], List[str], Optional[Dict[str, Any]], bool]:
    keep, reasons, signals = evaluate_agentic_record(obj, cfg)
    if not keep:
        return None, reasons, signals, False

    record, did_redact = build_output_record(
        obj,
        cfg,
        source_file=source_file,
        sample_index=sample_index,
        signals=signals,
    )
    return record, [], signals, did_redact


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def process_files(
    input_files: Iterable[Path],
    output_path: Path,
    cfg: AgenticFilterConfig,
    *,
    reject_output_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> AgenticStats:
    stats = AgenticStats()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    if reject_output_path is not None:
        reject_output_path.parent.mkdir(parents=True, exist_ok=True)
        reject_output_path.write_text("", encoding="utf-8")

    for path in input_files:
        stats.files_seen += 1
        for sample_index, parsed in enumerate(iter_json_array_objects(path)):
            stats.objects_seen += 1
            if parsed.error is not None:
                stats.parse_errors += 1
                if reject_output_path is not None:
                    append_jsonl(
                        reject_output_path,
                        {
                            "source_file": path.name,
                            "sample_index": sample_index,
                            "rejection_reasons": ["parse_error"],
                            "parse_error": parsed.error,
                        },
                    )
                continue

            obj = parsed.obj
            if obj is None:
                stats.parse_errors += 1
                if reject_output_path is not None:
                    append_jsonl(
                        reject_output_path,
                        {
                            "source_file": path.name,
                            "sample_index": sample_index,
                            "rejection_reasons": ["parse_error"],
                            "parse_error": "Parsed event missing object payload",
                        },
                    )
                continue

            record, reasons, signals, did_redact = adapt_agentic_record(
                obj,
                cfg,
                source_file=path.name,
                sample_index=sample_index,
            )

            if record is not None:
                append_jsonl(output_path, record)
                stats.kept += 1
                if did_redact:
                    stats.redacted_records += 1
            else:
                stats.rejected += 1
                if reject_output_path is not None:
                    reject_record, reject_did_redact = build_output_record(
                        obj,
                        cfg,
                        source_file=path.name,
                        sample_index=sample_index,
                        signals=signals,
                        rejection_reasons=reasons,
                    )
                    if reject_record is not None:
                        append_jsonl(reject_output_path, reject_record)
                        if reject_did_redact:
                            stats.redacted_records += 1
                    else:
                        append_jsonl(
                            reject_output_path,
                            {
                                "source_file": path.name,
                                "sample_index": sample_index,
                                "rejection_reasons": reasons,
                                "signals": signals,
                            },
                        )

            if limit is not None and stats.kept >= limit:
                return stats

    return stats


def stats_to_dict(stats: AgenticStats, cfg: AgenticFilterConfig) -> Dict[str, Any]:
    return {
        "stats": asdict(stats),
        "filter_config": asdict(cfg),
    }
