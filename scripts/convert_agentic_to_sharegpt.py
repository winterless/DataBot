#!/usr/bin/env python3
"""
Convert agentic trajectory JSON/JSONL files into a ShareGPT structure that is
strictly compatible with `SharegptStyleInstructionHandler`.

Key normalization rules:
- keep only per-turn `from` / `value` inside top-level `conversations`
- move regular system prompts to top-level `system`
- move declared tools to top-level `tools`
- convert assistant turns with `tool_calls` into explicit `function_call` turns
- merge consecutive tool/system observation turns into a single `observation`
- preserve alternating ShareGPT slots:
  - even positions: `human` / `observation`
  - odd positions: `gpt` / `function_call`

Examples:
python scripts/convert_agentic_to_sharegpt.py \
  --input out/agentic/step35_agentic_sft \
  --output out/agentic/step35_agentic_sharegpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasearcher.agentic_adaptor import _is_system_observation_text  # noqa: E402


def _iter_input_files(input_path: Path, glob_pattern: str) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return sorted(p for p in input_path.glob(glob_pattern) if p.is_file())


def _resolve_output_file(input_root: Path, output_root: Path, source_file: Path) -> Path:
    if input_root.is_file():
        return output_root
    return output_root / source_file.name


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object record at {path}:{line_number}")
            yield record


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                raise ValueError(f"Expected object record at {path}[{idx}]")
        return data
    if isinstance(data, dict):
        return [data]
    raise ValueError(f"Unsupported JSON payload in {path}")


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _iter_jsonl_records(path)
    return _load_json_records(path)


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _wrap(tag: str, value: str) -> str:
    return f"<{tag}>\n{value}\n</{tag}>"


def _serialize_tool_calls(tool_calls: List[Any]) -> str:
    return _stringify(tool_calls)


def _is_observation_turn(turn: Dict[str, Any]) -> bool:
    role = str(turn.get("role", "")).strip()
    content = _stringify(turn.get("content", ""))
    return role == "tool" or (role == "system" and _is_system_observation_text(content))


def _normalize_text_parts(parts: List[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if str(part).strip()).strip()


def _build_message(role: str, value: str) -> Optional[Dict[str, Any]]:
    normalized_value = value.strip()
    if not normalized_value:
        return None
    return {"from": role, "value": normalized_value}


def _message_side(role: str) -> str:
    if role in {"human", "observation"}:
        return "input"
    if role in {"gpt", "function_call"}:
        return "output"
    raise ValueError(f"Unsupported ShareGPT role: {role}")


def _merge_sharegpt_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse adjacent messages that belong to the same ShareGPT side.

    This repairs dirty trajectories such as:
    - `human -> human`
    - `gpt -> function_call`
    - `function_call -> function_call`

    If either assistant-side message contains a tool call, keep the merged role as
    `function_call`; otherwise keep `gpt`.
    """
    merged: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("from", "")).strip()
        value = str(message.get("value", "")).strip()
        if not role or not value:
            continue

        if not merged:
            merged.append({"from": role, "value": value})
            continue

        prev = merged[-1]
        prev_role = str(prev.get("from", "")).strip()
        if _message_side(prev_role) != _message_side(role):
            merged.append({"from": role, "value": value})
            continue

        merged_role = prev_role
        if role == "function_call" or prev_role == "function_call":
            merged_role = "function_call"

        prev["from"] = merged_role
        prev["value"] = _normalize_text_parts([str(prev.get("value", "")), value])

    return merged


def _maybe_close_pending_observation(converted_turns: List[Dict[str, Any]]) -> None:
    """
    Some source trajectories start a new user turn immediately after a tool
    observation, without an assistant natural-language wrap-up. Insert a short
    assistant close-out so the resulting ShareGPT sequence still alternates
    legally for `SharegptStyleInstructionHandler`.
    """
    if not converted_turns:
        return
    last_role = str(converted_turns[-1].get("from", "")).strip()
    if last_role == "observation":
        converted_turns.append({"from": "gpt", "value": "Tool result received."})


def _build_observation_message(
    turns: List[Dict[str, Any]],
    *,
    tool_response_tag: str,
) -> Optional[Dict[str, Any]]:
    parts: List[str] = []
    for turn in turns:
        content = _stringify(turn.get("content", "")).strip()
        if content:
            parts.append(_wrap(tool_response_tag, content))
    return _build_message("observation", _normalize_text_parts(parts))


def _build_function_call_message(
    turn: Dict[str, Any],
    *,
    tool_call_tag: str,
) -> Optional[Dict[str, Any]]:
    content = _stringify(turn.get("content", "")).strip()
    tool_calls = turn.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return None
    parts: List[str] = []
    if content:
        parts.append(content)
    parts.append(_wrap(tool_call_tag, _serialize_tool_calls(tool_calls)))
    return _build_message("function_call", _normalize_text_parts(parts))


def _collect_available_tools(conversations: List[Any]) -> List[Any]:
    merged: List[Any] = []
    seen: set[str] = set()
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        tools = turn.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            key = _stringify(tool)
            if key in seen:
                continue
            seen.add(key)
            merged.append(tool)
    return merged


def _collect_system_prompt(conversations: List[Any]) -> str:
    parts: List[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", "")).strip()
        if role != "system" or _is_observation_turn(turn):
            continue
        content = _stringify(turn.get("content", "")).strip()
        if content:
            parts.append(content)
    return _normalize_text_parts(parts)


def _convert_conversations(
    conversations: List[Any],
    *,
    tool_call_tag: str,
    tool_response_tag: str,
) -> List[Dict[str, Any]]:
    converted_turns: List[Dict[str, Any]] = []
    idx = 0
    total = len(conversations)

    while idx < total:
        raw_turn = conversations[idx]
        turn = raw_turn if isinstance(raw_turn, dict) else {"role": "", "content": _stringify(raw_turn)}
        role = str(turn.get("role", "")).strip()

        if role == "system" and not _is_observation_turn(turn):
            idx += 1
            continue

        if role == "user":
            _maybe_close_pending_observation(converted_turns)
            message = _build_message("human", _stringify(turn.get("content", "")))
            if message is not None:
                converted_turns.append(message)
            idx += 1
            continue

        if role == "assistant":
            tool_calls = turn.get("tool_calls")
            if isinstance(tool_calls, list) and tool_calls:
                message = _build_function_call_message(turn, tool_call_tag=tool_call_tag)
                if message is not None:
                    converted_turns.append(message)

                observation_turns: List[Dict[str, Any]] = []
                next_idx = idx + 1
                while next_idx < total:
                    next_raw = conversations[next_idx]
                    next_turn = (
                        next_raw
                        if isinstance(next_raw, dict)
                        else {"role": "", "content": _stringify(next_raw)}
                    )
                    if not _is_observation_turn(next_turn):
                        break
                    observation_turns.append(next_turn)
                    next_idx += 1

                observation_message = _build_observation_message(
                    observation_turns,
                    tool_response_tag=tool_response_tag,
                )
                if observation_message is not None:
                    converted_turns.append(observation_message)

                idx = next_idx
                continue

            message = _build_message("gpt", _stringify(turn.get("content", "")))
            if message is not None:
                converted_turns.append(message)
            idx += 1
            continue

        if _is_observation_turn(turn):
            observation_turns = [turn]
            next_idx = idx + 1
            while next_idx < total:
                next_raw = conversations[next_idx]
                next_turn = (
                    next_raw
                    if isinstance(next_raw, dict)
                    else {"role": "", "content": _stringify(next_raw)}
                )
                if not _is_observation_turn(next_turn):
                    break
                observation_turns.append(next_turn)
                next_idx += 1

            observation_message = _build_observation_message(
                observation_turns,
                tool_response_tag=tool_response_tag,
            )
            if observation_message is not None:
                converted_turns.append(observation_message)
            idx = next_idx
            continue

        raise ValueError(f"Unsupported role during ShareGPT conversion: {role or '<empty>'}")

    return _merge_sharegpt_messages(converted_turns)


def _validate_strict_sharegpt_turns(messages: List[Dict[str, Any]]) -> None:
    odd_tags = ("human", "observation")
    even_tags = ("gpt", "function_call")
    accept_tags = (odd_tags, even_tags)
    for turn_idx, message in enumerate(messages):
        role = str(message.get("from", "")).strip()
        if role not in accept_tags[turn_idx % 2]:
            raise ValueError(
                "Converted messages do not satisfy ShareGPT alternation: "
                f"index={turn_idx}, role={role}, messages={messages}"
            )


def _convert_record(
    record: Dict[str, Any],
    *,
    tool_call_tag: str,
    tool_response_tag: str,
) -> Dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Record missing conversations list")

    converted_turns = _convert_conversations(
        conversations,
        tool_call_tag=tool_call_tag,
        tool_response_tag=tool_response_tag,
    )
    _validate_strict_sharegpt_turns(converted_turns)

    output_record: Dict[str, Any] = {"conversations": converted_turns}

    system_prompt = _collect_system_prompt(conversations)
    if system_prompt:
        output_record["system"] = system_prompt

    available_tools = _collect_available_tools(conversations)
    if available_tools:
        output_record["tools"] = _stringify(available_tools)

    return output_record


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert agentic JSON/JSONL trajectories into ShareGPT-like turns."
    )
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument("--output", required=True, help="Output file or directory.")
    parser.add_argument(
        "--glob",
        default="*.jsonl",
        help="Glob used when --input is a directory. Default: *.jsonl",
    )
    parser.add_argument(
        "--tool-call-tag",
        default="tool_call",
        help="Wrapper tag name for assistant tool calls. Default: tool_call",
    )
    parser.add_argument(
        "--tool-response-tag",
        default="tool_response",
        help="Wrapper tag name for tool/system observations. Default: tool_response",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    input_files = _iter_input_files(input_path, args.glob)
    if not input_files:
        raise FileNotFoundError(f"No input files matched: {input_path}")

    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_records = 0
    for source_file in input_files:
        target_file = _resolve_output_file(input_path, output_path, source_file)
        target_file.parent.mkdir(parents=True, exist_ok=True)
        with target_file.open("w", encoding="utf-8") as out_fh:
            for record in _iter_records(source_file):
                converted = _convert_record(
                    record,
                    tool_call_tag=args.tool_call_tag,
                    tool_response_tag=args.tool_response_tag,
                )
                out_fh.write(json.dumps(converted, ensure_ascii=False) + "\n")
                converted_records += 1

    print(
        json.dumps(
            {
                "input": str(input_path),
                "output": str(output_path),
                "files": len(input_files),
                "records": converted_records,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
