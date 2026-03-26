#!/usr/bin/env python3
"""
Convert agentic trajectory JSON/JSONL files into a ShareGPT-like format.

Default transformation:
- keep top-level record fields, but rewrite `conversations`
- rename per-turn `role` -> `from`
- rename per-turn `content` -> `value`
- wrap assistant `tool_calls` with <tool_call>...</tool_call>
- wrap tool/system observations with <tool_response>...</tool_response>

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
from typing import Any, Dict, Iterable, Iterator, List


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


def _convert_turn(
    turn: Dict[str, Any],
    *,
    tool_call_tag: str,
    tool_response_tag: str,
) -> Dict[str, Any]:
    role = str(turn.get("role", "")).strip()
    content = _stringify(turn.get("content", ""))
    value_parts: List[str] = []
    is_observation = _is_observation_turn(turn)

    if content:
        if is_observation:
            value_parts.append(_wrap(tool_response_tag, content))
        else:
            value_parts.append(content)

    tool_calls = turn.get("tool_calls")
    if role == "assistant" and isinstance(tool_calls, list) and tool_calls:
        value_parts.append(_wrap(tool_call_tag, _serialize_tool_calls(tool_calls)))

    new_turn = {
        key: value
        for key, value in turn.items()
        if key not in {"role", "content"}
    }
    new_turn["from"] = "observation" if is_observation else role
    new_turn["value"] = "\n\n".join(part for part in value_parts if part).strip()
    return new_turn


def _convert_record(
    record: Dict[str, Any],
    *,
    tool_call_tag: str,
    tool_response_tag: str,
) -> Dict[str, Any]:
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("Record missing conversations list")

    converted = dict(record)
    converted["conversations"] = [
        _convert_turn(
            turn if isinstance(turn, dict) else {"role": "", "content": _stringify(turn)},
            tool_call_tag=tool_call_tag,
            tool_response_tag=tool_response_tag,
        )
        for turn in conversations
    ]
    return converted


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
