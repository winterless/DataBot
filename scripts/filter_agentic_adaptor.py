#!/usr/bin/env python3
"""
从超大 JSON 分片中流式筛选适合 agentic SFT 的轨迹。

默认策略：
- 默认允许样本停在 tool / shell observation / 中间状态
- 默认去掉 reasoning_content
- 默认做基础 PII 脱敏（邮箱 / 电话）
- 目录输入时默认按 input file 拆分输出
- 默认不限制最小 turn 数和闭环数
- 默认不限制最大 turn 数和 tool 返回长度
- 默认过滤最终样本长度超过 32k 字符的样本

示例：
python scripts/filter_agentic_adaptor.py \
  --input data/hf__stepfun-ai__Step-3.5-Flash-SFT/json/general \
  --output out/agentic/step35_agentic_sft \
  --reject-output out/agentic/step35_agentic_rest
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasearcher.agentic_adaptor import (  # noqa: E402
    AgenticFilterConfig,
    iter_input_files,
    process_files,
)


def _log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [AgenticAdaptor] {msg}", file=sys.stderr)


def _merge_stats(total: dict, current: dict) -> None:
    for key, value in current.items():
        total[key] = total.get(key, 0) + value


def _resolve_split_output_dir(input_path: Path, output_path: Path) -> Path:
    if input_path.is_file():
        return output_path.parent
    if output_path.suffix:
        return output_path.parent / output_path.stem
    return output_path


def _resolve_file_output_path(input_path: Path, output_path: Path, source_file: Path) -> Path:
    if input_path.is_file():
        return output_path
    out_dir = _resolve_split_output_dir(input_path, output_path)
    return out_dir / f"{source_file.stem}.jsonl"


def _resolve_summary_output_path(input_path: Path, output_path: Path, summary_arg: str) -> Path:
    if summary_arg:
        return Path(summary_arg)
    if input_path.is_file():
        return output_path.with_suffix(output_path.suffix + ".summary.json")
    if output_path.suffix:
        return output_path.parent / f"{output_path.stem}.summary.json"
    return output_path / "summary.json"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter agentic SFT-friendly trajectories from huge JSON shards."
    )
    parser.add_argument("--input", required=True, help="Input file or directory.")
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output path. For a single input file, this is the JSONL file path. "
            "For a directory input, default behavior is split-by-input: if this "
            "is a directory path, files are written under it; if it ends with "
            ".jsonl, a sibling directory named by the stem is created."
        ),
    )
    parser.add_argument(
        "--glob",
        default="chunk_*.json",
        help="Glob used when --input is a directory. Default: chunk_*.json",
    )
    parser.add_argument(
        "--reject-output",
        default="",
        help=(
            "Optional output path for the complement set (rejected samples). "
            "Uses the same split-by-input behavior as --output."
        ),
    )
    parser.add_argument(
        "--summary-output",
        default="",
        help="Optional JSON summary path.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Stop after keeping N records.")
    parser.add_argument(
        "--min-turns",
        type=int,
        default=0,
        help="Minimum turn count. 0 means no limit.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=0,
        help="Maximum turn count. 0 means no limit.",
    )
    parser.add_argument(
        "--min-closed-tool-loops",
        type=int,
        default=0,
        help="Minimum closed tool loops. 0 means no requirement.",
    )
    parser.add_argument(
        "--max-tool-content-chars",
        type=int,
        default=0,
        help="Maximum chars in any single tool payload. 0 means no limit.",
    )
    parser.add_argument(
        "--max-sample-chars",
        type=int,
        default=32000,
        help="Maximum chars in a single normalized sample. 0 means no limit.",
    )
    parser.add_argument(
        "--require-declared-tools-or-system",
        action="store_true",
        help="Require at least one system turn or tools schema turn.",
    )
    parser.add_argument(
        "--require-final-assistant",
        action="store_true",
        help="Require the last turn to be assistant.",
    )
    parser.add_argument(
        "--keep-reasoning-content",
        action="store_true",
        help="Keep reasoning_content instead of stripping it.",
    )
    parser.add_argument(
        "--keep-pii",
        action="store_true",
        help="Disable basic PII redaction.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    reject_output_root = Path(args.reject_output) if args.reject_output else None
    summary_output_path = _resolve_summary_output_path(input_path, output_path, args.summary_output)

    cfg = AgenticFilterConfig(
        min_turns=max(1, args.min_turns) if args.min_turns > 0 else None,
        max_turns=max(1, args.max_turns) if args.max_turns > 0 else None,
        min_closed_tool_loops=max(0, args.min_closed_tool_loops),
        max_tool_content_chars=(
            max(1, args.max_tool_content_chars)
            if args.max_tool_content_chars > 0
            else None
        ),
        max_sample_chars=max(1, args.max_sample_chars) if args.max_sample_chars > 0 else None,
        require_final_assistant=args.require_final_assistant,
        require_declared_tools_or_system=args.require_declared_tools_or_system,
        strip_reasoning_content=not args.keep_reasoning_content,
        redact_pii=not args.keep_pii,
    )

    input_files = iter_input_files(input_path, glob_pattern=args.glob)
    if not input_files:
        _log("No input files matched.")
        return 1

    _log(f"Matched {len(input_files)} input file(s)")
    if input_path.is_file():
        _log(f"Writing filtered records to: {output_path}")
        if reject_output_root is not None:
            _log(f"Writing complement records to: {reject_output_root}")
    else:
        split_dir = _resolve_split_output_dir(input_path, output_path)
        _log(f"Writing split-by-input outputs under: {split_dir}")
        if reject_output_root is not None:
            reject_split_dir = _resolve_split_output_dir(input_path, reject_output_root)
            _log(f"Writing split-by-input complement outputs under: {reject_split_dir}")

    remaining_limit = args.limit or None
    total_stats = {
        "files_seen": 0,
        "objects_seen": 0,
        "kept": 0,
        "rejected": 0,
        "parse_errors": 0,
        "redacted_records": 0,
    }
    per_file_outputs = []

    for source_file in input_files:
        if remaining_limit is not None and remaining_limit <= 0:
            break

        file_output_path = _resolve_file_output_path(input_path, output_path, source_file)
        file_reject_output_path = (
            _resolve_file_output_path(input_path, reject_output_root, source_file)
            if reject_output_root is not None
            else None
        )
        file_limit = remaining_limit
        file_output_path.parent.mkdir(parents=True, exist_ok=True)
        _log(f"Processing {source_file.name} -> {file_output_path}")
        stats = process_files(
            [source_file],
            file_output_path,
            cfg,
            reject_output_path=file_reject_output_path,
            limit=file_limit,
        )
        _merge_stats(total_stats, asdict(stats))
        file_summary = {
            "input_file": str(source_file),
            "filtered_jsonl": str(file_output_path),
            "stats": asdict(stats),
        }
        if file_reject_output_path is not None:
            file_summary["rejected_jsonl"] = str(file_reject_output_path)
        per_file_outputs.append(file_summary)
        if remaining_limit is not None:
            remaining_limit -= stats.kept

    summary = {
        "stats": total_stats,
        "filter_config": asdict(cfg),
        "input": {
            "path": str(input_path),
            "glob": args.glob,
            "matched_files": [str(p) for p in input_files],
        },
        "output": {
            "mode": "single_file" if input_path.is_file() else "split_by_input",
            "path": str(output_path),
            "reject_path": str(reject_output_root) if reject_output_root is not None else None,
            "files": per_file_outputs,
        },
    }
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    _log(
        "Done. "
        f"seen={total_stats['objects_seen']} kept={total_stats['kept']} "
        f"rejected={total_stats['rejected']} redacted={total_stats['redacted_records']} "
        f"parse_errors={total_stats['parse_errors']}"
    )
    _log(f"Summary written to: {summary_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
