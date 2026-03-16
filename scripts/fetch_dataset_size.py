#!/usr/bin/env python3
"""
测试 HF Datasets Server /size API — 获取数据集在 HuggingFace 上的大小。

用法:
  python scripts/fetch_dataset_size.py ibm/duorc
  python scripts/fetch_dataset_size.py togethercomputer/CoderForge-Preview
  python scripts/fetch_dataset_size.py ibm/duorc --total-only   # 仅输出总字节数
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# 支持从项目根目录运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasearcher.datasets_server_api import fetch_size, fetch_total_size_bytes


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "N/A"
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/fetch_dataset_size.py <dataset_repo_id> [--total-only] [--json]", file=sys.stderr)
        print("Example: python scripts/fetch_dataset_size.py ibm/duorc", file=sys.stderr)
        return 2
    repo_id = sys.argv[1].strip()
    if not repo_id:
        print("Error: empty dataset name", file=sys.stderr)
        return 2
    total_only = "--total-only" in sys.argv

    if total_only:
        num_bytes, err = fetch_total_size_bytes(repo_id)
        if err:
            print(f"Error: {err}", file=sys.stderr)
            return 1
        print(num_bytes if num_bytes is not None else "")
        return 0

    data, err = fetch_size(repo_id)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    size_obj = data.get("size") or {}
    ds = size_obj.get("dataset") or {}
    configs = size_obj.get("configs") or []
    splits = size_obj.get("splits") or []
    partial = data.get("partial", False)

    print(f"\n=== {repo_id} ===\n")
    print("【整体】")
    print(f"  行数: {ds.get('num_rows', 'N/A'):,}" if isinstance(ds.get("num_rows"), int) else f"  行数: {ds.get('num_rows', 'N/A')}")
    print(f"  原始文件: {_fmt_bytes(ds.get('num_bytes_original_files'))}")
    print(f"  Parquet: {_fmt_bytes(ds.get('num_bytes_parquet_files'))}")
    print(f"  内存估算: {_fmt_bytes(ds.get('num_bytes_memory'))}")
    if partial:
        print("  ⚠ partial=true: 数据集过大，实际可能更大")

    if configs:
        print("\n【按 config】")
        for c in configs[:5]:
            cfg = c.get("config", "?")
            rows = c.get("num_rows", 0)
            parquet = c.get("num_bytes_parquet_files")
            print(f"  {cfg}: {rows:,} 行, {_fmt_bytes(parquet)}")

    if splits:
        print("\n【按 split (前 8 个)】")
        for s in splits[:8]:
            cfg = s.get("config", "?")
            sp = s.get("split", "?")
            rows = s.get("num_rows", 0)
            parquet = s.get("num_bytes_parquet_files")
            print(f"  {cfg}/{sp}: {rows:,} 行, {_fmt_bytes(parquet)}")

    if "--json" in sys.argv:
        print("\n【原始 JSON】")
        print(json.dumps(data, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
