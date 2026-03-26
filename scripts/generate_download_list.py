#!/usr/bin/env python3
"""
从 sample_scored.jsonl 生成 download 列表：top_k 内且 size_gb 在 [min_gb, max_gb] 的进入 download。
配置来自 configs/datasearcher_api.json 的 selection.download_filter。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIG = Path("configs/datasearcher_api.json")
SCORED = Path("out/datasearcher/sample_scored.jsonl")
OUTPUT = Path("out/datasearcher/download_list.jsonl")


def main() -> int:
    if not SCORED.exists():
        print(f"sample_scored 不存在: {SCORED}", file=sys.stderr)
        return 1

    cfg = {}
    if CONFIG.exists():
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    sel = cfg.get("selection") or {}
    df = sel.get("download_filter") or {}
    top_k = int(df.get("top_k", 200))
    min_gb = float(df.get("min_gb", 2))
    max_gb = float(df.get("max_gb", 200))

    items = []
    with SCORED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    top = items[:top_k]
    filtered = [
        x
        for x in top
        if x.get("verified_meta", {}).get("size_gb") is not None
        and min_gb <= x["verified_meta"]["size_gb"] <= max_gb
    ]
    total_gb = sum(x["verified_meta"]["size_gb"] for x in filtered)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for item in filtered:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Top {top_k} 内 {min_gb}-{max_gb} GB: {len(filtered)} 个", file=sys.stderr)
    print(f"总大小: {round(total_gb, 3)} GB", file=sys.stderr)
    print(f"输出: {OUTPUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
