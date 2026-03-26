#!/usr/bin/env python3
"""临时脚本：用 recall_pool 的 size_bytes 补充 sample_scored.jsonl。recall 中无则跳过。"""
import json
import sys
from pathlib import Path

RECALL = Path("out/datasearcher/recall_pool.jsonl")
SCORED = Path("out/datasearcher/sample_scored.jsonl")


def main():
    if not RECALL.exists():
        print(f"recall_pool 不存在: {RECALL}", file=sys.stderr)
        return 1
    if not SCORED.exists():
        print(f"sample_scored 不存在: {SCORED}", file=sys.stderr)
        return 1

    # repo_id -> size_bytes (仅非 None)
    size_map = {}
    with RECALL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rid = str(obj.get("repo_id", "")).strip()
                vm = obj.get("verified_meta") or {}
                sb = vm.get("size_bytes")
                if rid and sb is not None:
                    size_map[rid] = sb
            except json.JSONDecodeError:
                continue

    print(f"recall 中有 size_bytes 的: {len(size_map)} 个", file=sys.stderr)

    lines = []
    updated = 0
    with SCORED.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                lines.append(line + "\n")
                continue
            rid = str(obj.get("repo_id", "")).strip()
            if rid in size_map:
                vm = obj.get("verified_meta") or {}
                sb = size_map[rid]
                vm["size_bytes"] = sb
                vm["size_gb"] = round(sb / (1024**3), 3)
                obj["verified_meta"] = vm
                updated += 1
            lines.append(json.dumps(obj, ensure_ascii=False) + "\n")

    SCORED.write_text("".join(lines), encoding="utf-8")
    print(f"已更新 {updated} 条", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
