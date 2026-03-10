#!/usr/bin/env python3
"""
DataSearcher Downloader — Download 步骤（按样本清单下载并校验）。

读取 DataSearcher 产出的 sample 文件（JSON 或 JSONL），从中解析出 datasets 列表；
按 source_type（huggingface / github）调用 huggingface-cli 或 git clone 下载到本地目录。
支持重试、下载后校验（hf cache verify / git fsck），并将每条结果追加写入 download_report（JSONL）。

与 pipeline_architecture.md 对应：§3.1 DataSearcher 的「本地样本 URI」产出，为 §5.1 数据准备契约提供 raw_dir/sample_file。
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    """Print execution status to stderr (does not pollute report JSONL)."""
    print(f"[{_now_ts()}] [Downloader] {msg}", file=sys.stderr)


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


def _extract_from_json_obj(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []

    if isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("download_list"), list):
        return [x for x in obj["data"]["download_list"] if isinstance(x, dict)]

    if isinstance(obj.get("data"), dict) and isinstance(obj["data"].get("valid_datasets"), list):
        return [x for x in obj["data"]["valid_datasets"] if isinstance(x, dict)]

    if isinstance(obj.get("datasets"), list):
        return [x for x in obj["datasets"] if isinstance(x, dict)]

    if all(k in obj for k in ("source_type", "repo_id")):
        return [obj]

    return []


def load_samples(sample_path: Path) -> List[Dict[str, Any]]:
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    datasets: List[Dict[str, Any]] = []
    if sample_path.suffix == ".jsonl":
        with sample_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at line {i}: {e}") from e
                datasets.extend(_extract_from_json_obj(obj))
    else:
        obj = json.loads(sample_path.read_text(encoding="utf-8"))
        datasets.extend(_extract_from_json_obj(obj))

    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in datasets:
        st = str(d.get("source_type", "")).strip().lower()
        rid = str(d.get("repo_id", "")).strip()
        if not st or not rid:
            continue
        dedup[(st, rid)] = d
    return list(dedup.values())


def build_download_cmd(item: Dict[str, Any], target_root: Path) -> Tuple[List[str], Path]:
    source_type = str(item.get("source_type", "")).strip().lower()
    repo_id = str(item.get("repo_id", "")).strip()
    if source_type == "huggingface":
        local_dir = target_root / f"hf__{_sanitize(repo_id)}"
        cmd = [
            "huggingface-cli",
            "download",
            "--repo-type",
            "dataset",
            repo_id,
            "--local-dir",
            str(local_dir),
        ]
        return cmd, local_dir
    if source_type == "github":
        local_dir = target_root / f"gh__{_sanitize(repo_id)}"
        cmd = ["git", "clone", "--depth", "1", f"https://github.com/{repo_id}.git", str(local_dir)]
        return cmd, local_dir
    raise ValueError(f"Unsupported source_type: {source_type}")


def build_verify_cmds(item: Dict[str, Any], local_dir: Path) -> List[List[str]]:
    source_type = str(item.get("source_type", "")).strip().lower()
    repo_id = str(item.get("repo_id", "")).strip()
    if source_type == "huggingface":
        return [
            [
                "hf",
                "cache",
                "verify",
                repo_id,
                "--repo-type",
                "dataset",
                "--local-dir",
                str(local_dir),
                "--fail-on-missing-files",
            ]
        ]
    if source_type == "github":
        return [
            ["git", "-C", str(local_dir), "rev-parse", "HEAD"],
            ["git", "-C", str(local_dir), "fsck", "--full"],
        ]
    raise ValueError(f"Unsupported source_type: {source_type}")


def run_with_retry(cmd: List[str], retries: int) -> Tuple[bool, str]:
    last_err = ""
    for attempt in range(1, retries + 2):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return True, ""
        last_err = (proc.stderr or proc.stdout or "").strip()
        if attempt <= retries:
            time.sleep(min(3 * attempt, 10))
    return False, last_err


def run_cmd(cmd: List[str]) -> Tuple[bool, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or "").strip()


def run_verify_steps(verify_cmds: List[List[str]]) -> Tuple[bool, str, str]:
    for verify_cmd in verify_cmds:
        ok, err = run_cmd(verify_cmd)
        if not ok:
            return False, " ".join(verify_cmd), err
    return True, "", ""


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _index_key(source_type: str, repo_id: str) -> str:
    return f"{source_type.strip().lower()}::{repo_id.strip()}"


def load_download_index(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except Exception:
                continue
            source_type = str(row.get("source_type", "")).strip().lower()
            repo_id = str(row.get("repo_id", "")).strip()
            if not source_type or not repo_id:
                continue
            out[_index_key(source_type, repo_id)] = row
    return out


def append_download_index(path: Path, record: Dict[str, Any]) -> None:
    append_jsonl(path, record)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download datasets from sample json/jsonl.")
    parser.add_argument("--sample", default="out/datasearcher/sample.jsonl")
    parser.add_argument("--target-dir", default="/home/unlimitediw/workspace/DataBot_dataset")
    parser.add_argument("--report", default="out/datasearcher/download_report.jsonl")
    parser.add_argument("--download-index", default="state/datasearcher/download_index.jsonl")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    sample_path = Path(args.sample)
    if not sample_path.exists():
        fallback = Path("out/datasearcher/sample.json")
        if fallback.exists():
            sample_path = fallback
        else:
            raise FileNotFoundError(
                f"Neither sample file exists: {args.sample} nor fallback out/datasearcher/sample.json"
            )

    target_root = Path(args.target_dir)
    target_root.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    download_index_path = Path(args.download_index)
    download_index = load_download_index(download_index_path)

    datasets = load_samples(sample_path)
    if args.max_items > 0:
        datasets = datasets[: args.max_items]

    _log(f"Starting: sample={sample_path}, target={target_root}, report={report_path}")
    _log(f"Loaded {len(datasets)} datasets (index entries: {len(download_index)})")
    if args.dry_run:
        _log("DRY-RUN mode: no actual download")
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, item in enumerate(datasets, start=1):
        source_type = str(item.get("source_type", "")).strip().lower()
        repo_id = str(item.get("repo_id", "")).strip()
        dataset_name = str(item.get("dataset_name", repo_id)).strip() or repo_id
        key = _index_key(source_type, repo_id)

        index_hit = download_index.get(key, {})
        index_status = str(index_hit.get("status", "")).strip().lower()
        index_path = Path(str(index_hit.get("path", "")).strip()) if index_hit.get("path") else None
        if index_status == "success" and index_path and index_path.exists():
            skip_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "success",
                    "action": "skip_index_hit",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "path": str(index_path),
                },
            )
            _log(f"[{idx}/{len(datasets)}] SKIP_INDEX {dataset_name}")
            continue
        try:
            cmd, local_dir = build_download_cmd(item, target_root)
            verify_cmds = build_verify_cmds(item, local_dir)
        except Exception as e:
            fail_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "failed",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "error": str(e),
                },
            )
            _log(f"[{idx}/{len(datasets)}] FAILED {dataset_name}: {e}")
            continue

        if local_dir.exists() and any(local_dir.iterdir()):
            _log(f"[{idx}/{len(datasets)}] VERIFY_EXISTING {dataset_name} ...")
            verified, failed_cmd, verify_err = run_verify_steps(verify_cmds)
            if verified:
                skip_count += 1
                index_record = {
                    "ts": _now_ts(),
                    "status": "success",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "path": str(local_dir),
                    "action": "skip_existing",
                }
                download_index[key] = index_record
                append_download_index(download_index_path, index_record)
                append_jsonl(
                    report_path,
                    {
                        "ts": _now_ts(),
                        "status": "success",
                        "action": "skip_existing",
                        "dataset_name": dataset_name,
                        "source_type": source_type,
                        "repo_id": repo_id,
                        "path": str(local_dir),
                        "verify_cmds": verify_cmds,
                    },
                )
                _log(f"[{idx}/{len(datasets)}] SKIP+VERIFIED {dataset_name}")
            else:
                fail_count += 1
                append_jsonl(
                    report_path,
                    {
                        "ts": _now_ts(),
                        "status": "failed",
                        "dataset_name": dataset_name,
                        "source_type": source_type,
                        "repo_id": repo_id,
                        "path": str(local_dir),
                        "verify_cmds": verify_cmds,
                        "failed_verify_cmd": failed_cmd,
                        "error": verify_err,
                    },
                )
                _log(f"[{idx}/{len(datasets)}] FAILED_VERIFY {dataset_name}: {verify_err}")
            continue

        if args.dry_run:
            _log(f"[{idx}/{len(datasets)}] DRY-RUN {dataset_name}: {' '.join(cmd)}")
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "dry_run",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "cmd": cmd,
                    "verify_cmds": verify_cmds,
                    "path": str(local_dir),
                },
            )
            continue

        _log(f"[{idx}/{len(datasets)}] DOWNLOADING {dataset_name} ...")
        ok, err = run_with_retry(cmd, retries=args.retries)
        if ok:
            print(f"[{idx}/{len(datasets)}] VERIFYING {dataset_name} ...")
            verified, failed_cmd, verify_err = run_verify_steps(verify_cmds)
            if verified:
                ok_count += 1
                index_record = {
                    "ts": _now_ts(),
                    "status": "success",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "path": str(local_dir),
                    "action": "downloaded",
                }
                download_index[key] = index_record
                append_download_index(download_index_path, index_record)
                append_jsonl(
                    report_path,
                    {
                        "ts": _now_ts(),
                        "status": "success",
                        "dataset_name": dataset_name,
                        "source_type": source_type,
                        "repo_id": repo_id,
                        "cmd": cmd,
                        "verify_cmds": verify_cmds,
                        "path": str(local_dir),
                    },
                )
                _log(f"[{idx}/{len(datasets)}] OK+VERIFIED {dataset_name}")
            else:
                fail_count += 1
                append_jsonl(
                    report_path,
                    {
                        "ts": _now_ts(),
                        "status": "failed",
                        "dataset_name": dataset_name,
                        "source_type": source_type,
                        "repo_id": repo_id,
                        "cmd": cmd,
                        "verify_cmds": verify_cmds,
                        "failed_verify_cmd": failed_cmd,
                        "path": str(local_dir),
                        "error": verify_err,
                    },
                )
                _log(f"[{idx}/{len(datasets)}] FAILED_VERIFY {dataset_name}: {verify_err}")
        else:
            fail_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "failed",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                    "cmd": cmd,
                    "verify_cmds": verify_cmds,
                    "path": str(local_dir),
                    "error": err,
                },
            )
            _log(f"[{idx}/{len(datasets)}] FAILED {dataset_name}: {err}")

    _log(f"Done: success={ok_count}, failed={fail_count}, skipped={skip_count}, total={len(datasets)}")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
