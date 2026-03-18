#!/usr/bin/env python3
"""
DataSearcher Sample Fetcher / Full Downloader。

模式：
- slice/full（无 download-list）：通过 Datasets Server API 拉取样本（前 N 行 JSON），保存到 samples/
- full（有 download-list）：全量下载到 data/，默认后台执行
- eval: 仅打分，不下载
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from .datasets_server_api import fetch_sample_via_api
    from .sample_scorer import score_samples
except ImportError:
    from datasets_server_api import fetch_sample_via_api
    from sample_scorer import score_samples


def _now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now_ts()}] [SampleFetcher] {msg}", file=sys.stderr)


def _sanitize(name: str) -> str:
    return name.replace("/", "__")


def _extract_from_json_obj(obj: Dict[str, Any], use_slice: bool = False) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []

    data = obj.get("data") if isinstance(obj.get("data"), dict) else None
    if data:
        if use_slice and isinstance(data.get("slice_download_list"), list):
            slice_items = [x for x in data["slice_download_list"] if isinstance(x, dict)]
            if slice_items:
                return slice_items
            if isinstance(data.get("download_list"), list):
                dl = [x for x in data["download_list"] if isinstance(x, dict)]
                if dl:
                    _log("slice_download_list empty, fallback to download_list")
                    return dl
        if isinstance(data.get("download_list"), list):
            return [x for x in data["download_list"] if isinstance(x, dict)]
        if isinstance(data.get("valid_datasets"), list):
            return [x for x in data["valid_datasets"] if isinstance(x, dict)]

    if isinstance(obj.get("datasets"), list):
        return [x for x in obj["datasets"] if isinstance(x, dict)]

    if all(k in obj for k in ("source_type", "repo_id")):
        return [obj]

    return []


def load_samples(sample_path: Path, use_slice: bool = False) -> List[Dict[str, Any]]:
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
                datasets.extend(_extract_from_json_obj(obj, use_slice=use_slice))
    else:
        obj = json.loads(sample_path.read_text(encoding="utf-8"))
        datasets.extend(_extract_from_json_obj(obj, use_slice=use_slice))

    dedup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for d in datasets:
        st = str(d.get("source_type", "")).strip().lower()
        rid = str(d.get("repo_id", "")).strip()
        if not st or not rid:
            continue
        dedup[(st, rid)] = d
    return list(dedup.values())


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _sanitize_local_name(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def _run_full_download(
    datasets: List[Dict[str, Any]],
    download_dir: Path,
    report_path: Path,
    log_fn,
    retries: int = 2,
) -> int:
    """全量下载：始终调用 snapshot_download，让 HF Hub 自动补齐缺失文件。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        log_fn("ERROR: huggingface_hub 未安装，无法全量下载")
        return 1

    download_dir.mkdir(parents=True, exist_ok=True)
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, item in enumerate(datasets, start=1):
        source_type = str(item.get("source_type", "")).strip().lower()
        repo_id = str(item.get("repo_id", "")).strip()
        dataset_name = str(item.get("dataset_name", repo_id)).strip() or repo_id

        if source_type != "huggingface":
            skip_count += 1
            append_jsonl(report_path, {"ts": _now_ts(), "status": "skipped", "reason": "HF-only", "repo_id": repo_id})
            log_fn(f"[{idx}/{len(datasets)}] SKIP {dataset_name} (非 HF)")
            continue

        local_dir = download_dir / f"hf__{_sanitize_local_name(repo_id)}"
        had_existing_files = local_dir.exists() and any(local_dir.iterdir())
        action = "RESUMING" if had_existing_files else "DOWNLOADING"
        log_fn(f"[{idx}/{len(datasets)}] {action} {dataset_name} -> {local_dir} ...")
        last_err = None
        for attempt in range(1, retries + 2):
            try:
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(local_dir),
                )
                ok_count += 1
                append_jsonl(
                    report_path,
                    {
                        "ts": _now_ts(),
                        "status": "success",
                        "action": "resumed" if had_existing_files else "downloaded",
                        "repo_id": repo_id,
                        "path": str(local_dir),
                    },
                )
                log_fn(f"[{idx}/{len(datasets)}] OK {dataset_name}")
                break
            except Exception as e:
                last_err = str(e)
                if attempt <= retries:
                    time.sleep(attempt * 2)
                    continue
                fail_count += 1
                append_jsonl(report_path, {"ts": _now_ts(), "status": "failed", "repo_id": repo_id, "error": last_err})
                log_fn(f"[{idx}/{len(datasets)}] FAILED {dataset_name}: {last_err}")
                break

    log_fn(f"Done: success={ok_count}, failed={fail_count}, skipped={skip_count}, total={len(datasets)}")
    return 0 if fail_count == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch dataset samples via HuggingFace Datasets Server API (no physical download)."
    )
    parser.add_argument("--sample", default="out/datasearcher/sample.json")
    parser.add_argument("--download-list", default="", help="从 sample_scored 生成的 download_list.jsonl，优先于 sample")
    parser.add_argument("--samples-dir", default="out/datasearcher/samples")
    parser.add_argument("--report", default="out/datasearcher/sample_report.jsonl")
    parser.add_argument("--row-length", type=int, default=5)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--mode",
        choices=("full", "slice", "eval"),
        default="slice",
        help="full=download_list, slice=slice_download_list, eval=仅打分已存在的samples(不下载)",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--random-sample", action="store_true", help="随机 offset 采样，而非取前 N 行")
    parser.add_argument("--recall-pool", default="out/datasearcher/recall_pool.jsonl")
    parser.add_argument("--scored-output", default="out/datasearcher/sample_scored.jsonl")
    parser.add_argument("--llm-scored", default="out/datasearcher/sample_llm_scored.jsonl", help="LLM 打分结果，若存在则合并为最高权重")
    parser.add_argument("--intent", default="", help="Intent 文本，用于打分时的语义匹配")
    parser.add_argument("--no-background", action="store_true", help="全量下载时前台执行（默认后台）")
    parser.add_argument("--download-dir", default="data", help="全量下载保存目录")
    parser.add_argument("--download-log", default="out/datasearcher/download_full.log", help="全量下载日志")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir)
    samples_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)

    # eval 模式：仅对已存在的 samples 打分，不下载
    if args.mode == "eval":
        _log("Mode=eval: skipping fetch, scoring existing samples only")
        recall_pool_path = Path(args.recall_pool)
        scored_output_path = Path(args.scored_output)
        llm_scored_path = Path(args.llm_scored)
        results = score_samples(
            samples_dir,
            recall_pool_path,
            scored_output_path,
            intent_text=args.intent,
            llm_scored_path=llm_scored_path,
        )
        _log(f"Scored {len(results)} samples -> {scored_output_path}")
        return 0

    # --download-list：全量下载模式
    if args.download_list and Path(args.download_list).exists():
        datasets = []
        with Path(args.download_list).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if obj.get("source_type") and obj.get("repo_id"):
                        datasets.append(obj)
                except json.JSONDecodeError:
                    continue
        _log(f"Loaded {len(datasets)} datasets from download_list={args.download_list}")

        if not args.no_background:
            # 后台执行：spawn 子进程，主进程立即返回
            log_path = Path(args.download_log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            script = Path(__file__).resolve()
            cmd = [sys.executable, str(script),
                "--download-list", args.download_list,
                "--samples-dir", args.samples_dir,
                "--download-dir", args.download_dir,
                "--download-log", args.download_log,
                "--no-background",
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=log_path.open("a", encoding="utf-8"),
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=Path.cwd(),
            )
            _log(f"全量下载已在后台启动 PID={proc.pid}，日志: {log_path}")
            return 0

        # 前台执行全量下载
        report_full = Path(args.download_log).parent / "download_full_report.jsonl"
        return _run_full_download(datasets, Path(args.download_dir), report_full, _log, args.retries)
    else:
        sample_path = Path(args.sample)
        if not sample_path.exists():
            fallback = Path("out/datasearcher/sample.json")
            if fallback.exists():
                sample_path = fallback
            else:
                raise FileNotFoundError(
                    f"Sample file not found: {args.sample} nor fallback out/datasearcher/sample.json"
                )
        use_slice = args.mode == "slice"
        datasets = load_samples(sample_path, use_slice=use_slice)

    if args.max_items > 0:
        datasets = datasets[: args.max_items]

    src = args.download_list or args.sample
    _log(f"Starting: source={src}, samples_dir={samples_dir}, mode={args.mode}")
    _log(f"Loaded {len(datasets)} datasets (API sample, no physical download)")
    if args.dry_run:
        _log("DRY-RUN mode")

    ok_count = 0
    fail_count = 0
    skip_count = 0

    for idx, item in enumerate(datasets, start=1):
        source_type = str(item.get("source_type", "")).strip().lower()
        repo_id = str(item.get("repo_id", "")).strip()
        dataset_name = str(item.get("dataset_name", repo_id)).strip() or repo_id

        if source_type != "huggingface":
            skip_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "skipped",
                    "reason": "Viewer_Disabled",
                    "message": "Datasets Server API is HuggingFace-only",
                    "dataset_name": dataset_name,
                    "source_type": source_type,
                    "repo_id": repo_id,
                },
            )
            _log(f"[{idx}/{len(datasets)}] SKIP {dataset_name} (GitHub, API not supported)")
            continue

        out_file = samples_dir / f"{_sanitize(repo_id)}_sample.json"
        if out_file.exists():
            skip_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "success",
                    "action": "skip_existing",
                    "dataset_name": dataset_name,
                    "repo_id": repo_id,
                    "path": str(out_file),
                },
            )
            _log(f"[{idx}/{len(datasets)}] SKIP_EXISTING {dataset_name}")
            continue

        if args.dry_run:
            _log(f"[{idx}/{len(datasets)}] DRY-RUN {dataset_name}")
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "dry_run",
                    "dataset_name": dataset_name,
                    "repo_id": repo_id,
                },
            )
            continue

        _log(f"[{idx}/{len(datasets)}] FETCHING {dataset_name} ...")
        rows, err = fetch_sample_via_api(
            repo_id,
            length=args.row_length,
            retries=args.retries,
            delay_sec=args.delay,
            random_sample=args.random_sample,
        )
        if err:
            fail_count += 1
            append_jsonl(
                report_path,
                {
                    "ts": _now_ts(),
                    "status": "failed",
                    "reason": "Viewer_Disabled" if err == "Viewer_Disabled" else "api_error",
                    "dataset_name": dataset_name,
                    "repo_id": repo_id,
                    "error": err,
                },
            )
            _log(f"[{idx}/{len(datasets)}] FAILED {dataset_name}: {err}")
            continue

        payload = {"dataset": repo_id, "rows": rows}
        out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        ok_count += 1
        append_jsonl(
            report_path,
            {
                "ts": _now_ts(),
                "status": "success",
                "dataset_name": dataset_name,
                "repo_id": repo_id,
                "path": str(out_file),
                "row_count": len(rows) if rows else 0,
            },
        )
        _log(f"[{idx}/{len(datasets)}] OK {dataset_name} -> {out_file.name} ({len(rows) if rows else 0} rows)")
        time.sleep(args.delay)

    _log(f"Done: success={ok_count}, failed={fail_count}, skipped={skip_count}, total={len(datasets)}")

    # 打分：在 slice 下载完成后对 samples 进行评分排序（若有 llm_scored 则合并）
    recall_pool_path = Path(args.recall_pool)
    scored_output_path = Path(args.scored_output)
    llm_scored_path = Path(args.llm_scored)
    results = score_samples(
        samples_dir,
        recall_pool_path,
        scored_output_path,
        intent_text=args.intent,
        llm_scored_path=llm_scored_path,
    )
    _log(f"Scored {len(results)} samples -> {scored_output_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
