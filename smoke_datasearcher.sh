#!/usr/bin/env bash
# DataSearcher 烟雾测试：按 MODE 执行 search | slice | download | eval
# 用法：
#   search  -> 仅召回
#   slice   -> search + API Sample（50 个，前 5 行 JSON，无物理下载）+ 打分
#   download -> search + slice + API Sample（10 个）
#   eval    -> 仅对已存在的 samples 打分（增量，不下载）
#   llm_eval -> 调用 API 模型对每个 sample 第一条数据打分（实时输出，可中断续跑）
set -euo pipefail

# Load .env if present (GITHUB_TOKEN, DASHSCOPE_API_KEY, etc.)
[[ -f .env ]] && set -a && . .env && set +a

API="${API:-aliyun}"       # export API=local to use local provider
MODE="${MODE:-slice}"     # search | slice | download | eval

mkdir -p out/datasearcher state/datasearcher

echo "[smoke] MODE=${MODE} API=${API}"

_run_search() {
  python "src/datasearcher/client.py" --api "${API}" --output "out/datasearcher/sample.json"
}

_run_slice() {
  EXTRA=""
  [[ -n "${RANDOM_SAMPLE:-}" ]] && EXTRA="--random-sample"
  python "src/datasearcher/downloader.py" --sample "out/datasearcher/sample.json" --samples-dir "out/datasearcher/samples" --mode slice $EXTRA
}

_run_download() {
  python "src/datasearcher/downloader.py" --sample "out/datasearcher/sample.json" --samples-dir "out/datasearcher/samples" --mode full
}

_run_eval() {
  python "src/datasearcher/downloader.py" --samples-dir "out/datasearcher/samples" --mode eval --recall-pool "out/datasearcher/recall_pool.jsonl" --scored-output "out/datasearcher/sample_scored.jsonl"
}

_run_llm_eval() {
  python "src/datasearcher/sample_llm_eval.py" \
    --samples-dir "out/datasearcher/samples" \
    --recall-pool "out/datasearcher/recall_pool.jsonl" \
    --output "out/datasearcher/sample_llm_scored.jsonl" \
    --prompt "configs/prompts/sample_eval_prompt.txt"
}

case "${MODE}" in
  search)
    _run_search
    ;;
  slice)
    _run_search
    _run_slice
    ;;
  download)
    _run_search
    _run_slice
    _run_download
    ;;
  eval)
    _run_eval
    ;;
  llm_eval)
    _run_llm_eval
    ;;
  *)
    echo "MODE must be search|slice|download|eval|llm_eval" >&2
    exit 2
    ;;
esac
