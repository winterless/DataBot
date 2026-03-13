#!/usr/bin/env bash
# DataSearcher 烟雾测试：按 MODE 执行 search | slice | download
# 用法：
#   search  -> 仅召回
#   slice   -> search + API Sample（50 个，前 5 行 JSON，无物理下载）
#   download -> search + slice + API Sample（10 个）
set -euo pipefail

# Load .env if present (GITHUB_TOKEN, DASHSCOPE_API_KEY, etc.)
[[ -f .env ]] && set -a && . .env && set +a

API="${API:-aliyun}"       # export API=local to use local provider
MODE="${MODE:-slice}"     # search | slice | download

mkdir -p out/datasearcher state/datasearcher

echo "[smoke] MODE=${MODE} API=${API}"

_run_search() {
  python "src/datasearcher/client.py" --api "${API}" --output "out/datasearcher/sample.json"
}

_run_slice() {
  python "src/datasearcher/downloader.py" --sample "out/datasearcher/sample.json" --samples-dir "out/datasearcher/samples" --mode slice
}

_run_download() {
  python "src/datasearcher/downloader.py" --sample "out/datasearcher/sample.json" --samples-dir "out/datasearcher/samples" --mode full
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
  *)
    echo "MODE must be search|slice|download" >&2
    exit 2
    ;;
esac
