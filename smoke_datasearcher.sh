#!/usr/bin/env bash
# DataSearcher 烟雾测试：按 MODE 执行 search（召回+规则筛选）、download（下载）或 all。
# 用法：MODE=search|download|all API=aliyun|local ./smoke_datasearcher.sh
# search 融合了 extract：API 召回 + preferred_size/exclude_size_human 规则筛选 -> download_list
set -euo pipefail

API="${API:-aliyun}"   # export API=local to use local provider
MODE="${MODE:-search}"    # search | download | all

mkdir -p out/datasearcher
case "${MODE}" in
  search) python "src/datasearcher/client.py" --api "${API}" ;;
  download) nohup python "src/datasearcher/downloader.py" > "out/datasearcher/download_nohup.log" 2>&1 & echo $! > "out/datasearcher/download_nohup.pid" ;;
  all) python "src/datasearcher/client.py" --api "${API}" && nohup python "src/datasearcher/downloader.py" > "out/datasearcher/download_nohup.log" 2>&1 & echo $! > "out/datasearcher/download_nohup.pid" ;;
  *) echo "MODE must be search|download|all" >&2; exit 2 ;;
esac
