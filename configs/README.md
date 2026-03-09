# 配置文件说明

本目录存放 DataBot 各模块配置，统一以 `pipeline_architecture.md` 为准。

| 文件 | 说明 |
|------|------|
| `datasearcher_api.json` | DataSearcher provider/request 配置；包含 `source_policy_file` 引用。 |
| `source_policy.json` | 数据源配比策略（MVP 默认 HuggingFace 60% / GitHub 40%）。 |
| `prompts/datasearcher_prompt.txt` | DataSearcher 检索提示模板。 |

`datasearcher_api.json` 的 `selection` 字段用于两层筛选参数：
- `recall_pool_size`：第一层召回池数量（如 120）
- `download_size`：第二层下载清单数量（如 10）
- `preferred_size`：数据大小范围过滤，避免过大或过小
  - `min_mb` / `max_mb`：GitHub 仓库体积范围（MB，如 1～500）
