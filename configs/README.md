# 配置文件说明

本目录存放 DataBot 各模块配置，统一以 `pipeline_architecture.md` 为准。

| 文件 | 说明 |
|------|------|
| `datasearcher_api.json` | DataSearcher provider/request 配置；包含 `source_policy_file` 引用。 |
| `source_policy.json` | 数据源配比策略（MVP 默认 HuggingFace 60% / GitHub 40%）。 |
| `prompts/datasearcher_prompt.txt` | DataSearcher 检索提示模板。 |
