"""
DataSearcher 包 — 执行层内置「数据发现」模块。

职责（见 pipeline_architecture.md §3.1）：
- 在目标约束（领域、语言、许可、规模等）下调用 LLM API（Gemini/Qwen/阿里云等）
  完成数据源检索与样本抽取。
- 输出：带元数据的候选数据集清单及本地样本 URI，供下游 UDatasets / AI Coder 使用。

本包包含：
- client：Sample 步骤，LLM 检索 + 结构化输出校验，产出 sample.json。
- downloader：Download 步骤，按 sample 下载 HuggingFace/GitHub 并校验，产出 download_report。
- agentic_adaptor：从超大原始 JSON 分片中流式筛选适合 agentic SFT 的轨迹。
"""

