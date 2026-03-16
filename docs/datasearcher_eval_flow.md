# DataSearcher 完整评估流程

## 一、流程概览

```
search → slice → llm_eval → eval
  │        │         │         │
  │        │         │         └─ 合并打分 → sample_scored.jsonl
  │        │         └─ LLM 逐条打分 → sample_llm_scored.jsonl
  │        └─ 拉取 samples → samples/*.json
  └─ LLM 检索 → sample.json + recall_pool.jsonl
```

## 二、各阶段说明

| 阶段 | 命令 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| **search** | `MODE=search ./smoke_datasearcher.sh` | - | `sample.json`, `recall_pool.jsonl` | LLM 解析 intent，调用 HF/GH API 检索 |
| **slice** | `MODE=slice ./smoke_datasearcher.sh` | sample.json | `samples/*.json` | 按 slice_download_list 拉取样本（API，无物理下载）。默认前 5 行；`RANDOM_SAMPLE=1` 时从完整范围随机抽取离散行 |
| **llm_eval** | `MODE=llm_eval ./smoke_datasearcher.sh` | samples/, recall_pool | `sample_llm_scored.jsonl` | 对每条 sample 第一条数据调用 LLM 打分（截断 3 万字符） |
| **eval** | `MODE=eval ./smoke_datasearcher.sh` | samples/, recall_pool, sample_llm_scored | `sample_scored.jsonl` | 合并 metadata + LLM 分数，纯本地计算 |

## 三、关键文件

| 文件 | 作用 |
|------|------|
| `out/datasearcher/sample.json` | search 产出的候选列表（含 slice_download_list） |
| `out/datasearcher/recall_pool.jsonl` | 召回池 metadata（downloads/likes/stars 等） |
| `out/datasearcher/samples/*.json` | 各数据集的 slice 样本（前 5 行） |
| `out/datasearcher/sample_llm_scored.jsonl` | LLM 原始打分（repo_id, llm_score） |
| `out/datasearcher/sample_scored.jsonl` | 最终打分结果（合并 metadata + LLM 分数） |
| `configs/prompts/sample_eval_prompt.txt` | LLM 评估 prompt（10 档锚定） |

## 四、完整重跑评估（含 3 万截断 LLM 打分）

若要**从头完整做一次评估**（含重新跑 LLM 打分）：

```bash
# 1. 检索（若已有 sample.json 且不想重跑可跳过）
MODE=search ./smoke_datasearcher.sh

# 2. 拉取 slice（若 samples 已存在可跳过）
MODE=slice ./smoke_datasearcher.sh

# 3. LLM 打分（3 万截断，从头重跑需加 --no-resume）
# 方式 A：删掉旧结果后正常跑（默认 resume）
rm -f out/datasearcher/sample_llm_scored.jsonl
MODE=llm_eval ./smoke_datasearcher.sh

# 方式 B：不删文件，强制重跑
python src/datasearcher/sample_llm_eval.py \
  --samples-dir out/datasearcher/samples \
  --recall-pool out/datasearcher/recall_pool.jsonl \
  --output out/datasearcher/sample_llm_scored.jsonl \
  --prompt configs/prompts/sample_eval_prompt.txt \
  --no-resume

# 4. 合并打分（eval 会自动读取 sample_llm_scored.jsonl）
MODE=eval ./smoke_datasearcher.sh
```

## 五、断点续跑

- **llm_eval**：默认 `resume`，已打分的 repo_id 会跳过；中断后直接再跑即可续跑
- **slice**：已存在的 `*_sample.json` 会跳过，不重复拉取

## 六、打分规则（sample_scorer）

1. **基础分**：row_count × 1.0
2. **平台热度**：HF downloads/likes 渐近 5 分（5×x/(x+k)），GH stars 线性
3. **Intent 匹配**：每个匹配 token +3.0
4. **LLM 分数**：若存在 `sample_llm_scored.jsonl`，合并为最高权重

## 七、LLM 评估配置

- 截断：`FIRST_ROW_TRUNCATE_CHARS = 30000`（约 16k tokens 下 3 万字符）
- 模型：`configs/datasearcher_api.json` 或默认 qwen-plus
- API：`API=aliyun` 或 `API=local`（同 search）
