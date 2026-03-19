# 弱智吧幽默分类 SFT 微调 — 开发计划 v0.1

> https://gemini.google.com/app/d68553d9dea8513b?hl=zh-cn

> **目标**: 基于弱智吧 (Ruozhiba) 段子分类数据，对 Qwen3-4B-Instruct-2507 进行 LoRA SFT 微调，使模型学会对中文幽默段子进行结构化分类解析（8 类 Top-3 + 思考过程），后续尝试 Qwen3-30B-A3B-Instruct-2507 (MoE) 对比效果。
>
> **作业要求**: CSS5120 Lab3 — 选基座模型 → 定义 SFT 目标 → 数据预处理 → 训练 → Before/After 评估
>
> **硬件**: 2× NVIDIA L20Z (80GB VRAM each)

---

## 总览：四阶段流水线

```
Phase 1  数据工程        ──→  Phase 2  SFT 训练      ──→  Phase 3  评估       ──→  Phase 4  报告
(~2 天)                     (~2 天)                     (~1 天)                  (~1 天)
┌─────────────────┐    ┌──────────────────────┐    ┌──────────────────┐    ┌────────────────┐
│ 1.1 CQIA 补全   │    │ 2.1 ShareGPT 格式化   │    │ 3.1 sglang 推理   │    │ 4.1 Loss 曲线  │
│ 1.2 去重防污染   │    │ 2.2 注册数据集        │    │ 3.2 定量指标      │    │ 4.2 Before/After│
│ 1.3 数据统计验证 │    │ 2.3 MVP 最小链路      │    │ 3.3 LLM-as-Judge │    │ 4.3 PDF 报告   │
└─────────────────┘    │ 2.4 显存压测          │    └──────────────────┘    └────────────────┘
                       │ 2.5 配置 & 训练       │
                       │ 2.6 Loss 监控         │
                       │ 2.7 权重合并 (Merge)  │
                       └──────────────────────┘
```

---

## Phase 1: 数据工程与深度清洗

### 1.1 CQIA 数据补全 — 新增 `thought_process` 字段 (导师蒸馏)

**问题**: 当前 `ruozhiba_cqia_classified.json` 中 classification 只有 `top3_categories`，缺少 `thought_process` 字段（贴吧数据有此字段）。CQIA 的 `output` 字段是对原文的正经解答，不等同于分类思考过程。

**方案**: 基于 `classify_cqia.py` 编写 `classify_cqia_updated.py`，为 240 条 CQIA 数据补充 `thought_process`。

> **导师蒸馏 (Distillation)**: 项目使用 **Claude-Opus-4-6** 作为导师模型生成 `thought_process` 标签。4B 参数量的 Qwen3 在理解弱智吧复杂逻辑谬误时，极度依赖高质量的思维链 (CoT) 数据。通过 Opus 级别模型生成的深度分析，可以强迫小模型学习更高维度的语义特征，突破其本身的认知局限。

#### 新脚本: `scripts/data/classify_cqia_updated.py`

```
输入:  data/CQIA/ruozhiba_cqia_classified.json  (240 条, 已有 top3_categories)
输出:  data/CQIA/ruozhiba_cqia_classified_v2.json (240 条, 新增 thought_process)
```

**关键变更**:
- System Prompt 对齐 `classify_config.yaml`（贴吧版），要求输出 `thought_process` + `top3_categories`
- User 输入: 只使用 `instruction` 字段（即弱智吧风格的问题原文）
- 保留原有 `output`, `classification` 字段不变，在 classification 中新增 `thought_process`
- 复用 `classify_jokes.py` 的鲁棒性优化（ThreadPoolExecutor, 断点续传, 原子写入）

#### 配置文件: `scripts/data/classify_cqia_updated_config.yaml`

```yaml
system_prompt: |
  # Role
  你是一个精通中文互联网亚文化、语言学和幽默解构的"弱智吧"资深吧友...
  (与 classify_config.yaml 相同的 system prompt，要求输出 thought_process + top3_categories)

files_to_process:
  - input: ruozhiba_cqia_classified.json
    output: ruozhiba_cqia_classified_v2.json

processing:
  max_workers: 4
  sleep_time: 0.5
  temperature: 0.3
  max_tokens: 1500   # thought_process 需要更多 token
```

**实现要点**:
1. 读取已有的 `ruozhiba_cqia_classified.json`
2. 对每条数据，使用 `instruction` 作为输入调用 LLM，获取 `thought_process` + `top3_categories`
3. 将新的 `thought_process` 合并到已有的 `classification` 对象中
4. 可选：对比新旧 `top3_categories` 是否一致（仅记录日志，不强制覆盖）

**API 速率控制与鲁棒性**:

调用 Claude-Opus-4-6 生成 240 条深度 `thought_process`（每条约 1500 tokens），API 限流是第一个工程瓶颈。脚本需内置以下防护:

```python
from tenacity import retry, wait_exponential, stop_after_attempt
import asyncio

# 限制最大并发数, 防止触发 API 429 限流
semaphore = asyncio.Semaphore(3)

@retry(wait=wait_exponential(multiplier=1, min=2, max=60), stop=stop_after_attempt(5))
async def classify_single(item):
    async with semaphore:
        # ... 调用 API
```

- **并发控制**: `asyncio.Semaphore(3)` 限制最大 3 个并发请求
- **指数退避重试**: `tenacity` 的 `wait_exponential` 在遇到 HTTP 429 时自动等待 2s → 4s → 8s → ... 最长 60s
- **断点续传**: 复用 `classify_jokes.py` 的 checkpoint 机制，中断后可从最后成功的条目恢复

**Prompt Caching (提示词缓存)**:

240 条数据共享完全相同的超长 System Prompt（~800 tokens），每次请求都重新编码该 Prompt 会产生大量冗余的输入 token 费用。若 API 平台支持 Prompt Caching（如 Anthropic 的 `cache_control` 或 OpenAI 的自动前缀缓存），强烈建议启用:

```python
# Anthropic 风格: 在 system message 中标记缓存锚点
messages = [
    {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"}  # 标记为可缓存
            }
        ]
    },
    {"role": "user", "content": item["instruction"]}
]
```

- **成本收益**: 缓存命中后输入 token 成本降低 ~90%，240 条数据几乎全部命中缓存（仅首次请求为 cache miss）
- **延迟收益**: 缓存命中时 TTFT (Time To First Token) 显著降低，变相缓解并发限流压力
- **兼容性**: 若 API 平台不支持显式缓存控制，跳过此优化即可，不影响功能正确性

---

### 1.2 去重防污染 — CQIA 测试集 vs 贴吧训练集

**目标**: 确保 CQIA 测试集 240 条不出现在训练集中，防止数据泄露 (Data Leakage)。

#### 新脚本: `scripts/data/dedup_test_vs_train.py`

```
输入:
  - 测试集: data/CQIA/ruozhiba_cqia_classified_v2.json (240 条, instruction 字段)
  - 训练集: data/tieba/best*_classified.json (9 个文件, text 字段)
输出:
  - 去重报告: data/dedup_report.json
  - 去重后训练集: data/tieba/best*_classified_dedup.json
```

**技术方案**:

| 步骤 | 方法 | 函数 / 工具 |
|------|------|-------------|
| **精确去重** | MD5 哈希比对 | `hashlib.md5(text.strip().encode()).hexdigest()` |
| **模糊去重** | 序列相似度 | `difflib.SequenceMatcher(None, a, b).ratio() > 0.9` |
| **加速** | 多进程 | `concurrent.futures.ProcessPoolExecutor` |

**实现逻辑**:
1. 将 CQIA 240 条 `instruction` 构建为哈希指纹集合 (`set`)
2. 遍历 9 个贴吧 `_classified.json` 文件中的 `text` 字段
3. 先精确匹配（哈希），再模糊匹配（SequenceMatcher, 阈值 0.9）
4. 输出: 命中的条目列表 + 去重后的训练集（从训练集中**移除**与测试集重复的条目）
5. 保存去重报告（精确命中数、模糊命中数、各文件影响条数）

**重要**: 去重方向是从**训练集中剔除**与测试集重复的条目，测试集 240 条保持不变。

---

### 1.3 数据统计与验证

去重完成后，生成数据统计报告:

```
去重前:
  贴吧训练集总量: 2813 条
    近三年 (2023-2025): 1035 条
    全量 (2018-2025):   2813 条
  CQIA 测试集: 240 条

去重后:
  贴吧训练集总量: XXXX 条 (移除 XX 条重复)
    近三年 (2023-2025): XXXX 条
    全量 (2018-2025):   XXXX 条
```

运行 `check_and_repair.py` 验证去重后数据的完整性。

---

## Phase 2: SFT 训练

### 2.1 ShareGPT 格式化

#### 新脚本: `scripts/data/build_sft_data.py`

将分类后的数据转换为 LLaMA-Factory 所需的 ShareGPT 格式。

```
输入:
  - data/tieba/best*_classified_dedup.json (Phase 1.2 去重后输出)
输出:
  - LLaMA-Factory/data/ruozhiba_last3.json   (近三年 2023-2025)
  - LLaMA-Factory/data/ruozhiba_all.json      (全量 2018-2025)
```

**Prompt 版本管理 (Prompt Versioning)**:

为防止微小的字符差异（中英文标点、空格、换行符）干扰评估，创建中心化配置文件:

#### 新文件: `configs/prompts.yaml`

```yaml
system_prompt: |
  你是一个弱智吧幽默解构专家。请对用户输入的段子进行深度分析，识别其幽默机制，并输出 Top-3 分类结果。
  请以 JSON 格式输出，包含 thought_process（分析推理）和 top3_categories（分类结果）两个字段。

  可选类别：古典弱智、奇怪提问、弱智科学家、人生态度、文字游戏、地狱笑话、谐音梗、文艺弱智。

categories:
  - 古典弱智
  - 奇怪提问
  - 弱智科学家
  - 人生态度
  - 文字游戏
  - 地狱笑话
  - 谐音梗
  - 文艺弱智
```

所有脚本通过读取此文件获取 `SYSTEM_PROMPT`，确保训练、推理、评估的 Prompt 环境完全镜像。

**ShareGPT 格式** (每条数据一个对话):

```json
{
  "conversations": [
    {
      "from": "system",
      "value": "(从 configs/prompts.yaml 读取的 system_prompt)"
    },
    {
      "from": "human",
      "value": "四川人至死不渝，重庆人乐不思蜀。"
    },
    {
      "from": "gpt",
      "value": "{\"thought_process\": \"这句话的核心笑点在于...\", \"top3_categories\": [{\"rank\": 1, \"category\": \"文字游戏\", \"confidence_score\": 0.7, \"reason\": \"...\"}, ...]}"
    }
  ]
}
```

**字段映射**:

| 来源字段 | ShareGPT 映射 |
|----------|---------------|
| `system` | 统一的系统提示词（从 `configs/prompts.yaml` 读取，含 8 类说明） |
| `text` (贴吧) | `human.value` |
| `classification` (含 `thought_process` + `top3_categories`) | `gpt.value` (序列化为 JSON 字符串) |

> **Prompt 对齐 (Prompt Parity)**: 不在 `human.value` 中注入年份前缀。测试集 (CQIA) 不含年份信息，若训练时注入【年份】标签会导致训练/测试分布不一致，使模型对特定年份词汇过拟合。年份仅作为数据集内部的管理元数据，不进入模型输入。

> **JSON 序列化防坑 — `ensure_ascii=False`**: 在 `build_sft_data.py` 中将 `classification` 对象序列化为 `gpt.value` 的 JSON 字符串时，`json.dumps()` **必须**设置 `ensure_ascii=False`。否则所有中文字符会被转义为 `\uXXXX` 格式（如 `文字游戏` → `\u6587\u5b57\u6e38\u620f`），导致模型在微调时浪费算力学习无意义的 ASCII 转义规则，严重影响输出自然中文的能力。正确写法:
> ```python
> gpt_value = json.dumps(classification, ensure_ascii=False, indent=None)
> ```

**数据集划分**:

| 数据集 | 文件 | 年份 | 预估条数 |
|--------|------|------|----------|
| `ruozhiba_last3.json` | 2023 + 2024 + 2025 | 近三年 | ~1035 (去重后) |
| `ruozhiba_all.json` | 2018-2025 全部 | 全量 | ~2813 (去重后) |

---

### 2.2 注册数据集

在 `LLaMA-Factory/data/dataset_info.json` 中添加:

```json
"ruozhiba_last3": {
  "file_name": "ruozhiba_last3.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
},
"ruozhiba_all": {
  "file_name": "ruozhiba_all.json",
  "formatting": "sharegpt",
  "columns": {
    "messages": "conversations"
  },
  "tags": {
    "role_tag": "from",
    "content_tag": "value",
    "user_tag": "human",
    "assistant_tag": "gpt",
    "system_tag": "system"
  }
}
```

---

### 2.3 MVP 最小可行链路

> **目的**: 先用最小配置跑通**数据 → 训练 → 推理 → 评估**全链路，验证所有脚本和配置正确后，再启动完整 checkpoint 搜索。

#### MVP 配置

| 参数 | 值 | 说明 |
|------|-----|------|
| Rank | 8 | 最小搜索维度 |
| Epoch | 3 | 最快完成训练 |
| `lora_alpha` | 16 | $2 \times \text{rank}$ |
| `report_to` | none | 先保存到本地，不依赖 wandb |
| GPU | 单卡 (GPU 0) | 验证单卡即可 |

#### MVP 配置文件: `configs/qwen3_4b_mvp.yaml`

基于 `qwen3_4b_base.yaml`，覆盖关键参数:

```yaml
### 继承 qwen3_4b_base.yaml 全部配置, 以下参数覆盖:
### model
model_name_or_path: /root/code/llm_ruozhiba/models/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
lora_alpha: 16

### dataset
dataset: ruozhiba_all
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-4b/lora/mvp_r8_e3
logging_steps: 10
save_strategy: epoch
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none                       # MVP 阶段不依赖 wandb

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
num_train_epochs: 3
lr_scheduler_type: cosine
warmup_ratio: 0.1
seed: 42                              # 与 base.yaml 保持一致
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.05
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 100
```

#### MVP 执行步骤

```bash
# 1. 训练 (单卡, ~数十分钟)
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/qwen3_4b_mvp.yaml

# 2. 推理 (使用 MVP checkpoint)
python scripts/inference/inference_eval.py \
  --adapter saves/qwen3-4b/lora/mvp_r8_e3 \
  --output results/results_mvp.json

# 3. 评估 (验证两阶段评估流程)
python scripts/viz/eval_metrics.py \
  --results results/results_mvp.json \
  --gold data/CQIA/ruozhiba_cqia_classified_v2.json
```

#### MVP 验证标准

- [ ] 训练正常完成，loss 呈下降趋势
- [ ] `saves/qwen3-4b/lora/mvp_r8_e3/` 下生成 adapter 文件
- [ ] 推理脚本成功加载 adapter 并生成 240 条结果
- [ ] 两阶段评估报告正常输出 (JSON 遵循率 + Top-1/Top-3)
- [ ] 混淆矩阵图正确生成

> **通过 MVP 后**: 确认全链路无误，配置 wandb (`wandb login`)，切换 `report_to: wandb`，启动下方完整 checkpoint 搜索。

---

### 2.4 显存水位探测 — Batch Size 动态压测

在启动正式训练前，通过**小步快跑压测法**找到 L20Z 80GB 显存的最佳 `per_device_train_batch_size`，目标是将显存峰值控制在 90%（约 72GB）以内。

**压测策略**: 不需要跑完整 Epoch。设置 `max_steps: 15` + `logging_steps: 1`，从保守值逐步上探，每次只跑 15 步即可触发完整前向/反向传播并测出显存峰值。

#### 压测脚本: `scripts/train/probe_batch_size.sh`

```bash
#!/bin/bash
# LLaMA-Factory 动态 Batch Size 压测 (Qwen3-4B on L20Z 80GB)
# 目的: 探测最大可用 per_device_train_batch_size

BATCH_SIZES=(16 24 32 48 64 72 80)
MODEL_PATH="/root/code/llm_ruozhiba/models/Qwen3-4B-Instruct-2507"
DATASET="ruozhiba_all"
TEMPLATE="qwen3_nothink"
OUTPUT_DIR="saves/probe_tmp"
MAX_STEPS=15

echo "开始 L20 80GB 显存极限压测..."
echo "测试阶梯: ${BATCH_SIZES[*]}"
echo "--------------------------------------------------"

rm -rf ${OUTPUT_DIR}

for BS in "${BATCH_SIZES[@]}"; do
    echo "正在压测 per_device_train_batch_size = ${BS} ..."

    cat <<EOF > configs/probe_tmp.yaml
model_name_or_path: ${MODEL_PATH}
trust_remote_code: true
stage: sft
do_train: true
finetuning_type: lora
lora_target: all
lora_rank: 8
dataset: ${DATASET}
template: ${TEMPLATE}
cutoff_len: 2048
output_dir: ${OUTPUT_DIR}
overwrite_output_dir: true
per_device_train_batch_size: ${BS}
gradient_accumulation_steps: 1
learning_rate: 1.0e-4
max_steps: ${MAX_STEPS}
logging_steps: 1
bf16: true
report_to: none
EOF

    CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/probe_tmp.yaml > probe_log.txt 2>&1

    if [ $? -eq 0 ]; then
        echo "Batch Size ${BS} 压测通过"
        grep -i "peak memory" probe_log.txt | tail -n 1
        echo "--------------------------------------------------"
    else
        echo "Batch Size ${BS} 压测失败 (OOM 或致命错误)"
        echo "安全临界值为上一个成功的 Batch Size"
        echo "详情: probe_log.txt"
        break
    fi
done

rm -f configs/probe_tmp.yaml
echo "压测流程结束。"
```

**操作步骤**:
1. 运行压测: `bash scripts/train/probe_batch_size.sh`
2. 另起终端实时监控: `watch -n 0.5 nvidia-smi`
3. 测出安全临界值后，配合 `gradient_accumulation_steps` 固定有效批次大小，写入 `configs/qwen3_4b_base.yaml`

**结果应用示例**: 若 BS=64 通过、BS=72 OOM，则正式配置:
```yaml
per_device_train_batch_size: 64
gradient_accumulation_steps: 1   # 有效 batch size = 64
```

---

### 2.5 训练配置与启动

#### 实验矩阵 — Checkpoint-Based 超参搜索

遵循"先完成再完美"原则，聚焦最容易出成果的超参区间。仅执行 **Rank=8** 和 **Rank=16** 两组实验，配合 checkpoint 生成 **10 个模型权重点**，统一使用 `learning_rate: 1.0e-4`，保证基准实验的稳定交付。

> **Rank=32 备选**: 若 Rank=8 与 Rank=16 的效果差异不显著，可追加 Rank=32 实验作为扩展对比。

**Qwen3-4B Checkpoint 训练矩阵** (以 `ruozhiba_all` 全量数据为基础):

| 训练 Run | Rank | `lora_alpha` | 最大 Epoch | 保存的 Checkpoint | GPU |
|----------|------|--------------|------------|-------------------|-----|
| **Run A** | 8 | 16 | 7 | epoch-3, 4, 5, 6, 7 | GPU 0 |
| **Run B** | 16 | 32 | 7 | epoch-3, 4, 5, 6, 7 | GPU 1 |

**产出 Checkpoint 路径** (共 10 个):

```
saves/qwen3-4b/lora/r8/checkpoint-epoch-{3,4,5,6,7}
saves/qwen3-4b/lora/r16/checkpoint-epoch-{3,4,5,6,7}
```

**维度解读**:
- **Epoch 维度** ($3 \sim 7$): 观察迭代次数增加时，模型对特定幽默风格的记忆力与泛化能力的平衡点。checkpoint 方式提供 5 个连续采样点（3/4/5/6/7），loss 曲线更平滑
- **Rank 维度** ($8, 16$): 研究参数更新空间的拓扑维度对 JSON 结构化输出准确率的影响
- `lora_alpha` 统一保持 $2 \times \text{rank}$

**并行调度** (双卡同步):
```
GPU 0: Run A (rank=8, 7 epochs)
GPU 1: Run B (rank=16, 7 epochs)
```

> **关键配置**: 在 `qwen3_4b_base.yaml` 中设置 `save_strategy: epoch`，并设置 `save_total_limit: 0`（不限制保存数量，保留所有 epoch checkpoint）。评估时从 epoch ≥ 3 的 checkpoint 中选取。

**数据规模对比实验** (可选, 在 checkpoint 搜索最优参数确定后):

| 实验 | 模型 | 数据集 | 超参 | 目的 |
|------|------|--------|------|------|
| **Run Best-Last3** | Qwen3-4B | 近三年 (ruozhiba_last3) | 最优 Rank × Epoch | 对比数据规模影响 |
| **Run Best-All** | Qwen3-4B | 全量 (ruozhiba_all) | 最优 Rank × Epoch | 对比数据规模影响 |
| **Run R32** | Qwen3-4B | 全量 (ruozhiba_all) | rank=32, 最优 Epoch | 扩展 Rank 维度对比 (可选) |
| **Run MoE** | Qwen3-30B-A3B | TBD | TBD | 对比模型规模 / MoE |

#### 配置文件模板: `configs/qwen3_4b_base.yaml`

所有 4B 实验共享的基础配置，通过命令行覆盖变量:

```yaml
### model
model_name_or_path: /root/code/llm_ruozhiba/models/Qwen3-4B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_target: all                      # 所有线性层

### dataset
dataset: ruozhiba_all
template: qwen3_nothink               # Qwen3 Instruct 纯指令模板, 无 <think> 标签
cutoff_len: 2048
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
logging_steps: 10
save_strategy: epoch                  # 每个 epoch 自动保存 checkpoint
save_total_limit: 0                   # 不限制保存数量，保留所有 checkpoint
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: wandb                      # Weights & Biases 可视化 (MVP 阶段可改为 none)

### train
per_device_train_batch_size: 8        # 默认值, 压测后替换为实际安全临界值
gradient_accumulation_steps: 1        # 配合压测结果调整, 固定有效 batch size
learning_rate: 1.0e-4
num_train_epochs: 7                   # 统一训练 7 个 epoch, 通过 checkpoint 对比
lr_scheduler_type: cosine
warmup_ratio: 0.1
seed: 42                              # 固定随机种子, 保证 val_size 切分一致
bf16: true
ddp_timeout: 180000000

### eval
val_size: 0.05
per_device_eval_batch_size: 8
eval_strategy: steps
eval_steps: 100
```

#### 启动脚本: `scripts/train/run_training.sh`

```bash
#!/bin/bash
# Checkpoint-Based Training: 每个 Rank 训练 7 epochs, 自动保存所有 epoch checkpoint
# 用法:
#   bash scripts/train/run_training.sh 0 8    # GPU 0, rank=8
#   bash scripts/train/run_training.sh 1 16   # GPU 1, rank=16
#
# 并行调度:
#   终端 1: bash scripts/train/run_training.sh 0 8
#   终端 2: bash scripts/train/run_training.sh 1 16

GPU_ID=${1:?Usage: $0 <GPU_ID> <RANK>}
RANK=${2:?Usage: $0 <GPU_ID> <RANK>}
ALPHA=$((RANK * 2))
OUTPUT_DIR="saves/qwen3-4b/lora/r${RANK}"

echo "=== GPU ${GPU_ID}: rank=${RANK} alpha=${ALPHA} epochs=7 ==="
echo "=== Checkpoints will be saved at: ${OUTPUT_DIR}/checkpoint-epoch-* ==="

CUDA_VISIBLE_DEVICES=${GPU_ID} llamafactory-cli train configs/qwen3_4b_base.yaml \
  --lora_rank ${RANK} \
  --lora_alpha ${ALPHA} \
  --output_dir ${OUTPUT_DIR} \
  --run_name "Qwen3-4B-Ruozhiba-R${RANK}"  # 隔离 wandb Run, 防止双卡日志覆写
```

> **Wandb 日志隔离**: 双卡并行训练共享同一个 `qwen3_4b_base.yaml`，若不通过 `--run_name` 区分，wandb 可能将两个 Run 的 Loss 曲线交织在同一仪表盘中。动态注入 `R8` / `R16` 后缀确保每个实验拥有独立的 Run 记录。

#### 配置文件: `configs/qwen3_30b_moe.yaml` (Run MoE — 后续)

```yaml
### model
model_name_or_path: /path/to/Qwen3-30B-A3B-Instruct-2507
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 16
lora_target: all                      # 含专家层的 up_proj, down_proj
lora_alpha: 32

### dataset
dataset: ruozhiba_all                 # 取决于 4B 对比结论
template: qwen3_nothink
cutoff_len: 2048
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: saves/qwen3-30b-a3b/lora/sft
logging_steps: 10
save_steps: 500
plot_loss: true
overwrite_output_dir: true

### train
per_device_train_batch_size: 1        # MoE 模型显存占用较高
gradient_accumulation_steps: 8        # 有效 batch size = 1 × 8 = 8
learning_rate: 5.0e-5                 # MoE 建议略低于 Dense (防止专家塌陷)
num_train_epochs: 3.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000
```

**MoE 微调注意事项**:
- 使用 `trust_remote_code: true` 加载原生负载均衡逻辑
- 学习率建议 `5e-5`（低于 Dense 模型的 `1e-4`），防止专家塌陷
- **强制配置** `moe_aux_loss_coeff: 0.01`（负载均衡辅助损失系数）。MoE 在短文本分类场景易出现"路由偷懒"——所有输入倾向于激活同一个通用专家。该系数强制模型均衡调用所有专家层，保障知识宽度的有效更新
- 不使用 ZeRO-3（与 LoRA 兼容性差），推荐 ZeRO-2 或不用 DeepSpeed
- 合并权重时必须使用 LLaMA-Factory 的 `export` 脚本
- 通过 TensorBoard/wandb 监控 `Load Balance Loss`，若该值趋于 0 或突发尖峰，说明发生专家塌陷
- **防塌陷验证**: 训练前 10 步使用 `max_steps: 10` 试跑，确认 Load Balance Loss 处于合理区间 ($0.001 \sim 0.1$)，排除配置错误

---

### 2.6 训练监控

- 观察 `logging_steps: 10` 输出的 loss 变化
- 如有验证集（`val_size: 0.05`），关注 eval_loss 是否上升（过拟合信号）
- 训练完成后在 `output_dir` 生成 `training_loss.png`（`plot_loss: true`）
- **MoE 专项**: 监控 `Load Balance Loss`，预警专家塌陷
- **Warmup 节奏观察**: 当前 `warmup_ratio: 0.1` 配合 ~2800 条数据和 7 epochs，总 Step 数较少。在 Run A (Rank=8) 启动后，通过 wandb 或终端日志**重点观察前 30-50 步的 loss 走势**。若 loss 在预热阶段结束后仍卡在高位不下降，说明预热过快/过慢，可微调为固定的 `warmup_steps: 50`（替换 `warmup_ratio`）

### 2.7 权重合并 (LoRA Merge)

`sglang` 原生针对完整的 Dense/MoE 模型进行高度优化的算子融合（PagedAttention、RadixAttention），直接提供完整的合并权重能获得最极致的推理速度。因此在选定最优 Checkpoint 后，需先将 LoRA adapter 与基座模型合并，导出为独立的完整模型。

**操作逻辑**: 在 Phase 2.6 确定最优 Checkpoint（例如 Rank=16, Epoch=5）后，利用 LLaMA-Factory 的 `export` 命令合并权重。

#### 新建配置文件: `configs/qwen3_4b_merge.yaml`

```yaml
### model
model_name_or_path: /root/code/llm_ruozhiba/models/Qwen3-4B-Instruct-2507
adapter_name_or_path: saves/qwen3-4b/lora/r16/checkpoint-epoch-5  # 替换为最优 checkpoint 路径
template: qwen3_nothink
finetuning_type: lora

### export
export_dir: models/Qwen3-4B-Ruozhiba-Merged
export_size: 5
export_device: auto
export_legacy_format: false
```

**执行命令**:

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export configs/qwen3_4b_merge.yaml
```

合并完成后，`models/Qwen3-4B-Ruozhiba-Merged` 即可作为独立的标准 HuggingFace 模型被 `sglang` 直接加载。

**注意事项**:
- 合并后的模型体积约 8-9 GB（与基座模型相当），确保磁盘空间充足
- 合并操作是只读的，不会修改原始 adapter 或基座模型
- 若需评估多个 checkpoint，可修改 `adapter_name_or_path` 后多次执行，输出到不同目录

---

## Phase 3: 评估

### 3.1 高并发推理脚本 (sglang)

#### 新脚本: `scripts/inference/inference_eval.py`

废弃基于 `transformers` + `PeftModel` 的逐条慢速推理方案，改用 `sglang` 的离线批量生成（Offline Batch Generation），利用 PagedAttention 和 RadixAttention 榨干 L20Z 显存带宽。

> **运行环境**: 使用系统自带的 `/usr/bin/python3`（已预装 `sglang`），而非 `env_sft` 虚拟环境。

```
输入:
  - 测试集: data/CQIA/ruozhiba_cqia_classified_v2.json (240 条)
  - 合并模型: models/Qwen3-4B-Ruozhiba-Merged (Phase 2.7 输出)
  - 基座模型: models/Qwen3-4B-Instruct-2507 (Baseline 对比)
输出:
  - 推理结果: results/results_best.json (合并模型)
  - 基线结果: results/results_baseline.json (原始基座)
```

**代码逻辑骨架**:

```python
import sglang as sgl
import json
import yaml

# 1. 启动 sglang 引擎 (单卡推理)
# 注意: 启动前务必通过 nvidia-smi 确认目标 GPU 完全空闲
engine = sgl.Engine(
    model_path="models/Qwen3-4B-Ruozhiba-Merged",
    tp_size=1,              # Tensor Parallelism = 1, 单卡; 双卡可设为 2
    mem_fraction_static=0.8  # 限制预分配显存为 80%, 防止残余进程导致 OOM
)

# 2. 读取中心化 Prompt (确保训练/推理一致)
with open("configs/prompts.yaml") as f:
    SYSTEM_PROMPT = yaml.safe_load(f)["system_prompt"]

# 3. 构造批量输入 (一次性打包 240 条)
prompts = []
for item in test_data:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": item["instruction"]}
    ]
    formatted_prompt = engine.get_tokenizer().apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts.append(formatted_prompt)

# 4. 设置确定性采样参数 (temperature=0 即 Greedy Decoding)
sampling_params = {"temperature": 0.0, "max_new_tokens": 1500}

# 5. 执行高并发推理 (sglang 自动调度 PagedAttention)
outputs = engine.generate(prompts, sampling_params)

# 6. 保存结果
for i, out in enumerate(outputs):
    test_data[i]["model_output"] = out["text"]

with open("results/results_best.json", "w", encoding="utf-8") as f:
    json.dump(test_data, f, ensure_ascii=False, indent=2)

engine.shutdown()
```

**Before/After 对比**: 同一脚本先加载**原始基座模型**推理 → `results_baseline.json`，再加载**合并模型**推理 → `results_best.json`。两次推理使用完全相同的 `SYSTEM_PROMPT` 和 `sampling_params`。

> **推理确定性**: `temperature=0.0` 在 sglang 中等价于 Greedy Decoding，无需额外设置 `do_sample` 参数。sglang 不会为此产生 warning。

> **性能优势**: 相比 transformers 逐条推理，sglang 批量推理 240 条的端到端耗时可缩短 5-10x（得益于 continuous batching + PagedAttention 的显存复用）。

> **显存预分配注意**: sglang 启动时默认预分配 ~90% 空闲显存给 KV Cache 池。若训练刚结束或有残余进程占用显存，极易触发 OOM。**启动前务必 `nvidia-smi` 确认目标 GPU 完全空闲**。代码中已通过 `mem_fraction_static=0.8` 限制预分配上限为 80%，为系统保留缓冲空间。

> **Tokenizer 模板安全性**: Phase 2.7 通过 `llamafactory-cli export` 导出的合并模型已包含完整的 `tokenizer_config.json`（含 Qwen3 指令模板），`engine.get_tokenizer().apply_chat_template()` 会直接使用模型目录内的原生模板，无需担心与 LLaMA-Factory 内部模板逻辑冲突。

---

### 3.2 定量指标 — 两阶段 JSON 评估协议

#### 新脚本: `scripts/viz/eval_metrics.py`

```
输入:
  - 金标准: data/CQIA/ruozhiba_cqia_classified_v2.json
  - 推理结果: results/results_run_*.json, results_baseline.json
输出:
  - 评估报告: results/eval_report.json
  - 混淆矩阵图: results/confusion_matrix_*.png
```

**两阶段评估流程**:

#### Stage 1: 原生 JSON 完整性 (格式遵循能力)

记录模型**原始输出**是否能通过标准 `json.loads()` 解析，衡量指令遵循能力:

| 指标 | 计算方式 | 函数/库 |
|------|----------|---------|
| **JSON 遵循率 (strict)** | 原始输出直接 `json.loads()` 成功率 | `json` |
| **JSON 遵循率 (tolerant)** | `re.search(r'\{.*\}', text, re.DOTALL)` 提取后 `json.loads()` 成功率 | `json`, `re` |
| **Valid Sample Rate (VSR)** | 经 `json-repair` 修复后可解析的比例 | `json_repair` |

> **VSR 阈值**: 若 VSR < 80%，该组超参下的模型虽然逻辑可能正确，但指令遵循能力完全不可用。

#### Stage 2: 逻辑准确率 (内容评估)

对 Stage 1 中解析/修复成功的样本，提取 `top3_categories` 进行逻辑准确率计算:

| 指标 | 计算方式 | 函数/库 |
|------|----------|---------|
| **Top-1 准确率** | 模型预测的 `top3_categories[0].category` == 金标准 `top3_categories[0].category` | 原生 Python |
| **Top-3 命中率** | 金标准 Top-1 类别是否出现在模型预测的 Top-3 中 | 原生 Python |
| **置信度 MAE** | 见下方公式 | `numpy` |

**JSON 失效惩罚机制 (Maximum Penalty)**:

若样本 JSON 解析失败且 `json-repair` 无法修复，该样本按最大误差计入:
- Top-1 准确率: 计为 $0$ (错误)
- 置信度 MAE: 计为 $1.0$ (最大误差)

$$\text{MAE}\_{\text{final}} = \frac{1}{N} \left( \sum_{i \in \text{valid}} |c_{pred,i} - c_{gold,i}| + \sum_{j \in \text{invalid}} 1.0 \right)$$

**惩罚影响量化 (Format Failure Penalty Impact)**:

为了在报告中透明化展示格式错误对总分的拖累程度，新增量化字段:

$$\text{Format\_Failure\_Penalty\_Impact} = \text{MAE}_{\text{final}} - \text{MAE}_{\text{valid\_only}}$$

其中 $\text{MAE}_{\text{valid\_only}}$ 仅基于解析成功的样本计算。该值越大，说明格式错误对总分的拖累越严重，为报告 Error Analysis 章节提供数据支撑。

#### 复合指标 — Strict / Repaired Accuracy

将"格式规范性"和"逻辑理解力"解耦为两项核心工程指标:

| 指标 | 定义 | 意义 |
|------|------|------|
| **Strict Accuracy** | 原始输出 `json.loads()` 成功 **且** Top-1 分类正确 | 模型在特定 system prompt 下的端到端工程可靠性 |
| **Repaired Accuracy** | 经 `json-repair` 修复后 Top-1 分类正确 | 模型的逻辑智能上限（剥离格式噪声后的真实分类能力） |

> **诊断价值**: 若 Strict 与 Repaired 差距巨大，说明微调的痛点在于指令格式遵循；若两者都低，说明模型未学会解构幽默。

#### 置信度校准分析 (Confidence Calibration)

当模型输出 `confidence_score` 时，除 MAE 外还需分析**校准度**——模型是否"知道自己不知道":

| 指标 | 计算方式 | 意义 |
|------|----------|------|
| **正确时平均置信度** | Top-1 命中样本的 `confidence_score` 均值 | 正确判断的自信程度 |
| **错误时平均置信度** | Top-1 未命中样本的 `confidence_score` 均值 | 错误判断的过度自信程度 |

> **诊断价值**: 若错误时置信度高于 $0.8$，说明模型处于"过度自信 (Overconfident)"状态，微调可能引发虚假确信；若错误时置信度低于 $0.4$，说明模型具备一定的不确定性感知能力。该分析为报告 Error Analysis 章节提供深度数据支撑。

#### 混淆矩阵 (Confusion Matrix)

对 Top-1 分类结果生成 8×8 混淆矩阵，同时输出**原始计数**和**归一化**两个版本:

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(gold_labels, pred_labels, labels=CATEGORIES)

# 原始计数矩阵
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.savefig('results/confusion_matrix_counts.png')

# 归一化矩阵 (按行归一化, 消除类别分布不均的干扰)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_normalized, annot=True, fmt='.2f', xticklabels=CATEGORIES, yticklabels=CATEGORIES)
plt.savefig('results/confusion_matrix_normalized.png')
```

**分析要点**:
- **归一化矩阵**: 数值为 $0.0 \sim 1.0$ 的区间，直观暴露各类别的真实召回率，不受测试集分布不均影响
- **对角线密度**: 各类别的召回率
- **离群点**: 如"谐音梗"被误认为"文字游戏"，讨论定义边界模糊性

**输出格式**:
```json
{
  "model": "Qwen3-4B-Instruct-2507 + LoRA (r16/checkpoint-epoch-5)",
  "test_set_size": 240,
  "stage1_json_strict": 0.88,
  "stage1_json_tolerant": 0.93,
  "stage1_vsr": 0.97,
  "stage2_top1_accuracy": 0.62,
  "stage2_top3_hit_rate": 0.85,
  "stage2_confidence_mae": 0.08,
  "stage2_top1_accuracy_with_penalty": 0.60,
  "stage2_mae_with_penalty": 0.11,
  "format_failure_penalty_impact": 0.03,
  "strict_accuracy": 0.55,
  "repaired_accuracy": 0.62,
  "calibration": {
    "avg_confidence_when_correct": 0.72,
    "avg_confidence_when_wrong": 0.58
  }
}
```

---

### 3.3 LLM-as-a-Judge (定性评估)

#### 新脚本: `scripts/llm_judge.py`

从测试集中抽取 20 条，使用外部 LLM（DeepSeek/Claude）作为裁判评分。

**评分维度** (1-10 分):
1. **逻辑准确度**: 是否正确识别了段子的核心笑点和逻辑机制？
2. **幽默感捕捉**: 是否理解了段子的幽默内涵而非仅字面分析？
3. **分析深度**: thought_process 是否深入拆解了多层幽默属性？

**双盲互换对决 (Position Swapping)**:

大模型存在**位置偏置 (Position Bias)**——倾向于给排在前面的选项打更高分。为消除此偏差，每条测试样本进行**两轮评分**:

| 轮次 | 分析 A | 分析 B |
|------|--------|--------|
| **Round 1** | 微调模型输出 | 金标准参考 |
| **Round 2** | 金标准参考 | 微调模型输出 |

**最终得分** = (Round 1 对微调模型的评分 + Round 2 对微调模型的评分) / 2

> 该机制将 API 调用量翻倍 (20 条 × 2 轮 = 40 次)，但能极大提升评估结果的学术公信力。

**Prompt 模板**:
```
请扮演一名资深的中文互联网文化评论员和语言学家。
以下是一段弱智吧段子及其两种分析结果。

【段子原文】
{joke_text}

【分析 A】
{analysis_a}

【分析 B】
{analysis_b}

请从以下三个维度对"分析 A"进行评分 (1-10)，并简要说明理由：
1. 逻辑准确度
2. 幽默感捕捉
3. 分析深度
```

> **注意**: Prompt 中不标注"微调模型"或"金标准"的身份标签，裁判 LLM 仅基于内容质量打分。Round 1 和 Round 2 通过互换 `analysis_a` / `analysis_b` 的内容实现双盲。

**评分方差控制 — 确定性推理**:

裁判 LLM 自身的输出也存在随机方差（同一输入可能给 7 分或 8 分）。为消除此噪声，强制使用确定性解码:

```python
response = client.chat.completions.create(
    model="deepseek-chat",  # 或 claude-opus
    messages=messages,
    temperature=0.0,         # 完全确定性输出
    top_p=0.001,             # 框架允许的最小值 (部分 API 不接受 0.0)
    max_tokens=800,
)
```

**评分锚点 (Scoring Anchors)**: 在 Prompt 中给出分值锚点说明，提高评分的区分度和一致性:

```
评分标准锚点:
- 1-3 分: 分析完全偏离段子核心, 或仅进行字面翻译, 未触及幽默机制
- 4-5 分: 识别了表面笑点, 但缺少深层逻辑拆解, 分类有明显偏差
- 6-7 分: 正确识别主要幽默机制, 分类合理, 但分析深度一般
- 8-9 分: 精准捕捉多层幽默属性, 分类准确且理由充分, 展现文化理解力
- 10 分: 分析堪称教科书级别, 兼具语言学深度和文化洞察力
```

> 锚点说明能将裁判的评分从「我觉得大概是 7 分」变为「按标准这属于 6-7 分档，给 7」，显著降低主观漂移。

---

## Phase 4: 报告与交付

### 4.1 报告内容 (对照 assignment.md 要求)

| 报告章节 | 内容 | 来源 |
|----------|------|------|
| **3.1 SFT Target** | 弱智吧幽默分类: 8 类 Top-3 + 思考过程 | readme.md, 2-4 个 Before/After 示例 |
| **3.2 Dataset** | 数据来源、去重过程、ShareGPT 格式、数据量 | dedup_report.json, data stats |
| **3.3 Training Setup** | Qwen3-4B, LoRA Checkpoint-Based 超参搜索 (Rank 8/16), bf16 | 训练 yaml 配置, Rank × Epoch checkpoint 结果表 |
| **3.4 Loss Curves** | 2 组训练的 training_loss 对比, eval_loss 曲线, 各 checkpoint 对比 | output_dir/training_loss.png |
| **3.5 Before/After** | 3-5 条测试集 Baseline vs SFT 对比 | results_baseline.json vs results_best.json |
| **混淆矩阵** | 8×8 分类混淆矩阵热力图 + 误分类分析 | confusion_matrix_*.png |

### 4.2 交付物清单

```
提交内容:
├── report.pdf                         # 实验报告
├── scripts/
│   ├── classify_cqia_updated.py       # CQIA thought_process 补全
│   ├── dedup_test_vs_train.py         # 去重脚本
│   ├── build_sft_data.py              # ShareGPT 格式化
│   ├── run_training.sh                # Checkpoint-Based 训练启动脚本
│   ├── probe_batch_size.sh            # 显存水位 Batch Size 压测脚本
│   ├── inference_eval.py              # sglang 高并发推理脚本
│   ├── eval_metrics.py                # 两阶段 JSON 评估 + 混淆矩阵
│   └── llm_judge.py                   # LLM-as-Judge
├── configs/
│   ├── prompts.yaml                   # 中心化 SYSTEM_PROMPT (训练/推理/评估共用)
│   ├── qwen3_4b_mvp.yaml              # MVP 配置 (rank=8, epoch=3, 本地保存)
│   ├── qwen3_4b_base.yaml             # 4B 基础配置 (checkpoint 搜索共用)
│   ├── qwen3_4b_merge.yaml            # LoRA 权重合并导出配置
│   └── qwen3_30b_moe.yaml             # MoE 训练配置 (后续)
├── models/
│   └── Qwen3-4B-Ruozhiba-Merged/      # 合并后的完整模型 (Phase 2.7 产出)
├── results/
│   ├── results_baseline.json          # 基座模型推理结果 (sglang)
│   ├── results_best.json              # 合并模型推理结果 (sglang)
│   ├── eval_report.json               # 评估报告
│   └── confusion_matrix_*.png         # 混淆矩阵图
└── README.md                          # 复现指南
```

---

## 文件变更清单

### 需要新建的文件

| 文件 | 用途 |
|------|------|
| `scripts/data/classify_cqia_updated.py` | 为 CQIA 数据补充 thought_process |
| `scripts/data/classify_cqia_updated_config.yaml` | 配置文件 |
| `scripts/data/dedup_test_vs_train.py` | CQIA vs 贴吧去重 |
| `scripts/data/build_sft_data.py` | 数据格式化为 ShareGPT |
| `scripts/train/run_training.sh` | Checkpoint-Based 训练启动脚本 |
| `scripts/train/probe_batch_size.sh` | 显存水位 Batch Size 压测脚本 |
| `scripts/inference/inference_eval.py` | sglang 高并发批量推理 |
| `scripts/viz/eval_metrics.py` | 两阶段 JSON 评估 + 混淆矩阵 |
| `scripts/llm_judge.py` | LLM-as-a-Judge 评估 |
| `configs/prompts.yaml` | 中心化 SYSTEM_PROMPT (训练/推理/评估共用) |
| `configs/qwen3_4b_base.yaml` | 4B 基础配置 (checkpoint 搜索共用) |
| `configs/qwen3_4b_mvp.yaml` | MVP 配置 (rank=8, epoch=3, 本地保存) |
| `configs/qwen3_4b_merge.yaml` | LoRA 权重合并导出配置 |
| `configs/qwen3_30b_moe.yaml` | MoE 训练配置 |

### 需要修改的文件

| 文件 | 变更 |
|------|------|
| `LLaMA-Factory/data/dataset_info.json` | 注册 `ruozhiba_last3`, `ruozhiba_all` |

### 不需要修改的文件

- `scripts/data/classify_jokes.py` — 保持不变
- `scripts/data/classify_cqia.py` — 保持不变（新建 _updated 版本）
- `scripts/data/filter_duplicates.py` — 保持不变（新建专用去重脚本）
- `data/tieba/best*_classified.json` — 原始数据不修改（去重操作生成新文件）
- `data/CQIA/ruozhiba_cqia_classified.json` — 原始数据不修改（补全生成 _v2）

---

## 依赖与环境

### 虚拟环境 `env_sft` (训练 + 数据处理)

```bash
source env_sft/bin/activate
```

### 系统 Python `/usr/bin/python3` (sglang 推理)

`sglang` 已预装在系统环境中，推理脚本 (`inference_eval.py`) 需使用 `/usr/bin/python3` 运行，不使用 `env_sft`。

```bash
/usr/bin/python3 scripts/inference/inference_eval.py ...
```

### 已安装依赖

```
llamafactory[metrics], accelerate
openai, tenacity, tqdm, python-dotenv, pyyaml
sglang (系统 Python, 已预装)
```

### 可能需要安装

```bash
# env_sft 虚拟环境 (训练 + 数据处理 + 评估)
uv pip install numpy        # 用于 eval_metrics.py 中的 MAE 计算
uv pip install json-repair  # 用于 Stage 2 JSON 修复 (Two-Stage Eval)
uv pip install scikit-learn # 用于 confusion_matrix 计算
uv pip install seaborn matplotlib  # 用于混淆矩阵热力图可视化
uv pip install wandb        # 用于训练过程可视化 (Weights & Biases)
```

---

## 执行顺序 Checklist

### Phase 1: 数据工程

- [ ] **1.1** 编写 `classify_cqia_updated.py` + config
- [ ] **1.1** 确认 `tenacity` 重试 + `Semaphore(3)` 并发限制已实现
- [ ] **1.1** 确认 Prompt Caching 已启用 (若 API 平台支持)
- [ ] **1.1** 运行补全 → `ruozhiba_cqia_classified_v2.json`
- [ ] **1.1** 验证: 所有 240 条都有 `thought_process` 字段
- [ ] **1.2** 编写 `dedup_test_vs_train.py`
- [ ] **1.2** 运行去重 → 输出 `dedup_report.json`
- [ ] **1.3** 统计去重后各年数据量，确认无数据泄露

### Phase 2: SFT 训练

- [ ] **2.1** 创建 `configs/prompts.yaml` (中心化 SYSTEM_PROMPT)
- [ ] **2.1** 编写 `build_sft_data.py`
- [ ] **2.1** 生成 `ruozhiba_last3.json` + `ruozhiba_all.json`
- [ ] **2.1** 验证: JSON 格式正确，conversations 结构完整
- [ ] **2.1** 确认 `json.dumps(ensure_ascii=False)` 已使用，中文未被转义
- [ ] **2.2** 在 `dataset_info.json` 中注册数据集
- [ ] **2.3** 创建 `configs/qwen3_4b_mvp.yaml`
- [ ] **2.3** MVP 训练: `CUDA_VISIBLE_DEVICES=0 llamafactory-cli train configs/qwen3_4b_mvp.yaml`
- [ ] **2.3** MVP 推理 + 评估: 验证全链路正常
- [ ] **2.4** 运行 `scripts/train/probe_batch_size.sh` 压测最大 Batch Size
- [ ] **2.4** 将压测结果写入 `configs/qwen3_4b_base.yaml`
- [ ] **2.5** 创建 `configs/qwen3_4b_base.yaml` + `scripts/train/run_training.sh`
- [ ] **2.5** 确认 YAML 中 `seed: 42` 已设置, 保证 `val_size` 切分一致
- [ ] **2.5** 配置 wandb: `wandb login`
- [ ] **2.5** 终端 1: `bash scripts/train/run_training.sh 0 8`
- [ ] **2.5** 终端 2: `bash scripts/train/run_training.sh 1 16`
- [ ] **2.5** 确认 wandb 上出现两个独立 Run (`Qwen3-4B-Ruozhiba-R8` / `R16`)
- [ ] **2.6** 观察前 30-50 步 loss，确认 warmup 节奏正常 (必要时切 `warmup_steps: 50`)
- [ ] **2.6** 检查 2 组 loss 曲线，从 10 个 checkpoint 中确定最优 Rank × Epoch 组合
- [ ] **2.7** 创建 `configs/qwen3_4b_merge.yaml`
- [ ] **2.7** 运行 `llamafactory-cli export` 合并最优 LoRA 权重 → `models/Qwen3-4B-Ruozhiba-Merged`

### Phase 3: 评估

- [ ] **3.1** 编写基于 `sglang` 的 `inference_eval.py`
- [ ] **3.1** 确认脚本运行在 `/usr/bin/python3` (含 sglang) 环境下
- [ ] **3.1** 确认目标 GPU 完全空闲 (`nvidia-smi`), 防止 sglang 预分配显存 OOM
- [ ] **3.1** 确认 `sampling_params` 中 `temperature: 0.0` 保证推理确定性
- [ ] **3.1** Baseline 推理 (原始基座) → `results_baseline.json`
- [ ] **3.1** Merged 模型推理 → `results_best.json`
- [ ] **3.2** 编写 `eval_metrics.py` (两阶段 JSON 评估)
- [ ] **3.2** Stage 1: 计算 JSON 遵循率 (strict/tolerant/VSR)
- [ ] **3.2** Stage 2: 计算 Top-1, Top-3, MAE (含惩罚机制)
- [ ] **3.2** 计算置信度校准: 正确/错误样本的平均置信度
- [ ] **3.2** 生成混淆矩阵 → `confusion_matrix_*.png`
- [ ] **3.2** 汇总评估报告 → `eval_report.json`
- [ ] **3.3** 编写 `llm_judge.py` (含双盲互换对决 + `temperature=0` + 评分锚点)
- [ ] **3.3** 抽取 20 条 × 2 轮位置互换评估 → `judge_results.json`

### Phase 4: 报告

- [ ] **4.1** 撰写实验报告 (PDF)
- [ ] **4.2** 整理代码，编写 README
- [ ] **4.2** 确认所有交付物完整

---

## 风险与备选方案

| 风险 | 影响 | 备选方案 |
|------|------|----------|
| CQIA 补全 API 调用限流 | 1.1 延迟 | 降低并发数，增加 sleep_time |
| 去重率过高导致训练数据不足 | 2.1 训练效果差 | 降低模糊去重阈值到 0.95 |
| 4B 模型 JSON 遵循率低 | 3.2 指标差 | 增加 `cutoff_len`，在 system prompt 中强化格式要求; 用 `json-repair` 二次挽救 |
| 30B MoE 模型显存溢出 | 2.3 Run MoE 失败 | 减小 batch size 到 1 + gradient_checkpointing |
| MoE 专家塌缩 | 2.3 MoE 效果差 | 观察 Load Balance Loss, 调整 `moe_aux_loss_coeff` |
| 训练过拟合 (eval_loss 上升) | 3.2 泛化差 | 减少 epoch (7→5→3), 增加 dropout |
| 网格搜索耗时过长 | 2.5 进度延迟 | 已裁剪至 Rank 8/16 两组 + 双卡并行, Rank=32 作为可选扩展 |

---

## 附录

### A. 数据量汇总

| 文件 | 年份 | 条数 | 用途 |
|------|------|------|------|
| `best176_2018_classified.json` | 2018 | 170 | 训练 (全量) |
| `best336_2019_classified.json` | 2019 | 333 | 训练 (全量) |
| `best365_2020_classified.json` | 2020 | 353 | 训练 (全量) |
| `best295_2021_1H_classified.json` | 2021H1 | 286 | 训练 (全量) |
| `best306_2021_2H_classified.json` | 2021H2 | 306 | 训练 (全量) |
| `best365_2022_classified.json` | 2022 | 330 | 训练 (全量) |
| `best365_2023_classified.json` | 2023 | 322 | 训练 (近三年 + 全量) |
| `best365_2024_classified.json` | 2024 | 352 | 训练 (近三年 + 全量) |
| `best365_2025_classified.json` | 2025 | 361 | 训练 (近三年 + 全量) |
| **贴吧小计** | — | **2813** | — |
| `ruozhiba_cqia_classified_v2.json` | — | 240 | **测试集 (不参与训练)** |

### B. 模型信息

| 模型 | 参数量 | 类型 | 路径 | 状态 |
|------|--------|------|------|------|
| Qwen3-4B-Instruct-2507 | 4B | Dense | `/root/code/llm_ruozhiba/models/Qwen3-4B-Instruct-2507` | ✅ 已下载 |
| Qwen3-30B-A3B-Instruct-2507 | 30B (3.3B active) | MoE | TBD | ⏳ 待下载 |

### C. LLaMA-Factory Template

- 使用 `qwen3_nothink` 模板: Qwen3 Instruct 专用，不输出 `<think>` 标签
- 备选 `qwen3`: 含 thinking 标签（不推荐用于分类任务）
