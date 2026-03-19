# 配置文件说明

> 本目录包含项目的所有训练配置、推理 prompt 配置和 LoRA 权重合并配置。

---

## 文件清单

| 文件 | 用途 | 使用阶段 |
|------|------|----------|
| `prompts.yaml` | 中心化 system prompt 与分类类别定义 | 全流程（训练/推理/评估） |
| `qwen3_4b_mvp.yaml` | MVP 最小可行链路训练配置 | Phase 2.3 |
| `qwen3_4b_base.yaml` | 正式训练共享基础配置（全量数据） | Phase 2.5 |
| `qwen3_4b_base_last3.yaml` | 近三年数据训练配置 | Phase 2.8 |
| `qwen3_4b_merge.yaml` | LoRA 权重合并配置模板 | Phase 2.7 |

> `upload/configs/` 当前同步保留以上 4 个核心配置文件，用于最小复现包；与主仓库同名文件保持字段兼容。

---

## 配置详解

### `prompts.yaml` — 统一 Prompt 配置

供 `build_sft_data.py`（训练数据构建）和后续推理/评估脚本引用的中心化配置：

- **`system_prompt`**: 弱智吧幽默解构专家的角色定义，要求输出 `thought_process` + `top3_categories` 的 JSON 格式
- **`categories`**: 8 种幽默分类列表（古典弱智、奇怪提问、弱智科学家、人生态度、文字游戏、地狱笑话、谐音梗、文艺弱智）

### `qwen3_4b_mvp.yaml` — MVP 训练配置

Phase 2.3 首次训练验证全链路通畅的最小配置：

| 参数 | 值 | 说明 |
|------|-----|------|
| 基座模型 | Qwen3-4B-Instruct-2507 | — |
| LoRA rank | 8 | 固定值 |
| LoRA alpha | 16 | rank × 2 |
| batch_size | 8 | 保守设置 |
| epochs | 3 | 快速验证 |
| cutoff_len | 2048 | 训练集实际最大 974 tokens |
| 模板 | qwen3_nothink | 禁用思考模式 |
| 输出目录 | `saves/qwen3-4b/lora/mvp_r8_e3` | 固定路径 |
| report_to | none | 不使用 wandb |

### `qwen3_4b_base.yaml` — 正式训练基础配置

Phase 2.5 双卡并行训练的共享基础配置，通过 CLI 参数覆盖实验变量：

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 16 | OOM 修复后从 32 降为 16 |
| gradient_accumulation | 2 | 有效 BS = 16 × 2 = 32 |
| epochs | 7 | 充分训练 |
| save_strategy | epoch | 每 epoch 保存 checkpoint |
| save_total_limit | 0 | 保留所有 checkpoint |
| report_to | wandb | 训练日志上报 |
| lora_rank / alpha | CLI 覆盖 | `run_training.sh` 动态注入 |
| output_dir | CLI 覆盖 | `run_training.sh` 动态注入 |

**实验矩阵（CLI 覆盖参数）:**

| Run | GPU | lora_rank | lora_alpha | output_dir |
|-----|-----|-----------|------------|------------|
| A | 0 | 8 | 16 | `saves/qwen3-4b/lora/r8` |
| B | 1 | 16 | 32 | `saves/qwen3-4b/lora/r16` |

### `qwen3_4b_base_last3.yaml` — 近三年数据训练配置

与 `qwen3_4b_base.yaml` 基本相同，主要差异：

| 参数 | 值 | 说明 |
|------|-----|------|
| dataset | ruozhiba_last3 | 近三年数据 (1,025 条) |
| eval_strategy | epoch | 按 epoch 评估（因步数较少） |

**实验矩阵（CLI 覆盖参数）:**

| Run | GPU | lora_rank | lora_alpha | output_dir |
|-----|-----|-----------|------------|------------|
| C | 0 | 8 | 16 | `saves/qwen3-4b/lora/r8_last3` |
| D | 1 | 16 | 32 | `saves/qwen3-4b/lora/r16_last3` |

### `qwen3_4b_merge.yaml` — LoRA 权重合并配置

Phase 2.7 将 LoRA adapter 合并回基座模型的模板配置，实际由 `batch_merge.sh` 通过 CLI 覆盖 `adapter_name_or_path` 和 `export_dir` 执行 20 个 checkpoint 的批量合并：

| 参数 | 值 |
|------|-----|
| 基座模型 | Qwen3-4B-Instruct-2507 |
| Adapter 路径 | CLI 覆盖 (模板: `saves/qwen3-4b/lora/r16/checkpoint-415`) |
| 输出目录 | CLI 覆盖 (模板: `models/merged/{tag}`) |
| 输出格式 | safetensors |

**单模型合并命令:**

```bash
CUDA_VISIBLE_DEVICES=0 llamafactory-cli export configs/qwen3_4b_merge.yaml
```

**批量合并 (20 个 checkpoint):**

```bash
bash scripts/train/batch_merge.sh
```

---

## 配置继承关系

```
prompts.yaml              ←─ build_sft_data.py / inference_eval.py (引用 system_prompt)
     │
qwen3_4b_mvp.yaml        ←─ Phase 2.3 MVP 训练 (固定参数)
     │
qwen3_4b_base.yaml       ←─ Phase 2.5 正式训练 全量 (CLI 覆盖 rank/alpha/output_dir)
     │
qwen3_4b_base_last3.yaml ←─ Phase 2.8 正式训练 近三年 (CLI 覆盖 rank/alpha/output_dir)
     │
qwen3_4b_merge.yaml      ←─ Phase 2.7 权重合并 (batch_merge.sh CLI 覆盖 adapter/export_dir)
```