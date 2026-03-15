# Changelog

## 2026-03-15 — 增加分析，Token 长度 EDA & cutoff_len 分析

### 概述

使用 Qwen3-4B-Instruct-2507 tokenizer 对训练集和测试集进行 token 长度分析，评估当前 `cutoff_len: 2048` 是否合理。

### 新增
- `doc/train_test_eda.md` — 训练集/测试集 Token 长度 EDA 探索报告

### 分析方法

- Tokenizer: `Qwen3-4B-Instruct-2507`（vocab_size = 151,643）
- 长度计算: `apply_chat_template` 后的完整序列长度（含 special tokens）
- 训练集: `LLaMA-Factory/data/ruozhiba_all.json`（2,785 条 ShareGPT 对话）
- 测试集: `data/CQIA/ruozhiba_cqia_classified_v2.json`（240 条）

### 关键发现

| 数据集 | Min | Max | Mean | Median | P95 | P99 |
|--------|-----|-----|------|--------|-----|-----|
| 训练集 | 454 | 974 | 583.8 | 579 | 672 | 723 |
| 测试集 | 497 | 996 | 618.1 | 615 | 700 | 757 |

- **100% 样本 ≤ 1024 tokens**（训练集最大 974，测试集最大 996）
- 82.2% 训练样本集中在 513–640 tokens 区间
- `cutoff_len: 2048` 存在约 50% 的冗余序列空间

### cutoff_len 截断影响

| cutoff_len | 训练集截断率 | 测试集截断率 |
|------------|-------------|-------------|
| 512 | 94.0% | 97.9% |
| 768 | 0.6% | 0.8% |
| 1024 | 0.0% | 0.0% |
| 2048 | 0.0% | 0.0% |

### 结论

- `cutoff_len: 1024` 即可覆盖全部样本，零截断 => 训练集最大 974，测试集最大 996 但是我希望留点余量给训练后的模型到2048的输出，毕竟训练后模型可能会生成更长的文本
- ~~从 2048 降至 1024 可减少 attention 计算量至 ~1/4，释放显存用于增大 batch_size~~
- ~~Phase 2.4 的 batch size 压测结果（BS=32 @ cutoff_len=2048）在降低 cutoff_len 后可进一步上探~~

---

## 2025-03-15 — Phase 2.4 显存水位探测 — Batch Size 动态压测

### 概述

通过小步快跑压测法（`max_steps: 15`），在 L20Z 80GB 单卡上逐级上探 `per_device_train_batch_size`，确定正式训练的安全临界值。

### 新增
- `scripts/probe_batch_size.sh` — Batch Size 动态压测脚本（自动激活 venv，自动清理临时文件）

### 压测结果 (GPU 1, Qwen3-4B + LoRA rank=8, cutoff_len=2048)

| Batch Size | 结果 | 备注 |
|-----------|------|------|
| 16 | ✅ Pass | |
| 24 | ✅ Pass | |
| 32 | ✅ Pass | 最大安全值 |
| 48 | ❌ OOM | `torch.OutOfMemoryError`: 需分配 21.74 GiB，仅剩 18.03 GiB |

### 结论

- **最大安全 Batch Size: 32**（跳至 48 时 OOM，中间值 36/40 无需额外探测，32 已为 2 的幂次最优选）
- MVP 阶段 BS=8 时峰值 53.5GB (65.6%)，BS=32 时模型+优化器+梯度刚好占满 ~61 GiB
- 推荐正式训练配置: `per_device_train_batch_size: 32`, `gradient_accumulation_steps: 1`

### 备注
- 压测使用 GPU 1（GPU 0 被 MVP 训练产物占用），结果可复现至任意 L20Z 卡
- 两次独立运行结果一致，BS=32 稳定通过

---

## 2025-03-15 — Phase 2.3 MVP 最小可行链路

### 概述

创建 MVP 训练配置并完成首次 LoRA SFT 训练，验证数据 → 训练全链路通畅。Phase 2.2（数据集注册）已在 2.1 中完成，跳过。

### 新增
- `configs/qwen3_4b_mvp.yaml` — MVP 训练配置（rank=8, epoch=3, lora_alpha=16, 单卡）

### 产出
- `LLaMA-Factory/saves/qwen3-4b/lora/mvp_r8_e3/` — 适配器权重 + 3 个 epoch checkpoint
  - `adapter_model.safetensors` (64MB), `adapter_config.json`
  - `checkpoint-{331,662,993}/` — 每 epoch 自动保存
  - `training_loss.png`, `training_eval_loss.png` — Loss 曲线图

### 训练参数

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-4B-Instruct-2507 |
| 方法 | LoRA (rank=8, alpha=16, target=all) |
| 模板 | qwen3_nothink |
| 数据集 | ruozhiba_all (2,645 train / 140 eval, 5% split) |
| Batch Size | 8 |
| Epochs | 3 |
| LR | 1e-4 (cosine, warmup=0.1) |
| 可训练参数 | 16,515,072 (LoRA) |
| GPU | 单卡 L20Z 80GB (峰值 53.5GB, 65.6%) |

### 训练结果

| 指标 | 值 |
|------|-----|
| 总步数 | 993 |
| 训练时长 | 12m 51s |
| 最终 train_loss | 0.9594 |
| 最终 eval_loss | 0.8820 |
| 吞吐量 | 10.28 samples/s, 1.29 steps/s |

### Eval Loss 轨迹

| Step | Epoch | Eval Loss |
|------|-------|-----------|
| 100 | 0.30 | 1.1137 |
| 200 | 0.60 | 0.9822 |
| 300 | 0.91 | 0.9427 |
| 400 | 1.21 | 0.9184 |
| 500 | 1.51 | 0.9053 |
| 600 | 1.81 | 0.8938 |
| 700 | 2.11 | 0.8886 |
| 800 | 2.42 | 0.8854 |
| 900 | 2.72 | 0.8824 |

### MVP 验证清单

- [x] 训练正常完成，loss 呈持续下降趋势
- [x] `saves/qwen3-4b/lora/mvp_r8_e3/` 下生成 adapter 文件 (64MB safetensors)
- [x] 3 个 epoch checkpoint 完整保存 (331/662/993)
- [x] Loss 曲线图正确生成 (`training_loss.png`, `training_eval_loss.png`)
- [x] 显存峰值 53.5GB (65.6%)，远未触及 OOM

### 备注

- Phase 2.2（数据集注册）已在 Phase 2.1 中同步完成（`dataset_info.json` 已包含 `ruozhiba_last3` 和 `ruozhiba_all`），无需额外操作
- Eval loss 在 epoch 1.5 后趋于平缓（0.905→0.882），未见过拟合迹象，正式训练可安全扩展至 5-7 epochs
- 推理评估脚本 (`inference_eval.py`, `eval_metrics.py`) 将在 Phase 3 中实现

---

## 2025-03-15 — Phase 2.1 ShareGPT 格式化

### 新增
- `configs/prompts.yaml` — 中心化 system prompt 与分类类别配置，供训练/推理/评估统一引用
- `scripts/build_sft_data.py` — 将去重后贴吧数据转换为 LLaMA-Factory ShareGPT 格式

### 修改
- `LLaMA-Factory/data/dataset_info.json` — 注册 `ruozhiba_last3` 和 `ruozhiba_all` 两个数据集

### 产出
- `LLaMA-Factory/data/ruozhiba_all.json` — 全量训练集 (2018-2025)，2785 条对话
- `LLaMA-Factory/data/ruozhiba_last3.json` — 近三年训练集 (2023-2025)，1025 条对话

### 结果摘要
- 输入: 9 个 `best*_classified_dedup.json` (Phase 1.2 产出，2786 条)
- 输出: 2785 条 ShareGPT 对话（跳过 1 条缺失 `thought_process` 的条目: 2021_2H #116）
- 格式验证: 全部 2785 条 `gpt.value` 均为合法 JSON，中文无 `\uXXXX` 转义
- 结构验证: 每条对话包含 system/human/gpt 三轮，角色标签正确
- 近三年: 1025 条 | 全量: 2785 条

---

## 2025-03-15 — Phase 1.3 数据统计与验证

### 验证内容
- 去重前后数据量统计（基于 `data/dedup_report.json`）
- `check_and_repair.py` 对 9 个 `_classified_dedup.json` 完整性校验
- CQIA 测试集 (`ruozhiba_cqia_classified_v2.json`) schema 验证
- 数据泄露再验证（MD5 精确匹配）

### 结果摘要

```
去重前:
  贴吧训练集总量: 2813 条
    近三年 (2023-2025): 1035 条
    全量 (2018-2025):   2813 条
  CQIA 测试集: 240 条

去重后:
  贴吧训练集总量: 2786 条 (移除 27 条重复)
    近三年 (2023-2025): 1025 条
    全量 (2018-2025):   2786 条
    精确匹配 (MD5): 18 条, 模糊匹配 (≥0.9): 9 条
```

| 文件 | 去重前 | 去重后 | 移除 |
|------|--------|--------|------|
| best176_2018 | 170 | 170 | 0 |
| best336_2019 | 333 | 333 | 0 |
| best365_2020 | 353 | 350 | 3 |
| best295_2021_1H | 286 | 283 | 3 |
| best306_2021_2H | 306 | 296 | 10 |
| best365_2022 | 330 | 329 | 1 |
| best365_2023 | 322 | 312 | 10 |
| best365_2024 | 352 | 352 | 0 |
| best365_2025 | 361 | 361 | 0 |
| **合计** | **2813** | **2786** | **27** |

### 完整性验证

- **去重后训练集**: 9/9 文件结构完整（`thought_process` + `top3_categories` 齐全）
- **已知例外**: 2021_2H No.120 "弱智吧今日看点 ·" — 非段子内容，无法分类，`build_sft_data.py` 已跳过
- **CQIA 测试集**: 240/240 条 `thought_process` + `top3_categories` 完整
- **数据泄露再验证**: ✅ 无泄露 — 去重后训练集与测试集无精确重叠

---

## 2025-03-15 — Phase 1.2 去重防污染

### 新增
- `scripts/dedup_test_vs_train.py` — CQIA 测试集 vs 贴吧训练集去重脚本

### 产出
- `data/dedup_report.json` — 去重报告（精确/模糊命中明细）
- `data/tieba/best*_classified_dedup.json` — 9 个去重后训练集文件

### 结果摘要
- 测试集: 240 条 (CQIA, `instruction` 字段)
- 训练集: 2813 → **2786** 条 (移除 27 条重复)
  - 精确命中 (MD5): 18 条
  - 模糊命中 (SequenceMatcher ≥ 0.9): 9 条
- 受影响文件: 5/9 个（2021_1H: -3, 2021_2H: -10, 2020: -3, 2022: -1, 2023: -10）
- 近三年 (2023-2025): 1035 → 1025 条
- 泄露验证: 去重后训练集中无测试集残留

---

## 2025-03-15 — Phase 1.1: CQIA 数据补全 (thought_process 导师蒸馏)

### 概述

为 240 条 CQIA 测试集数据补充 `thought_process` 字段，使其与贴吧训练集的 classification 格式对齐。使用 Claude-Opus-4-6 作为导师模型，对每条 `instruction` 生成深度幽默解构分析。

### 新增文件

| 文件 | 说明 |
|------|------|
| `scripts/classify_cqia_updated.py` | CQIA 数据补全脚本，复用 `classify_jokes.py` 鲁棒性模式 |
| `scripts/classify_cqia_updated_config.yaml` | 配置文件，System Prompt 对齐 `classify_config.yaml` |
| `data/CQIA/ruozhiba_cqia_classified_v2.json` | 输出数据（240 条，含 thought_process） |

### 技术要点

- **System Prompt 对齐**: 使用与贴吧分类相同的 prompt（含 `thought_process` + `top3_categories`），确保训练/测试数据格式一致
- **仅输入 instruction**: 不将 CQIA 的 `output`（正经 AI 解答）作为 LLM 输入，避免干扰分类判断
- **原始字段保留**: `output` 和原有 `top3_categories` 完全不变，仅在 `classification` 中新增 `thought_process`
- **Category Drift 日志**: 记录新旧 Top-1 分类差异（仅日志，不覆盖），发现约 20+ 条存在 drift，属正常现象（prompt 变更导致）
- **断点续传**: JSONL checkpoint 支持中断恢复，首次运行 228/240 成功 → 重试后 239/240 → Item #53（盲文）因 safety filter 手动补写

### 数据质量报告

```
Total items:          240
Schema valid:         True (instruction/output/classification)
thought_process:      240/240 present (0 null, 0 empty)
top3_categories:      240/240 exactly 3 categories
thought_process avg:  ~276 chars (min: 176, max: 456)
Data integrity vs v1: 0 mismatches (instruction/output/top3_categories 完全一致)

Top-1 category distribution:
  奇怪提问: 89    文字游戏: 48    文艺弱智: 34    弱智科学家: 27
  古典弱智: 22    谐音梗: 10      人生态度: 6     地狱笑话: 4
```

### 备注

- Item #53（盲文 Unicode 内容）触发 API safety filter，`thought_process` 为手动补写，内容与已有 `top3_categories` 一致
- 已清理 JSONL checkpoint 中间文件
