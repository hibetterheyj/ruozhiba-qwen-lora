# Upload Package — File Provenance

本目录为最小可复现子集；脚本路径与主仓库 **`scripts/`** 子目录分类一致（`train/`、`inference/`、`viz/`、`data/`）。默认在**完整仓库根目录**下执行 `upload/scripts/...`，以便访问 `LLaMA-Factory/`、`models/`、`env_sft/`。

## scripts/

| 文件 | 主仓库对应 | 说明 |
|------|------------|------|
| `data/build_sft_data.py` | `scripts/data/build_sft_data.py` | Phase 2.1 — ShareGPT 转换（需完整 `data/tieba/`） |
| `train/run_training.sh` | `scripts/train/run_training.sh` | Phase 2.5 — LoRA 训练启动 |
| `train/batch_merge.sh` | `scripts/train/batch_merge.sh` | Phase 2.7 — 批量合并 checkpoint |
| `inference/inference_eval.py` | `scripts/inference/inference_eval.py` | Phase 3.1 — vLLM 批量推理 |
| `inference/batch_inference.sh` | `scripts/inference/batch_inference.sh` | Phase 3.1 — 21 模型串行封装 |
| `viz/eval_metrics.py` | `scripts/viz/eval_metrics.py` | Phase 3.2 — 评估与可视化 |
| `viz/gen_before_after.py` | `scripts/viz/gen_before_after.py` | Phase 3.4 — Before/After 样本 |
| `viz/update_report_media.py` | `scripts/viz/update_report_media.py` | 报告媒体同步：复制关键图表到 `doc/report/lab3_report_latex/media/` |

## configs/

| 文件 | 主仓库对应 | 说明 |
|------|------------|------|
| `configs/prompts.yaml` | `configs/prompts.yaml` | 中心化 system prompt 与 8 类幽默定义 |
| `configs/qwen3_4b_base.yaml` | `configs/qwen3_4b_base.yaml` | 全量数据训练配置 |
| `configs/qwen3_4b_base_last3.yaml` | `configs/qwen3_4b_base_last3.yaml` | 近三年数据训练配置 |
| `configs/qwen3_4b_merge.yaml` | `configs/qwen3_4b_merge.yaml` | LoRA 合并模板 |

## data/

| 文件 | 主仓库来源 | 说明 |
|------|------------|------|
| `data/ruozhiba_all.json` | `LLaMA-Factory/data/ruozhiba_all.json` | 全量训练集（2,785 条 ShareGPT 对话） |
| `data/ruozhiba_last3.json` | `LLaMA-Factory/data/ruozhiba_last3.json` | 近三年训练集（1,025 条 ShareGPT 对话） |
| `data/ruozhiba_cqia_classified_v2.json` | `data/CQIA/ruozhiba_cqia_classified_v2.json` | 240 条 CQIA 测试集金标 |
| `data/dataset_info.json` | `LLaMA-Factory/data/dataset_info.json` | 数据集注册配置 |

## results/

| 文件 | 主仓库来源 | 说明 |
|------|------------|------|
| `results/eval_comparison.json` | `results/json/eval_comparison.json` | 21 模型对比总表 |
| `results/before_after_samples.json` | `results/before_after_samples.json` | 5 个代表性 before/after 样本 |
| `results/r8_loss_curves.json` | `results/training/r8_loss_curves.json` | R8 全量训练的 step-level loss 曲线 |
| `results/r16_loss_curves.json` | `results/training/r16_loss_curves.json` | R16 全量训练的 step-level loss 曲线 |
| `results/r8_last3_loss_curves.json` | `results/training/r8_last3_loss_curves.json` | R8 近三年训练的 step-level loss 曲线 |
| `results/r16_last3_loss_curves.json` | `results/training/r16_last3_loss_curves.json` | R16 近三年训练的 step-level loss 曲线 |

> `upload/results/` 仅保留轻量级 JSON 结果；图表 PDF/PNG 默认由 `upload/scripts/viz/eval_metrics.py` 在仓库根 `results/` 下重新生成。
