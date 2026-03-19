# Upload Package — File Provenance

本目录为最小可复现子集；脚本路径与主仓库 **`scripts/`** 子目录分类一致（`train/`、`inference/`、`viz/`、`data/`）。

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

## configs / data / results

见各目录内说明；默认在**完整仓库根目录**下执行 `upload/scripts/...`，以便访问 `LLaMA-Factory/`、`models/`、`env_sft/`。
