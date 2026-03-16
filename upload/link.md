# File Provenance — Upload Package

> Maps each file in `upload/` to its original source in the project repository.

---

## configs/

| File | Source | Description |
|------|--------|-------------|
| `prompts.yaml` | `configs/prompts.yaml` | Centralized system prompt and 8 humor categories |
| `qwen3_4b_base.yaml` | `configs/qwen3_4b_base.yaml` | LoRA SFT training config — full dataset (ruozhiba_all) |
| `qwen3_4b_base_last3.yaml` | `configs/qwen3_4b_base_last3.yaml` | LoRA SFT training config — last-3-year dataset (ruozhiba_last3) |
| `qwen3_4b_merge.yaml` | `configs/qwen3_4b_merge.yaml` | LoRA weight merge template (CLI-overridden by batch_merge.sh) |

## scripts/

| File | Source | Description |
|------|--------|-------------|
| `build_sft_data.py` | `scripts/build_sft_data.py` | Phase 2.1 — Convert classified tieba data → ShareGPT format |
| `run_training.sh` | `scripts/run_training.sh` | Phase 2.5 — Launch LoRA SFT training with CLI rank/alpha override |
| `batch_merge.sh` | `scripts/batch_merge.sh` | Phase 2.7 — Batch merge 20 LoRA checkpoints → standalone models |
| `inference_eval.py` | `scripts/inference_eval.py` | Phase 3.1 — vLLM offline batch inference (single/batch/multi mode) |
| `batch_inference.sh` | `scripts/batch_inference.sh` | Phase 3.1 — Shell wrapper for 21-model serial inference |
| `eval_metrics.py` | `scripts/eval_metrics.py` | Phase 3.2 — Two-stage JSON evaluation + visualization (11 chart types) |
| `gen_before_after.py` | `scripts/gen_before_after.py` | Phase 3.4 — Auto-select 5 representative before/after samples |

## data/

| File | Source | Description |
|------|--------|-------------|
| `ruozhiba_all.json` | `data/LLaMA-Factory/data/ruozhiba_all.json` | Full training set: 2,785 ShareGPT conversations (2018–2025) |
| `ruozhiba_last3.json` | `data/LLaMA-Factory/data/ruozhiba_last3.json` | Last-3-year training set: 1,025 ShareGPT conversations (2023–2025) |
| `ruozhiba_cqia_classified_v2.json` | `data/CQIA/ruozhiba_cqia_classified_v2.json` | Test set: 240 CQIA samples with gold labels + thought_process |
| `dataset_info.json` | `data/LLaMA-Factory/data/dataset_info.json` | LLaMA-Factory dataset registry (minimal: ruozhiba_all + ruozhiba_last3 only) |

## results/

| File | Source | Description |
|------|--------|-------------|
| `eval_comparison.json` | `results/json/eval_comparison.json` | 21-model cross-comparison table (metrics + best model + all-vs-last3 pairs) |
| `before_after_samples.json` | `results/before_after_samples.json` | 5 qualitative before/after examples (baseline vs r16_e5) |
