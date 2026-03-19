# Lab 3 Report: Chinese Humor Classification via LoRA SFT

**Course**: CSS5120 — March 2026  
**SFT Target**: Task Specialization (8-class humor classification with structured JSON output)

---

## 1. SFT Target Description

### 1.1 Target Behavior

The goal is to fine-tune a language model to analyze Chinese internet humor posts from *Ruozhiba* (弱智吧, a Baidu Tieba forum known for absurdist logic-based jokes) and produce:

1. **`thought_process`**: A chain-of-thought analysis dissecting the humor mechanism, wordplay, or logical fallacy embedded in the joke.
2. **`top3_categories`**: A ranked list of the three most applicable humor categories, each with a confidence score and a reason.

The 8 candidate categories are:

| Category (Chinese) | Category (English) | Description |
|--------------------|--------------------|-------------|
| 古典弱智 | Classic Absurdity | Self-contradictory logic, absurd causality |
| 奇怪提问 | Strange Questions | Seemingly reasonable questions with hidden fallacies |
| 弱智科学家 | Pseudo-Scientist | Pseudo-academic reasoning leading to absurd conclusions |
| 人生态度 | Life Philosophy | Nihilistic or philosophical humor about existence |
| 文字游戏 | Wordplay | Puns, semantic decomposition, character-level manipulation |
| 地狱笑话 | Dark Humor | Jokes touching taboo or dark themes |
| 谐音梗 | Homophone Puns | Humor derived from homophones or similar-sounding words |
| 文艺弱智 | Literary Absurdity | Poetic or literary-styled absurdist expressions |

### 1.2 Motivation

Chinese internet humor, especially from *Ruozhiba*, relies heavily on cultural context, wordplay, and logical paradoxes that standard LLMs struggle to classify correctly. A fine-tuned model could serve as an automated humor taxonomy tool for content moderation, recommendation systems, or cultural NLP research.

### 1.3 Example Prompts and Desired Outputs

**Example 1** — Wordplay (文字游戏):

> **Prompt**: 咖啡严格来说是不是也可以叫豆浆？  
> *(Strictly speaking, can coffee also be called soy milk?)*

> **Desired Output** (abbreviated):
> ```json
> {
>   "thought_process": "The joke decomposes '豆浆' (soy milk) literally: coffee beans are '豆' (beans), and the brewed liquid is '浆' (liquid/pulp), so coffee = bean + liquid = soy milk...",
>   "top3_categories": [
>     {"rank": 1, "category": "文字游戏", "confidence_score": 0.80, "reason": "..."},
>     {"rank": 2, "category": "奇怪提问", "confidence_score": 0.12, "reason": "..."},
>     {"rank": 3, "category": "弱智科学家", "confidence_score": 0.05, "reason": "..."}
>   ]
> }
> ```

**Example 2** — Strange Question (奇怪提问):

> **Prompt**: 石油也是油，为啥没人用它来炒菜？  
> *(Petroleum is also oil, so why doesn't anyone cook with it?)*

> **Desired Output**: Top-1 = 奇怪提问 (Strange Question), exploiting the polysemy of "油" (oil).

**Example 3** — Life Philosophy (人生态度):

> **Prompt**: 失踪是不是丢人的事情？  
> *(Is going missing a "losing face" matter?)*

> **Desired Output**: Top-1 = 文字游戏 (Wordplay) — "失踪" (missing) involves "丢" (lose) + "人" (person), literally "losing a person."

---

## 2. Dataset Source and Preprocessing

### 2.1 Data Sources

| Source | Description | Raw Count |
|--------|-------------|-----------|
| **Baidu Tieba Crawler** | Historical "best of" posts from Ruozhiba (2020–2025), scraped using a custom Tieba post crawler | ~3,200 |
| **CQIA Benchmark** | 240 Ruozhiba samples from the *Chinese Question and Instruction Archive* | 240 |

### 2.2 Modifications (Non-trivial)

The raw data consisted of plain text jokes without any labels, analysis, or structured output. The following modifications were applied:

1. **Teacher Distillation** (CQIA subset): Each of the 240 CQIA samples was processed through Claude Opus to generate `thought_process` + `top3_categories` with confidence scores and reasons. This converted unlabeled text into structured prompt–completion pairs.

2. **Tieba Data Classification**: The ~3,200 crawled posts were classified using a multi-model pipeline with majority voting.

3. **Deduplication**: Test-vs-train deduplication was performed using Jaccard similarity to prevent data contamination. After dedup, **2,785 training samples** remained (1 duplicate removed from 2,786).

4. **ShareGPT Format Conversion**: All samples were converted from raw text into the ShareGPT multi-turn format with system/human/gpt roles, enabling direct consumption by LLaMA-Factory.

5. **Temporal Subset**: A `last3` subset (1,025 samples from 2023–2025) was created to study the impact of data recency vs. data volume.

### 2.3 Final Data Format

Training data uses the ShareGPT format:

```json
{
  "conversations": [
    {"from": "system", "value": "[system prompt with category definitions]"},
    {"from": "human", "value": "咖啡严格来说是不是也可以叫豆浆？"},
    {"from": "gpt", "value": "{\"thought_process\": \"...\", \"top3_categories\": [...]}"}
  ]
}
```

### 2.4 Train/Validation Split

| Dataset | Total | Train (95%) | Eval (5%) | Year Range |
|---------|-------|-------------|-----------|------------|
| `ruozhiba_all` | 2,785 | 2,645 | 140 | 2020–2025 |
| `ruozhiba_last3` | 1,025 | 973 | 52 | 2023–2025 |

**Test set**: 240 CQIA samples (held out, not used in training).

---

## 3. Training Setup

### 3.1 Base Model

**Qwen3-4B-Instruct-2507**

- Selected for strong Chinese language understanding and instruction-following capability
- 4B parameter size fits comfortably on a single 80 GB GPU with LoRA
- Instruct-tuned variant provides a solid baseline for structured output generation

### 3.2 LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Method | LoRA (Low-Rank Adaptation) |
| Target modules | All linear layers |
| LoRA Rank | 8 and 16 (two parallel experiments) |
| LoRA Alpha | 16 (R8) / 32 (R16), maintaining Alpha/Rank = 2.0 |
| Trainable params | 16.5M (R8) / 33.0M (R16) |

### 3.3 Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 |
| LR schedule | Cosine with 10% warmup |
| Epochs | 7 (checkpoints saved per epoch) |
| Effective batch size | 32 (16 per device × 2 gradient accumulation) |
| Max sequence length | 2,048 |
| Precision | bf16 |
| Optimizer | AdamW |
| Seed | 42 |
| Template | `qwen3_nothink` (thinking mode disabled) |

### 3.4 Experiment Matrix

| Experiment | Dataset | LoRA Rank | Train Samples | Steps/Epoch | Total Steps |
|------------|---------|-----------|---------------|-------------|-------------|
| R8 | all (2,785) | 8 | 2,645 | 83 | 581 |
| R16 | all (2,785) | 16 | 2,645 | 83 | 581 |
| R8_last3 | last3 (1,025) | 8 | 973 | 31 | 217 |
| R16_last3 | last3 (1,025) | 16 | 973 | 31 | 217 |

### 3.5 Hardware and Environment

| Component | Specification |
|-----------|--------------|
| GPU | 2× NVIDIA H800 (80 GB VRAM each) |
| Training framework | LLaMA-Factory + Hugging Face Trainer |
| Inference backend | vLLM 0.17.1 |
| Virtual environment | `env_sft` (PyTorch 2.10.0 + CUDA 12.8) |

Two experiments were run in parallel on separate GPUs (R8 on GPU 0, R16 on GPU 1).

---

## 4. Loss Curves / Training Signals

### 4.1 Evaluation Loss Comparison (All-data Experiments)

| Step | Epoch | R8 Eval Loss | R16 Eval Loss |
|------|-------|-------------|---------------|
| 100 | ~1.2 | 1.0295 | 0.9842 |
| 200 | ~2.4 | 0.9258 | 0.9034 |
| 300 | ~3.6 | 0.8988 | 0.8886 |
| 400 | ~4.8 | 0.8885 | **0.8859** |
| 500 | ~6.0 | **0.8870** | 0.8915 |

> **Training loss and eval loss curve plots** are generated by LLaMA-Factory and stored at:
> - `LLaMA-Factory/saves/qwen3-4b/lora/r8/training_loss.png` / `training_eval_loss.png`
> - `LLaMA-Factory/saves/qwen3-4b/lora/r16/training_loss.png` / `training_eval_loss.png`
> - `LLaMA-Factory/saves/qwen3-4b/lora/r8_last3/training_loss.png` / `training_eval_loss.png`
> - `LLaMA-Factory/saves/qwen3-4b/lora/r16_last3/training_loss.png` / `training_eval_loss.png`

Additionally, a combined eval loss trend chart across all 4 experimental groups is available at `results/charts/line_eval_loss.png`.

### 4.2 Observations

1. **Stable convergence**: Both R8 and R16 show monotonic eval loss decrease through epoch ~5, with no anomalous spikes or instability during warmup.

2. **R16 overfitting signal**: R16 reaches its global minimum eval loss (0.8859) at step 400 (~epoch 5), then increases to 0.8915 at step 500 (~epoch 6). This 0.0056 increase indicates mild overfitting — the larger parameter count (33M vs 16.5M) causes the model to begin memorizing training data.

3. **R8 remains stable**: R8's eval loss continues to decrease through step 500, reaching 0.8870 without overfitting within 7 epochs. The lower rank acts as an implicit regularizer.

4. **Minimal gap**: The best eval loss values are nearly identical (R8: 0.8870 vs R16: 0.8859, Δ = 0.12%), suggesting both ranks capture similar task-relevant features.

### 4.3 Learning Rate Schedule

Cosine schedule with 10% warmup (~58 steps): linear ramp from 0 → 1e-4, then cosine decay to ~0 by step 581. No restarts or anomalies observed.

---

## 5. Before vs. After Comparison

We compare the **baseline** (Qwen3-4B-Instruct-2507, no SFT) against the **best SFT model** (r16_e5: LoRA R16, all data, epoch 5) on 5 representative test samples not used in training.

### 5.1 Quantitative Summary

| Metric | Baseline | SFT (r16_e5) | Improvement |
|--------|----------|-------------|-------------|
| Top-1 Accuracy | 0.233 | **0.613** | +163% |
| Top-3 Hit Rate | 0.588 | **0.883** | +50% |
| JSON Strict Parse | 0.996 | **1.000** | — |
| Valid Sample Rate | 1.000 | 1.000 | — |

### 5.2 Qualitative Examples

#### Example 1: Baseline Wrong → SFT Correct

> **Prompt**: 苦行僧以苦为乐，那么他受苦是不是可以说他在奖励自己  
> *(An ascetic takes joy in suffering — so can we say his suffering is self-reward?)*

| | Baseline | SFT (r16_e5) |
|---|---------|-------------|
| **Top-1** | 弱智科学家 ❌ | 奇怪提问 ✅ |
| **Gold** | 奇怪提问 | 奇怪提问 |
| **Format** | String list (no confidence scores) | Dict list with confidence scores |

The baseline misidentified this as "pseudo-scientist" reasoning, while the SFT model correctly recognized it as a "strange question" that exploits a circular logic paradox.

#### Example 2: Baseline Wrong → SFT Correct

> **Prompt**: 为什么医院床位那么紧张，不弄成上下铺呢？🤗  
> *(Why are hospital beds so scarce? Why not use bunk beds?)*

| | Baseline | SFT (r16_e5) |
|---|---------|-------------|
| **Top-1** | 地狱笑话 ❌ | 奇怪提问 ✅ |
| **Gold** | 奇怪提问 | 奇怪提问 |

The baseline incorrectly classified this as "dark humor" (likely due to the medical context), while the SFT model correctly identified the core mechanism: a naive but seemingly logical suggestion that ignores practical reality.

#### Example 3: Format Improvement (Both Correct)

> **Prompt**: 为什么白天能看见月亮，晚上却看不见太阳呢？  
> *(Why can we see the moon during the day but not the sun at night?)*

| | Baseline | SFT (r16_e5) |
|---|---------|-------------|
| **Top-1** | 奇怪提问 ✅ | 奇怪提问 ✅ |
| **Format** | `top3_categories: ["奇怪提问", "弱智科学家", "古典弱智"]` | `top3_categories: [{rank: 1, category: "奇怪提问", confidence_score: 0.75, reason: "..."}]` |

Both models classify correctly, but the SFT model produces the target structured format with per-category confidence scores and reasons, while the baseline outputs a plain string list without confidence or justification.

#### Example 4: Both Correct, Deeper Analysis

> **Prompt**: 失踪是不是丢人的事情？  
> *(Is going missing a matter of "losing face"?)*

| | Baseline | SFT (r16_e5) |
|---|---------|-------------|
| **Top-1** | 文字游戏 ✅ | 文字游戏 ✅ |

Both correctly identify the wordplay: "失踪" (missing/disappeared) literally involves "丢" (losing) + "人" (person), creating a double meaning with "丢人" (embarrassing/losing face). The SFT model additionally provides calibrated confidence scores (0.75 for wordplay) and explicit reasoning for each candidate category.

#### Example 5: SFT Failure Case (Honest Analysis)

> **Prompt**: 并非无人爱我，病危通知单是死神小姐予我的情书  
> *(It's not that no one loves me — the critical condition notice is a love letter from Lady Death.)*

| | Baseline | SFT (r16_e5) |
|---|---------|-------------|
| **Top-1** | 文字游戏 ❌ | 文字游戏 ❌ |
| **Gold** | 文艺弱智 | 文艺弱智 |

Both models fail to identify the "literary absurdity" (文艺弱智) category, instead classifying it as "wordplay." This is understandable: the sample combines poetic imagery with morbid romanticism, which shares surface features with wordplay. The confusion between 文艺弱智 and 文字游戏 is a known systematic error pattern visible in the confusion matrices.

---

## 6. Extended Analysis

### 6.1 Full 21-Model Comparison

All 20 SFT models + 1 baseline were evaluated on the 240-sample CQIA test set using greedy decoding (`temperature=0.0`). The complete comparison table is available in `results/json/eval_comparison.json`.

**Top 5 Models by Strict Accuracy**:

| Rank | Model Tag | Dataset | LoRA Rank | Epoch | Strict Acc | Top-3 Hit | Eval Loss |
|------|-----------|---------|-----------|-------|-----------|-----------|-----------|
| 1 | **r16_e5** | all | 16 | 5 | **0.613** | 0.883 | **0.8858** |
| 2 | r16_e7 | all | 16 | 7 | 0.604 | **0.892** | 0.8914 |
| 3 | r8_e7 | all | 8 | 7 | 0.579 | 0.871 | 0.8870 |
| 4 | r8_e6 | all | 8 | 6 | 0.575 | 0.854 | 0.8870 |
| 5 | r16_e6 | all | 16 | 6 | 0.575 | 0.883 | 0.8914 |

### 6.2 Key Findings

1. **R16 > R8**: R16 outperforms R8 in 9/10 paired comparisons (avg +4.5% strict accuracy). The advantage is more pronounced on smaller datasets (last3: +6.3% vs all: +2.7%).

2. **Full data > Subset**: Models trained on all 2,785 samples consistently outperform those trained on the 1,025 last-3-year subset (avg +9.1% strict accuracy), demonstrating that data volume outweighs recency for this task.

3. **Eval loss as checkpoint selection signal**: The training-time eval loss minimum (r16_e5, 0.8858) precisely corresponds to the best downstream accuracy (0.613), validating eval loss as a reliable early stopping criterion.

4. **Non-monotonic accuracy vs. epoch**: Accuracy does not increase monotonically with training epochs. Both R8 and R16 show a dip at epoch 4 before recovering, suggesting complex interactions between learning dynamics and evaluation.

### 6.3 Visualizations

All visualizations are organized under `results/`:

| Category | Files | Description |
|----------|-------|-------------|
| Confusion Matrices | `confusion_matrices/` (9 files) | Baseline + top-3 models, counts and normalized |
| Heatmaps | `heatmaps/` (14 files) | Rank × Epoch grids for 7 metrics × 2 datasets |
| Trend Charts | `charts/` (8 files) | Line plots, bar charts, radar chart |

**Key heatmaps** (in `results/heatmaps/`):
- `heatmap_all_strict_accuracy.png`: Shows R16 row consistently darker (higher) than R8, with E5 column brightest for R16.
- `heatmap_all_eval_loss.png`: Mirror image of accuracy heatmap — lowest loss at R16-E5.

---

## 7. Conclusion

LoRA SFT on Qwen3-4B-Instruct-2507 significantly improves Chinese humor classification:
- **Top-1 accuracy**: 0.233 → 0.613 (+163%)
- **Top-3 hit rate**: 0.588 → 0.883 (+50%)
- **JSON format compliance**: near-perfect across all models (VSR = 100%)

The optimal configuration is **R16 + full dataset + epoch 5**, which achieves the best accuracy while maintaining perfect structured output adherence. Training eval loss serves as a reliable proxy for downstream performance, and larger LoRA rank provides consistent benefits at the cost of slightly earlier overfitting onset.

---

## Appendix A: Category Distribution in Test Set

| Category | Count | Percentage |
|----------|-------|------------|
| 奇怪提问 (Strange Questions) | 89 | 37.1% |
| 文字游戏 (Wordplay) | 48 | 20.0% |
| 古典弱智 (Classic Absurdity) | 31 | 12.9% |
| 弱智科学家 (Pseudo-Scientist) | 27 | 11.3% |
| 人生态度 (Life Philosophy) | 22 | 9.2% |
| 文艺弱智 (Literary Absurdity) | 11 | 4.6% |
| 谐音梗 (Homophone Puns) | 8 | 3.3% |
| 地狱笑话 (Dark Humor) | 4 | 1.7% |

## Appendix B: File Index

| File | Description |
|------|-------------|
| `scripts/data/build_sft_data.py` | ShareGPT format conversion |
| `scripts/data/dedup_test_vs_train.py` | Test-train deduplication |
| `scripts/inference/inference_eval.py` | vLLM batch inference |
| `scripts/viz/eval_metrics.py` | Two-stage evaluation + visualization |
| `scripts/viz/gen_before_after.py` | Before/after sample generation |
| `configs/qwen3_4b_base.yaml` | Training config (full data) |
| `configs/qwen3_4b_base_last3.yaml` | Training config (last3 data) |
| `configs/qwen3_4b_merge.yaml` | LoRA merge config |
| `configs/prompts.yaml` | Centralized system prompt |
| `results/json/eval_comparison.json` | 21-model comparison table |
| `results/before_after_samples.json` | 5 before/after examples |
