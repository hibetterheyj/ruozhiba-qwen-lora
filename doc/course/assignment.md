# Lab3 SFT

CSS5120
March 2026

## 1 Overview

In this lab, you will perform a Supervised Fine-Tuning (SFT) run on a language model of your choice. Your goal is to define a clear SFT target, prepare a modified dataset, train the model, and evaluate before/after behavior.

## 2 Task Requirements

### 2.1 Choose a Base Model

You may choose any language model that you can run in your environment (e.g., Hugging Face models). Examples include small open models suitable for Colab. You must clearly state:
• Model name/checkpoint
• Why it is appropriate

### 2.2 Define an SFT Target

Your SFT target can be any reasonable objective. Below are several typical SFT targets. You may use one of them or propose your own.

- **Behavior shaping (style/persona constraints)**
  Example targets:
  • Always answer in a concise, polite tone (e.g., limit to 3 sentences).
  • Always provide an answer in the format {Answer: X} for multiple-choice questions.
- **Instruction following (command compliance)**
  Example targets:
  • Follow rules like “Output must be valid JSON with fields title, steps.”
  • Refuse to output extra text beyond the requested format (no explanations).
- **Structured output (format control)**
  Example targets:
  • Extract key fields from text into a schema (e.g., name, date, location).
  • Convert informal text into a standardized template (e.g., meeting notes with fixed headings).
- **Domain adaptation (domain-specific examples)**
  Example targets:
  • Answer questions using medical/finance/legal domain wording or conventions.
  • Summarize technical passages using the terminology of a specific domain.
- **Task specialization (classification/QA/rewriting)**
  Example targets:
  • Sentiment classification with a fixed label set.
  • Short QA where answers must cite one of the provided options (A/B/C/D/E).

### 2.3 Dataset Requirement

You may start from an existing dataset, but you cannot use it exactly as-is. You must make a non-trivial modification (small or large) such that the training data better matches your chosen SFT target.

- **Acceptable modifications (examples):**
  • Rewrite the outputs to enforce style/format constraints (e.g., {Answer: X} only).
  • Augment examples (e.g., add instruction prefixes, add negative/contrast cases).
  • Convert non-QA data into input–output pairs (e.g., transform plain text corpora into explicit prompt–completion format, such as converting summaries, classification labels, or structured records into clear instruction–response pairs).
- **Not acceptable:**
  • Using the dataset exactly as provided with no meaningful changes.
  • Only shuffling the dataset or changing train/val split without modifying content/format.

## 3 Report Requirements

Your report should be clear and include the following:

### 3.1 SFT Target Description

• What behavior/task you want the model to learn
• Why this target matters (brief motivation)
• A few example prompts and the desired outputs (2–4 examples)

### 3.2 Dataset Source and Preprocessing

• Dataset source (link or citation)
• How you modified it (what changed and why)
• Final data format used for training (e.g., prompt/completion, text, or messages)
• Train/validation split sizes

### 3.3 Training Setup

• Base model and key hyperparameters
• Whether you used parameter-efficient tuning (e.g., LoRA) and key LoRA settings (if applicable)
• Any hardware/environment notes (e.g., Colab GPU, mixed precision)

### 3.4 Loss Curves / Training Signals

Include at least one figure or table showing:
• Training loss over steps
• Validation loss over steps (if available)
Briefly comment on what you observe (e.g., stable decrease, overfitting signs, noisy training).

### 3.5 Before vs. After Comparison

Show qualitative comparison using a small test set of prompts not used in training.
• Model output before SFT vs after SFT
• Highlight whether the model matches the target format/behavior better

## 4 Submission

Submit the following:

### Code

• A runnable notebook or Python scripts including:
  – data preprocessing
  – training
  – evaluation (before/after comparison)
• Include a README file with clear instructions to reproduce the results.

### Report (PDF)

• A PDF report following the report requirements described above.
