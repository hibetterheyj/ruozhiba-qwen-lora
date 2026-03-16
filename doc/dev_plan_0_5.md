# 弱智吧幽默分类 SFT 微调 — 开发计划 v0.5

> 基于 `dev_plan_0_3.md` 更新。Phase 1-3.2 全部完成，Phase 3.4 Before/After 已完成。
>
> **v0.3 → v0.5 变更**:
> 1. Phase 3.3 LLM-as-Judge **暂时跳过** — 不影响 assignment 核心要求
> 2. Phase 3.4 Before/After 对比 **已完成** — 5 条代表性样本
> 3. Phase 4 报告 v0.1 **已完成** — 英文实验报告初稿 `doc/lab3_report_v0_1.md`

---

## 已完成工作总览

| Phase | 状态 | 产出 |
|-------|------|------|
| 1.1 CQIA thought_process 导师蒸馏 | ✅ 完成 | `data/CQIA/ruozhiba_cqia_classified_v2.json` (240 条) |
| 1.2 去重防污染 | ✅ 完成 | `data/tieba/best*_classified_dedup.json` (2786 条) |
| 1.3 数据统计验证 | ✅ 完成 | `data/dedup_report.json` |
| 2.1 ShareGPT 格式化 | ✅ 完成 | `LLaMA-Factory/data/ruozhiba_all.json` (2785 条) |
| 2.2 数据集注册 | ✅ 完成 | `dataset_info.json` 已注册 |
| 2.3 MVP 训练 | ✅ 完成 | `saves/qwen3-4b/lora/mvp_r8_e3/` |
| 2.4 BS 压测 | ✅ 完成 | BS=16×2 (有效 32) |
| 2.5 双卡并行训练 (all) | ✅ 完成 | R8 7 epochs + R16 7 epochs |
| 2.6 训练监控 | ✅ 完成 | 最优 eval_loss: R16 checkpoint-415 (0.8859) |
| 2.7 全量 LoRA 权重合并 | ✅ 完成 | `models/merged/` (20 个模型) |
| 2.8 双卡并行训练 (last3) | ✅ 完成 | R8_last3 7 epochs + R16_last3 7 epochs |
| 3.1 vLLM 批量推理 | ✅ 完成 | `results/results_*.json` (21 个) |
| 3.2 定量评估 + 可视化 | ✅ 完成 | 混淆矩阵 (9) + 热力图 (14) + 图表 (8) |
| 3.3 LLM-as-Judge | ⏭️ 跳过 | — |
| 3.4 Before/After 对比 | ✅ 完成 | `results/before_after_samples.json` (5 条) |
| 4.1 英文报告 v0.1 | ✅ 完成 | `doc/lab3_report_v0_1.md` |

---

## Phase 3.3: LLM-as-Judge — 暂时跳过

**原因**: assignment.md 不要求 LLM-as-Judge。核心要求 (3.1 SFT Target, 3.2 Dataset, 3.3 Training Setup, 3.4 Loss Curves, 3.5 Before/After) 已全部覆盖。

**未来可选**: 若需要进一步提升报告质量，可对 top 2-3 模型 (r16_e5, r16_e7, r8_e7) 执行 DeepSeek-Chat 双盲评估。设计已在 `dev_plan_0_2.md` Phase 3.3 中完备。

---

## Phase 3.4: Before/After 对比 — 已完成

### 产出

- **脚本**: `scripts/gen_before_after.py` — 从 baseline 和 r16_e5 推理结果中自动选取 5 条代表性样本
- **数据**: `results/before_after_samples.json`

### 样本选取策略

| 编号 | 类型 | 样本 | 结果 |
|------|------|------|------|
| 1 | baseline 错误 → SFT 正确 | 苦行僧以苦为乐... | 弱智科学家→奇怪提问 ✅ |
| 2 | baseline 错误 → SFT 正确 | 医院床位上下铺... | 地狱笑话→奇怪提问 ✅ |
| 3 | 格式改进 (均正确) | 白天看月亮... | string list → dict list with confidence |
| 4 | 均正确, SFT 分析更深 | 失踪是不是丢人... | 文字游戏 ✅, SFT 附带 confidence + reason |
| 5 | SFT 仍然失败 | 病危通知单情书... | 文艺弱智 误判为 文字游戏 ❌ |

---

## Phase 4: 报告 — v0.1 已完成

### 4.1 报告文件

`doc/lab3_report_v0_1.md` — 英文实验报告，覆盖 assignment.md 全部 5 个 section:

| Assignment Section | Report Section | 状态 |
|-------------------|---------------|------|
| 3.1 SFT Target Description | §1 | ✅ |
| 3.2 Dataset Source and Preprocessing | §2 | ✅ |
| 3.3 Training Setup | §3 | ✅ |
| 3.4 Loss Curves / Training Signals | §4 | ✅ |
| 3.5 Before vs. After Comparison | §5 | ✅ |

### 4.2 待办 (报告完善)

| 任务 | 优先级 | 说明 |
|------|--------|------|
| 嵌入训练 loss 曲线图到报告 | P1 | 当前报告引用了图片路径，PDF 版需要嵌入实际图片 |
| 嵌入热力图/混淆矩阵到报告 | P1 | 同上 |
| 导出 PDF | P1 | Markdown → PDF (pandoc 或其他工具) |
| README 复现指南 | P2 | 更新 `readme.md` 包含完整复现步骤 |
| 图片尺寸/DPI 优化 | P3 | 确保 PDF 中图片清晰可读 |

---

## 交付物清单

```
提交内容:
├── doc/lab3_report_v0_1.md             # ✅ 英文实验报告 v0.1
├── scripts/
│   ├── classify_cqia_updated.py        # ✅ Phase 1.1 — CQIA thought_process 补全
│   ├── dedup_test_vs_train.py          # ✅ Phase 1.2 — 去重脚本
│   ├── build_sft_data.py              # ✅ Phase 2.1 — ShareGPT 格式化
│   ├── probe_batch_size.sh            # ✅ Phase 2.4 — 显存 BS 压测
│   ├── run_training.sh                # ✅ Phase 2.5 — 训练启动
│   ├── batch_merge.sh                 # ✅ Phase 2.7 — 批量权重合并
│   ├── inference_eval.py              # ✅ Phase 3.1 — vLLM 批量推理
│   ├── batch_inference.sh             # ✅ Phase 3.1 — 批量推理封装
│   ├── eval_metrics.py                # ✅ Phase 3.2 — 两阶段评估 + 可视化
│   └── gen_before_after.py            # ✅ Phase 3.4 — Before/After 样本生成
├── configs/
│   ├── prompts.yaml                   # ✅ 中心化 system prompt
│   ├── qwen3_4b_mvp.yaml             # ✅ MVP 配置
│   ├── qwen3_4b_base.yaml            # ✅ 正式训练配置 (全量)
│   ├── qwen3_4b_base_last3.yaml      # ✅ 近三年训练配置
│   └── qwen3_4b_merge.yaml           # ✅ LoRA 合并配置模板
├── results/
│   ├── results_*.json                 # ✅ 21 个推理结果 (240 条/文件)
│   ├── before_after_samples.json      # ✅ 5 条 Before/After 对比
│   ├── json/eval_comparison.json      # ✅ 21 模型对比总表
│   ├── json/eval_*.json               # ✅ 各模型详细评估
│   ├── confusion_matrices/            # ✅ 混淆矩阵 (9 张)
│   ├── heatmaps/                      # ✅ Rank×Epoch 热力图 (14 张)
│   └── charts/                        # ✅ 趋势/对比图表 (8 张)
└── readme.md                          # 🔲 待更新 — 复现指南
```
