# 文档目录说明

> 本目录包含项目的作业要求、开发计划、操作指南、训练执行手册、数据分析报告、实验报告和变更日志。

---

## 文件清单

| 文件 | 内容 | 说明 |
|------|------|------|
| `assignment.md` | 作业要求 | CSS5120 Lab3 SFT 微调实验的原始要求 |
| `Lab3_SFT.pdf` | 实验指导书 | 实验指导书 PDF 版本 |
| `dev_plan_0_1.md` | 开发计划 v0.1 | 四阶段流水线规划（数据工程 → SFT 训练 → 评估 → 报告） |
| `dev_plan_0_2.md` | 开发计划 v0.2 | 评估方案设计更新 |
| `dev_plan_0_3.md` | 开发计划 v0.3 | vLLM 迁移后更新 |
| `dev_plan_0_4.md` | 开发计划 v0.5 | 最终状态：Phase 1-4.2 完成 |
| `howto.md` | 操作指南 | 常用操作步骤记录 |
| `changelog.md` | 变更日志 | 各 Phase 的详细执行记录与结果 |
| `training_execution.md` | 训练执行手册 | Phase 2.5 双卡并行训练的 tmux 执行步骤 |
| `train_test_eda.md` | Token 长度 EDA 报告 | 训练集/测试集的 token 长度分布分析 |
| `train_analysis1.md` | 第一批训练日志分析 | MVP + 正式训练的 Loss 曲线详细分析 |
| `test_analysis1.md` | 评估分析报告 | Phase 3 定量评估结果、可视化产物清单 |

---

## `report/` 子目录

实验报告及嵌入图片：

| 文件 | 说明 |
|------|------|
| `lab3_report.md` | 英文实验报告（嵌入 8 张 Figure） |
| `media/fig1_train_eval_loss.png` | Figure 1: 训练+评估 loss 曲线 |
| `media/fig2_eval_loss_trend.png` | Figure 2: Eval loss 趋势 |
| `media/fig3_strict_accuracy_trend.png` | Figure 3: Strict accuracy 趋势 |
| `media/fig4_heatmap_strict_accuracy.png` | Figure 4: Rank×Epoch 热力图 |
| `media/fig5_confusion_grid.png` | Figure 5: 混淆矩阵网格 |
| `media/fig6_per_category_recall.png` | Figure 6: 各类别 recall |
| `media/fig7_baseline_vs_top3.png` | Figure 7: Baseline vs top-3 |
| `media/fig8_all_vs_last3.png` | Figure 8: 全量 vs last3 差值 |

---

## `proposal/` 子目录

项目方案设计文档：

| 文件 | 说明 |
|------|------|
| `final_proposal_gemini_update.md` | 最终方案（更新版） |
| `final_proposal_gemini.md` | 最终方案（初版） |
| `deepseek.md` / `gemini.md` / `qwen.md` | 各 LLM 生成的方案草稿 |
| `my_ideas_qa.md` | 个人构思与问答记录 |

---

## 项目进度概览

基于 `changelog.md` 记录的各阶段完成状态：

| 阶段 | 状态 | 关键产出 |
|------|------|----------|
| Phase 1.1 — CQIA 数据补全 | ✅ 完成 | `ruozhiba_cqia_classified_v2.json` (240 条, 含 thought_process) |
| Phase 1.2 — 去重防污染 | ✅ 完成 | 9 个 `*_classified_dedup.json`，移除 27 条重复 |
| Phase 1.3 — 数据统计验证 | ✅ 完成 | 2786 条训练集 + 240 条测试集，零泄露 |
| Phase 2.1 — ShareGPT 格式化 | ✅ 完成 | `ruozhiba_all.json` (2785 条) + `ruozhiba_last3.json` (1025 条) |
| Phase 2.3 — MVP 训练 | ✅ 完成 | LoRA adapter (64MB)，eval_loss = 0.8820 |
| Phase 2.4 — 显存压测 | ✅ 完成 | 最大安全 BS = 32 (L20Z 80GB) |
| Phase 2.5 — 正式训练 (all) | ✅ 完成 | 双卡并行 R8 + R16，7 epochs |
| Phase 2.7 — 权重合并 | ✅ 完成 | `models/merged/` (20 个模型, 各 7.6 GB) |
| Phase 2.8 — 正式训练 (last3) | ✅ 完成 | R8_last3 + R16_last3，7 epochs |
| Phase 3.1 — vLLM 批量推理 | ✅ 完成 | 21 个模型推理结果 |
| Phase 3.2 — 定量评估 + 可视化 | ✅ 完成 | 混淆矩阵 (9) + 热力图 (14) + 图表 (8) |
| Phase 3.4 — Before/After 对比 | ✅ 完成 | 5 条代表性样本 |
| Phase 4.1 — 报告 v0.1 | ✅ 完成 | `doc/lab3_report_v0_1.md` |
| Phase 4.2 — 报告图片嵌入 | ✅ 完成 | `doc/report/lab3_report.md` + 8 张 Figure |