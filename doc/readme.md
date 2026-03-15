# 文档目录说明

> 本目录包含项目的作业要求、开发计划、操作指南、训练执行手册、数据分析报告和变更日志。

---

## 文件清单

| 文件 | 内容 | 说明 |
|------|------|------|
| `assignment.md` | 作业要求 | CSS5120 Lab3 SFT 微调实验的原始要求 |
| `Lab3_SFT.pdf` | 实验指导书 | 实验指导书 PDF 版本 |
| `dev_plan_0_1.md` | 开发计划 v0.1 | 四阶段流水线规划（数据工程 → SFT 训练 → 评估 → 报告） |
| `howto.md` | 操作指南 | 常用操作步骤记录 |
| `changelog.md` | 变更日志 | 各 Phase 的详细执行记录与结果 |
| `training_execution.md` | 训练执行手册 | Phase 2.5 双卡并行训练的 tmux 执行步骤 |
| `train_test_eda.md` | Token 长度 EDA 报告 | 训练集/测试集的 token 长度分布分析 |
| `train_analysis1.md` | 第一批训练日志分析 | MVP + 正式训练的 Loss 曲线详细分析 |

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
| Phase 2.5 — 正式训练配置 | ✅ 完成 | 双卡并行 R8 + R16，BS=16×2 梯度累积 |
| Phase 2.6 — 训练监控 | ✅ 完成 | 最优 eval_loss = 0.8859 (R16 checkpoint-415) |
| Phase 2.7 — 权重合并 | ✅ 完成 | `Qwen3-4B-Ruozhiba-Merged/` (7.6 GB) |
| Token 长度 EDA | ✅ 完成 | 100% 样本 ≤ 1024 tokens，但是保留2048 |
| Phase 3 — 评估 | 🔲 未开始 | — |
| Phase 4 — 报告 | 🔲 未开始 | — |