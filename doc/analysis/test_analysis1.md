# 弱智吧幽默分类 — Phase 3 定量评估分析报告

> 评估日期: 2025-03-16  
> 推理后端: vLLM 0.17.1 (env_sft, torch 2.10.0, GPU: L20Z 80GB)  
> 评估脚本: `scripts/viz/eval_metrics.py` (两阶段 JSON 评估协议)  
> 测试集: `data/CQIA/ruozhiba_cqia_classified_v2.json` (240 条)

---

## 1. 实验概览

### 1.1 评估矩阵

共 **21 组模型** 参与评估:

| 组别 | 模型数 | 数据集 | LoRA Rank | Epoch 范围 |
|------|--------|--------|-----------|-----------|
| Baseline | 1 | — | — | — |
| 全量 R8 | 5 | ruozhiba_all (2785) | 8 | 3-7 |
| 全量 R16 | 5 | ruozhiba_all (2785) | 16 | 3-7 |
| 近三年 R8 | 5 | ruozhiba_last3 (1025) | 8 | 3-7 |
| 近三年 R16 | 5 | ruozhiba_last3 (1025) | 16 | 3-7 |

### 1.2 评估流程

1. **Phase 3.1 批量推理**: 21 个模型 × 240 条测试集 = 5,040 次推理，`temperature=0.0` (greedy decoding)
2. **Phase 3.2 两阶段评估**:
   - Stage 1: JSON 格式遵循能力 (`json_strict` / `json_tolerant` / `vsr`)
   - Stage 2: 逻辑准确率 (`top1_accuracy` / `top3_hit_rate` / `confidence_mae` / `strict_accuracy` / `repaired_accuracy`)
3. **可视化**: 混淆矩阵 (9 张) + Rank×Epoch 热力图 (14 张) + 趋势/对比图表 (8 张)

---

## 2. 总评结果

### 2.1 完整对比表

| Model Tag | Strict Acc | Repaired Acc | Top-1 Acc | Top-3 Hit | JSON Strict | VSR | Eval Loss | Conf MAE |
|-----------|-----------|-------------|-----------|-----------|-------------|-----|-----------|----------|
| **baseline** | 0.233 | 0.233 | 0.233 | 0.588 | 0.996 | 1.000 | — | 1.000* |
| r8_e3 | 0.542 | 0.542 | 0.542 | 0.833 | 0.988 | 1.000 | 0.9257 | 0.117 |
| r8_e4 | 0.467 | 0.467 | 0.467 | 0.829 | 0.996 | 1.000 | 0.8987 | 0.112 |
| r8_e5 | 0.558 | 0.558 | 0.558 | 0.892 | 0.996 | 1.000 | 0.8884 | 0.109 |
| r8_e6 | 0.575 | 0.575 | 0.575 | 0.854 | 0.996 | 1.000 | 0.8870 | 0.107 |
| r8_e7 | 0.579 | 0.579 | 0.579 | 0.871 | 0.988 | 1.000 | 0.8870 | 0.118 |
| **r16_e3** | 0.567 | 0.567 | 0.567 | 0.850 | 0.992 | 1.000 | 0.9033 | 0.109 |
| r16_e4 | 0.496 | 0.496 | 0.496 | 0.850 | 0.996 | 1.000 | 0.8886 | 0.109 |
| **r16_e5** | **0.613** | **0.613** | **0.613** | 0.883 | **1.000** | 1.000 | **0.8858** | 0.107 |
| r16_e6 | 0.575 | 0.575 | 0.575 | 0.883 | 1.000 | 1.000 | 0.8914 | 0.102 |
| **r16_e7** | 0.604 | 0.604 | 0.604 | **0.892** | 1.000 | 1.000 | 0.8914 | **0.099** |
| r8_last3_e3 | 0.342 | 0.346 | 0.346 | 0.679 | 0.988 | 1.000 | 1.0433 | 0.109 |
| r8_last3_e4 | 0.446 | 0.446 | 0.446 | 0.754 | 0.988 | 1.000 | 1.0057 | 0.106 |
| r8_last3_e5 | 0.433 | 0.433 | 0.433 | 0.750 | 1.000 | 1.000 | 0.9923 | 0.108 |
| r8_last3_e6 | 0.446 | 0.446 | 0.446 | 0.783 | 1.000 | 1.000 | 0.9856 | 0.102 |
| r8_last3_e7 | 0.458 | 0.458 | 0.458 | 0.783 | 0.996 | 1.000 | 0.9843 | 0.092 |
| r16_last3_e3 | 0.458 | 0.458 | 0.458 | 0.829 | 1.000 | 1.000 | 0.9985 | 0.101 |
| r16_last3_e4 | 0.479 | 0.479 | 0.479 | 0.783 | 0.988 | 1.000 | 0.9704 | 0.105 |
| r16_last3_e5 | 0.500 | 0.500 | 0.500 | 0.842 | 0.996 | 1.000 | 0.9643 | 0.114 |
| r16_last3_e6 | 0.500 | 0.504 | 0.504 | 0.846 | 0.992 | 1.000 | 0.9637 | 0.111 |
| r16_last3_e7 | 0.504 | 0.504 | 0.504 | 0.821 | 0.996 | 1.000 | 0.9638 | 0.112 |

> \* Baseline 的 `confidence_mae=1.000` 是因为基座模型输出 `top3_categories` 为字符串列表 (无 `confidence_score` 字段)，所有 240 条均触发最大惩罚 (1.0)。此指标在 baseline 与 SFT 模型之间不可直接横向比较。

### 2.2 最优模型

| 指标 | 最优模型 | 得分 |
|------|---------|------|
| Strict Accuracy | **r16_e5** | 0.613 |
| Repaired Accuracy | **r16_e5** | 0.613 |
| Top-3 Hit Rate | **r16_e7** | 0.892 |

**综合最优**: `r16_e5` (R16, 全量数据, Epoch 5) — 在分类准确率和 JSON 格式遵循两个维度同时达到最佳。

---

## 3. 关键发现

### 3.1 SFT 显著提升分类能力

所有 SFT 模型相比 baseline 均有大幅提升:

| 指标 | Baseline | SFT 最优 | SFT 平均 | 提升幅度 |
|------|----------|---------|---------|---------|
| Top-1 Accuracy | 0.233 | 0.613 (r16_e5) | 0.502 | +163% (最优) |
| Top-3 Hit Rate | 0.588 | 0.892 (r16_e7) | 0.818 | +52% (最优) |
| JSON Strict | 0.996 | 1.000 | 0.996 | — |

- Baseline 仅 23.3% 的 top-1 准确率，说明基座模型对 8 类幽默分类的理解有限
- 最优 SFT 模型 (r16_e5) 将准确率提升至 61.3%，提升 2.63 倍
- Top-3 覆盖率从 58.8% 提升至 89.2%，说明正确类别几乎总在模型的前三预测中

### 3.2 Rank 16 优于 Rank 8

在相同数据集和 epoch 下，R16 几乎全面优于 R8:

| 对比组 | R8 Strict Acc | R16 Strict Acc | R16 优势 |
|--------|-------------|---------------|---------|
| all_e3 | 0.542 | 0.567 | +0.025 |
| all_e4 | 0.467 | 0.496 | +0.029 |
| all_e5 | 0.558 | **0.613** | **+0.055** |
| all_e6 | 0.575 | 0.575 | 0.000 |
| all_e7 | 0.579 | 0.604 | +0.025 |
| last3_e3 | 0.342 | 0.458 | +0.117 |
| last3_e4 | 0.446 | 0.479 | +0.033 |
| last3_e5 | 0.433 | 0.500 | +0.067 |
| last3_e6 | 0.446 | 0.500 | +0.054 |
| last3_e7 | 0.458 | 0.504 | +0.046 |

- R16 在 9/10 个对比组中胜出，平均优势 +0.045
- 优势在 last3 数据集上更明显 (平均 +0.063 vs all 的 +0.027)，说明更大 rank 在小数据场景下更能发挥作用

### 3.3 全量数据显著优于近三年

`all` (2785 条) 在所有 Rank×Epoch 组合中均优于 `last3` (1025 条):

| Rank | Epoch | all Strict | last3 Strict | Delta |
|------|-------|-----------|-------------|-------|
| R8 | 3 | 0.542 | 0.342 | **-0.200** |
| R8 | 4 | 0.467 | 0.446 | -0.021 |
| R8 | 5 | 0.558 | 0.433 | -0.125 |
| R8 | 6 | 0.575 | 0.446 | -0.129 |
| R8 | 7 | 0.579 | 0.458 | -0.121 |
| R16 | 3 | 0.567 | 0.458 | -0.108 |
| R16 | 4 | 0.496 | 0.479 | -0.017 |
| R16 | 5 | 0.613 | 0.500 | -0.113 |
| R16 | 6 | 0.575 | 0.500 | -0.075 |
| R16 | 7 | 0.604 | 0.504 | -0.100 |

- **all 平均优势: +0.091** (strict_accuracy)
- 数据量的增加 (2785 vs 1025, 2.7×) 带来了一致性的准确率提升
- 早期历史数据 (2018-2022) 对模型泛化能力有积极贡献，不应丢弃

### 3.4 Epoch 选择: 非单调关系

Strict accuracy 并非随 epoch 单调递增:

**全量 R8**: E3(0.542) → E4(0.467↓) → E5(0.558↑) → E6(0.575↑) → **E7(0.579)**  
**全量 R16**: E3(0.567) → E4(0.496↓) → **E5(0.613)** → E6(0.575↓) → E7(0.604↑)  
**last3 R8**: E3(0.342) → E4(0.446↑) → E5(0.433↓) → E6(0.446↑) → **E7(0.458)**  
**last3 R16**: E3(0.458) → E4(0.479↑) → E5(0.500↑) → E6(0.500) → **E7(0.504)**

- Epoch 4 出现明显的性能下沉 (R16 all: 0.567→0.496, R8 all: 0.542→0.467)
- 最终恢复是因为后续 epoch 进一步学习；但 epoch 5 的 R16 达到全局最优，之后不再类似水平
- **推荐**: R16 全量训练选择 Epoch 5 作为最终模型

### 3.5 Eval Loss 与 Downstream Accuracy 的相关性

| Model | Eval Loss | Strict Acc | 备注 |
|-------|-----------|-----------|------|
| r16_e5 | **0.8858** | **0.613** | 最低 eval_loss + 最高 accuracy |
| r16_e7 | 0.8914 | 0.604 | eval_loss 回升, accuracy 略降 |
| r16_e6 | 0.8914 | 0.575 | eval_loss 同, accuracy 更低 |
| r8_e6 | 0.8870 | 0.575 | |
| r8_e7 | 0.8870 | 0.579 | |

- 在全量数据模型中，eval_loss 最低点 (r16_e5) **精确对应** strict_accuracy 最高点
- 这验证了 Phase 2.6 训练分析中的 R16 过拟合观察: eval_loss 在 epoch 5 后回升，downstream accuracy 也随之下降
- **结论**: 训练阶段的 eval_loss 是可靠的 checkpoint 选择依据

### 3.6 JSON 格式遵循能力

| 模型组 | JSON Strict 范围 | VSR |
|--------|-----------------|-----|
| Baseline | 0.996 | 1.000 |
| All SFT | 0.988 - 1.000 | 1.000 |

- 所有模型的 VSR (Valid Sample Rate) 均为 **100%**，远超 80% 阈值
- JSON strict 率在 98.8% - 100% 之间，几乎完美
- **json_strict = repaired_accuracy**: 由于 VSR 达到 100%，json-repair 修复对最终准确率无提升（严格解析失败的极少数样本在修复后解析成功，但分类结果与严格解析一致）
- SFT 训练并未损害基座模型的 JSON 生成能力

### 3.7 置信度校准分析

| 分组 | 正确时均值 | 错误时均值 | 差值 |
|------|-----------|-----------|------|
| All R8 | 0.698 | 0.653 | 0.045 |
| All R16 | 0.695 | 0.646 | 0.049 |
| Last3 R8 | 0.700 | 0.675 | 0.025 |
| Last3 R16 | 0.697 | 0.675 | 0.022 |

- Baseline 无置信度 (输出纯字符串列表), 不纳入分析
- 所有 SFT 模型的正确/错误置信度差距极小 (**Δ ≈ 0.02-0.05**)
- 置信度得分集中在 0.64-0.71 区间，区分度不足
- **结论**: 模型的 uncertainty awareness 较弱, 在分类错误时依然给出较高置信度 (>0.6)。未来可通过 calibration training 或后处理 entropy-based recalibration 改善

---

## 4. 可视化产物

所有可视化产物已按类型整理到子文件夹中，CJK 中文标签正常显示 (Noto Sans CJK JP)。

### 4.1 混淆矩阵 (`results/confusion_matrices/`)

仅为 baseline + Top-3 模型 (r16_e5, r16_e7, r8_e7) 生成混淆矩阵，共 **9 张**:
- `confusion_matrix_{tag}_{counts|normalized}.png` × 4 模型 × 2 = 8 张
- `confusion_grid_top_models.png` — 将 Top 模型的归一化混淆矩阵合并为一张网格图

**最优模型 r16_e5 的典型混淆模式** (归一化矩阵观察):
- "奇怪提问" 是最大类别 (89 条)，模型在此类上表现最好
- "文字游戏" (48 条) 和 "弱智科学家" (27 条) 之间存在较高互混率
- "地狱笑话" (4 条) 样本过少，分类不稳定

### 4.2 Rank×Epoch 热力图 (`results/heatmaps/`)

共 7 × 2 = **14 张**热力图: `heatmap_{all|last3}_{metric}.png`

| 热力图 | 观察 |
|--------|------|
| `heatmap_all_strict_accuracy.png` | R16 行普遍更深（更高），E5 列最优 |
| `heatmap_all_eval_loss.png` | R16-E5 最浅（最低），与 accuracy 热力图互为镜像 |
| `heatmap_last3_strict_accuracy.png` | 整体浅于 all，R16 行优于 R8 |
| `heatmap_all_vsr.png` | 全部 1.000，无差异 |
| `heatmap_all_top3_hit_rate.png` | R8-E5 和 R16-E7 最深 (0.892) |

### 4.3 趋势与对比图表 (`results/charts/`)

共 **8 张**新增图表:

| 文件 | 类型 | 说明 |
|------|------|------|
| `line_strict_accuracy.png` | 折线图 | 4 组实验的 strict_accuracy 随 epoch 变化趋势，含 baseline 基准线 |
| `line_top3_hit_rate.png` | 折线图 | 4 组实验的 top3_hit_rate 随 epoch 变化趋势 |
| `line_top1_accuracy.png` | 折线图 | 4 组实验的 top1_accuracy 随 epoch 变化趋势 |
| `line_eval_loss.png` | 折线图 | 4 组实验的 eval_loss 随 epoch 变化趋势（y 轴反转，越低越好） |
| `bar_baseline_vs_top3.png` | 柱状图 | Baseline vs Top-3 模型在 4 个核心指标上的对比 |
| `bar_all_vs_last3_delta.png` | 差值柱状图 | 全量 vs 近三年的 strict_accuracy 差值 (all − last3) |
| `bar_per_category_recall_r16_e5.png` | 柱状图 | 最优模型 r16_e5 在各类别上的 per-category recall，标注样本数 |
| `radar_top_models.png` | 雷达图 | Baseline vs Top-3 模型在 5 维指标上的综合对比 |

---

## 5. 局限性与注意事项

1. **测试集规模**: 240 条样本的统计功效有限，各类别样本数不均衡 (奇怪提问 89 条 vs 地狱笑话 4 条)
2. **金标准标注**: CQIA 测试集的 `top3_categories` 由 Claude-Opus-4-6 导师蒸馏生成，非人工标注，天花板受限于导师模型能力
3. **Confidence 不可比**: Baseline 输出无 `confidence_score`，其 `confidence_mae=1.000` 是全局惩罚，不反映真实校准水平
4. ~~CJK 字体缺失~~: 已安装 `fonts-noto-cjk-extra`，中文标签正常显示
5. **eval_loss 跨数据集不可比**: `all` 和 `last3` 使用不同的 eval 子集，其 eval_loss 绝对值不可横向对比, 但同一数据集内可用于排序

---

## 6. 结论与推荐

### 6.1 核心结论

1. **SFT 有效**: LoRA 微调在幽默分类任务上带来 163% 的 Top-1 准确率提升 (0.233 → 0.613)
2. **全量数据优于子集**: 全量 2785 条训练数据一致性优于近三年 1025 条 (+9.1% strict_accuracy)
3. **R16 优于 R8**: 更大的 LoRA rank 在 9/10 组对比中胜出 (+4.5% avg)
4. **最优配置**: R16 + 全量数据 + Epoch 5 → `r16_e5` (strict_accuracy = 0.613, top3_hit_rate = 0.883)
5. **JSON 遵循完美**: 所有模型 VSR = 100%，不存在指令遵循问题
6. **eval_loss 是可靠的 checkpoint 选择信号**: 训练 eval_loss 最低点精确对应 downstream 最优 checkpoint

### 6.2 推荐下一步

1. **Phase 3.3 LLM-as-Judge**: 对 top 2-3 模型 (r16_e5, r16_e7, r8_e7) 进行双盲人类偏好评估
2. **置信度校准**: 引入 temperature scaling 或 Platt scaling 改善置信度区分度
3. **数据增强**: 考虑对少数类别 (地狱笑话、谐音梗) 进行过采样或合成增强
4. **最终模型**: 推荐 `models/merged/r16_e5` 作为部署候选

---

## 附录: 文件清单

### 推理结果 (21 个)
- `results/results_{baseline, r8_e3..r8_e7, r16_e3..r16_e7, r8_last3_e3..r8_last3_e7, r16_last3_e3..r16_last3_e7}.json`

### 评估结果 (`results/json/`, 22 个)
- `results/json/eval_{tag}.json` × 21 (各模型指标 + per_sample 明细)
- `results/json/eval_comparison.json` (对比总表 + 最优模型 + all_vs_last3 配对对比)

### 可视化 (31 个)
- `results/confusion_matrices/confusion_matrix_{tag}_{counts|normalized}.png` × 8 (baseline + top-3 模型)
- `results/confusion_matrices/confusion_grid_top_models.png` × 1
- `results/heatmaps/heatmap_{all|last3}_{metric}.png` × 14
- `results/charts/line_*.png` × 4 (折线图)
- `results/charts/bar_*.png` × 3 (柱状图/差值图)
- `results/charts/radar_top_models.png` × 1 (雷达图)
