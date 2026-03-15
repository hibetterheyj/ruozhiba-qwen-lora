# Changelog

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
