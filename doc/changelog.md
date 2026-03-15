# Changelog

## 2025-03-15 — Phase 2.1 ShareGPT 格式化

### 新增
- `configs/prompts.yaml` — 中心化 system prompt 与分类类别配置，供训练/推理/评估统一引用
- `scripts/build_sft_data.py` — 将去重后贴吧数据转换为 LLaMA-Factory ShareGPT 格式

### 修改
- `LLaMA-Factory/data/dataset_info.json` — 注册 `ruozhiba_last3` 和 `ruozhiba_all` 两个数据集

### 产出
- `LLaMA-Factory/data/ruozhiba_all.json` — 全量训练集 (2018-2025)，2785 条对话
- `LLaMA-Factory/data/ruozhiba_last3.json` — 近三年训练集 (2023-2025)，1025 条对话

### 结果摘要
- 输入: 9 个 `best*_classified_dedup.json` (Phase 1.2 产出，2786 条)
- 输出: 2785 条 ShareGPT 对话（跳过 1 条缺失 `thought_process` 的条目: 2021_2H #116）
- 格式验证: 全部 2785 条 `gpt.value` 均为合法 JSON，中文无 `\uXXXX` 转义
- 结构验证: 每条对话包含 system/human/gpt 三轮，角色标签正确
- 近三年: 1025 条 | 全量: 2785 条

---

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
