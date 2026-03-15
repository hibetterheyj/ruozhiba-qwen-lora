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

---

## 2025-03-15 — Phase 1.1: CQIA 数据补全 (thought_process 导师蒸馏)

### 概述

为 240 条 CQIA 测试集数据补充 `thought_process` 字段，使其与贴吧训练集的 classification 格式对齐。使用 Claude-Opus-4-6 作为导师模型，对每条 `instruction` 生成深度幽默解构分析。

### 新增文件

| 文件 | 说明 |
|------|------|
| `scripts/classify_cqia_updated.py` | CQIA 数据补全脚本，复用 `classify_jokes.py` 鲁棒性模式 |
| `scripts/classify_cqia_updated_config.yaml` | 配置文件，System Prompt 对齐 `classify_config.yaml` |
| `data/CQIA/ruozhiba_cqia_classified_v2.json` | 输出数据（240 条，含 thought_process） |

### 技术要点

- **System Prompt 对齐**: 使用与贴吧分类相同的 prompt（含 `thought_process` + `top3_categories`），确保训练/测试数据格式一致
- **仅输入 instruction**: 不将 CQIA 的 `output`（正经 AI 解答）作为 LLM 输入，避免干扰分类判断
- **原始字段保留**: `output` 和原有 `top3_categories` 完全不变，仅在 `classification` 中新增 `thought_process`
- **Category Drift 日志**: 记录新旧 Top-1 分类差异（仅日志，不覆盖），发现约 20+ 条存在 drift，属正常现象（prompt 变更导致）
- **断点续传**: JSONL checkpoint 支持中断恢复，首次运行 228/240 成功 → 重试后 239/240 → Item #53（盲文）因 safety filter 手动补写

### 数据质量报告

```
Total items:          240
Schema valid:         True (instruction/output/classification)
thought_process:      240/240 present (0 null, 0 empty)
top3_categories:      240/240 exactly 3 categories
thought_process avg:  ~276 chars (min: 176, max: 456)
Data integrity vs v1: 0 mismatches (instruction/output/top3_categories 完全一致)

Top-1 category distribution:
  奇怪提问: 89    文字游戏: 48    文艺弱智: 34    弱智科学家: 27
  古典弱智: 22    谐音梗: 10      人生态度: 6     地狱笑话: 4
```

### 备注

- Item #53（盲文 Unicode 内容）触发 API safety filter，`thought_process` 为手动补写，内容与已有 `top3_categories` 一致
- 已清理 JSONL checkpoint 中间文件
