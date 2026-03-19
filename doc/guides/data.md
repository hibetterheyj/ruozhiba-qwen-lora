# 数据来源与数据说明

## 数据来源总览

| 来源 | 说明 | 规模（约） |
|------|------|------------|
| 百度贴吧弱智吧 | 年度佳帖等，爬虫 + 后处理 | ~3200 条原始 |
| [COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | 弱智吧子集，作**测试集** | 240 条 |
| GitHub 弱智吧语料 | 2018–2021 年度精选 | 1361 条 |

去重、分类与 ShareGPT 转换后，训练集约 **2785** 条（`ruozhiba_all`），近三年子集约 **1025** 条（`ruozhiba_last3`）。详见仓库根目录 [`data/readme.md`](../data/readme.md)。

## 数据统计（贴吧年度等）

| 数据源 | 文件 | 条数 |
|--------|------|------|
| 贴吧 2025 | `data/tieba/best365_2025.json` | 361 |
| 贴吧 2024 | `best365_2024.json` | 352 |
| 贴吧 2023 | `best365_2023.json` | 322 |
| 贴吧 2022 | `best365_2022.json` | 330 |
| 贴吧 2021 下半年 | `best306_2021_2H.json` | 306 |
| 贴吧 2021 上半年 | `best295_2021_1H.json` | 286 |
| 贴吧 2020 | `best365_2020.json` | 353 |
| 贴吧 2019 | `best336_2019.json` | 333 |
| 贴吧 2018 | `best176_2018.json` | 170 |
| 经典 100 条 | `best100_all.json` | 8 类 |
| CQIA | `data/CQIA/ruozhiba_ruozhiba.jsonl` | 240 |
| GitHub 年度 | `data/ruozhiba/data/ruozhiba-post-annual.json` | 1361 |

> 实际条数可能少于文件名中的数字（帖子删除等）。

## 数据格式（摘要）

### 贴吧原始 JSON (`data/tieba/best*_YYYY.json`)

```json
[
  { "no": 1, "text": "段子正文" }
]
```

可选字段：`score`。命名示例：`best365_2020.json`、`best306_2021_2H.json`。

### 分类后 (`best*_classified.json`)

在原文段上增加 `classification`：

- `thought_process`：分析过程  
- `top3_categories`：`rank`, `category`, `confidence_score`, `reason`（8 类 humor 之一）

### CQIA

- 原始：`ruozhiba_ruozhiba.jsonl`（instruction / output 等）  
- 清洗：`ruozhiba_cqia_cleaned.json`  
- 分类金标 / 测试：`ruozhiba_cqia_classified_v2.json`（路径以 `data/CQIA/` 为准）

### GitHub 语料 (`data/ruozhiba/data/`)

`ruozhiba-post-annual.json`：`author`, `content`, `l_num`, `ctime` 等。

### 经典 100 条 (`best100_all.json`)

按类别分组的 `category` / `description` / `jokes[]`。

### 爬虫目录结构 (`crawler/threads/`)

```
{帖子ID}_{标题}/
├── *_dump.jsonl
├── extracted.json
├── checkpoint.json
├── images/
└── posts/page_*.json
```

### SFT 训练格式

ShareGPT 多轮：`system` / `human` / `gpt`，`gpt` 侧为 JSON 字符串（`thought_process` + `top3_categories`）。成品见 `data/LLaMA-Factory/data/ruozhiba_all.json` 等。

## 处理流水线

详见 [`reproduction.md`](reproduction.md) 中的 ASCII 流程图（爬取 → 去重 → 分类 → 校验 → `build_sft_data.py` → 训练 → 合并 → 推理 → 评估）。
