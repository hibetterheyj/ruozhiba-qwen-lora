# 数据目录说明

> 本目录存放项目的所有原始数据、处理后数据及训练数据。

---

## 目录结构

```
data/
├── readme.md                    # 本文件
├── catogory_ideas.md            # 分类体系设计笔记
├── init_ideas.md                # 初期构思
├── dedup_report.json            # 去重报告（精确/模糊命中明细）
│
├── tieba/                       # 贴吧年度佳帖数据 (2018-2025)
│   ├── best365_2025.json / _classified.json / _classified_dedup.json
│   ├── best365_2024.json / _classified.json / _classified_dedup.json
│   ├── best365_2023.json / _classified.json / _classified_dedup.json
│   ├── best365_2022.json / _classified.json / _classified_dedup.json
│   ├── best306_2021_2H.json / _classified.json / _classified_dedup.json
│   ├── best295_2021_1H.json / _classified.json / _classified_dedup.json
│   ├── best365_2020.json / _classified.json / _classified_dedup.json
│   ├── best336_2019.json / _classified.json / _classified_dedup.json
│   ├── best176_2018.json / _classified.json / _classified_dedup.json
│   └── best100_all.json         # 按类别组织的经典 100 条
│
├── CQIA/                        # COIG-CQIA 弱智吧子集（测试集）
│   ├── ruozhiba_ruozhiba.jsonl  #   原始 JSONL (240 条)
│   ├── ruozhiba_cqia_cleaned.json #  清洗后 JSON
│   ├── ruozhiba_cqia_classified.json # LLM 分类后（仅 top3_categories）
│   ├── ruozhiba_cqia_classified_v2.json # 补全 thought_process 后的最终版
│   ├── ruozhiba_cqia_test.json  #   测试集
│   └── ruozhiba_cqia_test_classified.json
│
├── ruozhiba/                    # GitHub 弱智吧语料
│   └── data/
│       ├── ruozhiba-post-annual.json              # 原始 1361 条年度精选
│       ├── ruozhiba-post-annual-processed.json     # 预处理后
│       ├── ruozhiba-post-annual-processed_filtered.json # 去重后
│       ├── ruozhiba-post-annual-processed_exact_matched.json
│       ├── ruozhiba-post-annual-processed_fuzzy_matched.json
│       ├── ruozhiba-title-good.json                # 吧主推荐帖
│       └── ruozhiba-title-norm.json                # 普通帖标题
│
└── LLaMA-Factory/               # SFT 训练数据备份
    └── data/
        ├── ruozhiba_all.json    #   全量训练集 (2018-2025, 2785 条)
        ├── ruozhiba_last3.json  #   近三年训练集 (2023-2025, 1025 条)
        └── dataset_info.json    #   LLaMA-Factory 数据集注册配置
```

---

## 数据来源

| 来源 | 说明 | 链接 |
|------|------|------|
| **贴吧年度帖** | 2018-2025 年度弱智吧精选帖，通过爬虫提取 | 详见 `crawler/` 目录 |
| **COIG-CQIA** | Hugging Face 上的中文指令数据集弱智吧子集 (240 条) | https://huggingface.co/datasets/m-a-p/COIG-CQIA/tree/main/ruozhiba |
| **GitHub 语料** | 已有的弱智吧历史数据集 (2018-2021, 1361 条) | https://github.com/Leymore/ruozhiba |

---

## 数据统计

### 训练集（贴吧年度佳帖，去重后）

| 文件 | 去重前 | 去重后 | 移除 |
|------|--------|--------|------|
| best176_2018 | 170 | 170 | 0 |
| best336_2019 | 333 | 333 | 0 |
| best365_2020 | 353 | 350 | 3 |
| best295_2021_1H | 286 | 283 | 3 |
| best306_2021_2H | 306 | 296 | 10 |
| best365_2022 | 330 | 329 | 1 |
| best365_2023 | 322 | 312 | 10 |
| best365_2024 | 352 | 352 | 0 |
| best365_2025 | 361 | 361 | 0 |
| **合计** | **2813** | **2786** | **27** |

> 去重方式：CQIA 测试集 vs 贴吧训练集，MD5 精确匹配 (18 条) + SequenceMatcher ≥ 0.9 模糊匹配 (9 条)。

### 测试集

| 数据集 | 条数 | 说明 |
|--------|------|------|
| CQIA 弱智吧 | 240 | instruction/output 对，已补全 `thought_process`（v2），用于推理评估 |

### SFT 训练数据（ShareGPT 格式）

| 文件 | 条数 | 覆盖范围 |
|------|------|----------|
| `ruozhiba_all.json` | 2,785 | 2018-2025 全量（跳过 1 条缺失 thought_process） |
| `ruozhiba_last3.json` | 1,025 | 2023-2025 近三年 |
| `dataset_info.json` | — | LLaMA-Factory 数据集注册配置（ShareGPT 格式 role/tag 映射） |

---

## 数据处理流程

```
原始数据                    处理流程                      输出
──────                    ────────                    ────
贴吧爬虫 (crawler/)   ──→ extract_posts.py         ──→ best*_YYYY.json
CQIA JSONL           ──→ extract_cqia_data.py      ──→ ruozhiba_cqia_cleaned.json
GitHub 语料           ──→ process_ruozhiba_past_    ──→ ruozhiba-post-annual-processed.json
                          annual.py
                              │
                              ▼
                     filter_duplicates.py           ──→ *_filtered.json
                              │
                              ▼
                     classify_jokes.py              ──→ *_classified.json
                     classify_cqia.py
                     classify_cqia_updated.py       ──→ *_classified_v2.json
                              │
                              ▼
                     check_and_repair.py            ──→ 完整性校验
                              │
                              ▼
                     dedup_test_vs_train.py         ──→ *_classified_dedup.json
                              │
                              ▼
                     build_sft_data.py              ──→ ruozhiba_all.json (ShareGPT)
```

---

## 参考资料

- COIG-CQIA 数据集: https://huggingface.co/datasets/m-a-p/COIG-CQIA
- GitHub 弱智吧语料: https://github.com/Leymore/ruozhiba
- Qwen3 + LLaMA-Factory 微调: https://qwen.readthedocs.io/en/latest/training/llama_factory.html
- LLaMA-Factory 文档: https://github.com/hiyouga/LLaMA-Factory
- 其他参考: https://github.com/nick7nlp/evol-ruozhiba
