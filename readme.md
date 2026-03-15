# llm_ruozhiba

> **CSS5120 计算语言学 Lab3: Supervised Fine-Tuning (SFT)**
>
> 基于 **弱智吧 (Ruozhiba)** 语料的中文幽默分类模型微调项目。

## 项目简介

[弱智吧](https://tieba.baidu.com/f?kw=%E5%BC%B1%E6%99%BA) 是百度贴吧上著名的中文互联网幽默社区。本项目围绕该社区的段子数据，完成以下流程：

1. **数据采集** — 从贴吧年度帖、GitHub 已有语料、Hugging Face COIG-CQIA 数据集三个来源收集约 3000+ 条弱智吧段子
2. **幽默分类** — 利用 LLM 将每条段子标注为 8 种幽默类型中的 top-3 类别（含置信度）
3. **SFT 微调** — 基于分类标注数据，使用 LoRA 微调 Qwen3 系列模型
4. **效果评估** — 定量 (BLEU/ROUGE) + 定性 (前后对比) + 人工评估

### 8 种幽默分类

| 类别 | 说明 | 示例 |
|------|------|------|
| **古典弱智** | 荒诞因果、不合逻辑的行为 | "慢着，这屎里有毒。" |
| **奇怪提问** | 利用逻辑/语义漏洞提出怪问题 | "咖啡严格来说是不是也可以叫豆浆？" |
| **弱智科学家** | 科学术语 + 荒谬结论 | — |
| **人生态度** | 自嘲式哲学世界观 | — |
| **文字游戏** | 一词多义、语义错位 | "四川人至死不渝，重庆人乐不思蜀" |
| **地狱笑话** | 黑色幽默、缺乏共情 | — |
| **谐音梗** | 利用同音/近音字制造笑点 | — |
| **文艺弱智** | 诗意语言 + 荒诞主题 | — |

---

## 项目结构

```
.
├── readme.md                          # 本文件
│
├── crawler/                           # 🕷️ 数据爬取
│   ├── keaixiaojiycw-tieba-post-crawler/  # 基于 aiotieba 的异步贴吧爬虫
│   ├── processing_scripts/            # 爬取后处理脚本
│   └── threads/                       # 爬取的原始帖子数据 (2018-2025)
│
├── data/                              # 📦 整理后的数据集
│   ├── readme.md                      #   数据目录说明
│   ├── dedup_report.json              #   去重报告
│   ├── tieba/                         #   贴吧年度佳帖 (2018-2025, 含分类/去重版本)
│   ├── CQIA/                          #   COIG-CQIA 弱智吧子集 (240 条, 测试集)
│   ├── ruozhiba/                      #   GitHub 弱智吧语料 (1361 条)
│   └── LLaMA-Factory/                 #   SFT 训练数据备份
│       └── data/
│           ├── ruozhiba_all.json      #     全量训练集 (2785 条)
│           └── ruozhiba_last3.json    #     近三年训练集 (1025 条)
│
├── scripts/                           # ⚙️ 数据处理与训练脚本
│   ├── readme.md                      #   脚本说明
│   ├── classify_jokes.py              #   LLM 批量分类 (多线程, 断点续传)
│   ├── classify_cqia.py               #   CQIA 数据 LLM 分类
│   ├── classify_cqia_updated.py       #   CQIA thought_process 补全 (导师蒸馏)
│   ├── build_sft_data.py              #   去重数据 → ShareGPT 格式转换
│   ├── dedup_test_vs_train.py         #   测试集 vs 训练集去重
│   ├── filter_duplicates.py           #   精确 + 模糊去重
│   ├── run_training.sh                #   训练启动脚本 (CLI 覆盖 rank/output_dir)
│   ├── probe_batch_size.sh            #   Batch Size 动态压测
│   └── ...                            #   校验/修复/测试脚本
│
├── configs/                           # 🔧 训练与推理配置
│   ├── readme.md                      #   配置说明
│   ├── prompts.yaml                   #   统一 system prompt 与分类类别
│   ├── qwen3_4b_mvp.yaml             #   MVP 训练配置 (rank=8, 3 epochs)
│   ├── qwen3_4b_base.yaml            #   正式训练基础配置 (7 epochs, BS=16×2)
│   └── qwen3_4b_merge.yaml           #   LoRA 权重合并配置
│
├── models/                            # 🤖 模型权重
│   ├── Qwen3-4B-Instruct-2507/       #   基座模型
│   └── Qwen3-4B-Ruozhiba-Merged/     #   合并后的微调模型 (7.6 GB)
│
├── LLaMA-Factory/                     # 🏭 LLaMA-Factory 框架 (submodule)
│   ├── data/                          #   训练数据 + dataset_info.json
│   └── saves/                         #   训练产物
│       └── qwen3-4b/lora/
│           ├── mvp_r8_e3/             #     MVP 训练输出
│           ├── r8/                    #     正式训练 R8 (7 epoch checkpoints)
│           └── r16/                   #     正式训练 R16 (7 epoch checkpoints)
│
├── doc/                               # 📝 文档
│   ├── readme.md                      #   文档目录说明
│   ├── assignment.md                  #   作业要求
│   ├── dev_plan_0_1.md                #   开发计划 v0.1
│   ├── changelog.md                   #   变更日志
│   ├── training_execution.md          #   双卡并行训练执行手册
│   ├── train_test_eda.md              #   Token 长度 EDA 报告
│   ├── train_analysis1.md             #   第一批训练日志分析
│   └── proposal/                      #   项目方案
│
└── env_sft/                           # 🐍 Python 虚拟环境
```

---

## 数据格式详解

### 1. 贴吧年度佳帖 — 原始格式 (`data/tieba/best*_YYYY.json`)

每年的弱智吧精选帖，JSON 数组，每个元素是一条段子：

```json
[
  {
    "no": 1,
    "text": "四川人至死不渝，重庆人乐不思蜀。"
  },
  {
    "no": 2,
    "text": "..."
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `no` | `int` | 帖子排名序号 |
| `text` | `string` | 段子正文 |
| `score` | `int \| null` | 得分（部分年份有） |

**文件命名规则**: `best{数量}_{年份}{可选后缀}.json`
- `best365_2020.json` → 2020 年 365 佳帖
- `best306_2021_2H.json` → 2021 年下半年 306 佳帖
- `best176_2018.json` → 2018 年 176 佳帖（因帖子删除，实际数量可能少于标题数字）

### 2. 分类后数据 (`data/tieba/best*_classified.json`)

在原始字段基础上增加 `classification` 对象：

```json
[
  {
    "no": 1,
    "text": "四川人至死不渝，重庆人乐不思蜀。",
    "score": null,
    "classification": {
      "thought_process": "这句话的核心笑点在于谐音和双关...",
      "top3_categories": [
        {
          "rank": 1,
          "category": "文字游戏",
          "confidence_score": 0.7,
          "reason": "核心笑点在于成语中'渝'和'蜀'的一词双义..."
        },
        {
          "rank": 2,
          "category": "谐音梗",
          "confidence_score": 0.2,
          "reason": "..."
        },
        {
          "rank": 3,
          "category": "文艺弱智",
          "confidence_score": 0.1,
          "reason": "..."
        }
      ]
    }
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `classification.thought_process` | `string` | LLM 的分析推理过程 |
| `classification.top3_categories` | `array` | 前 3 个最可能的分类 |
| `top3_categories[].rank` | `int` | 排名 (1-3) |
| `top3_categories[].category` | `string` | 分类名称 (8 种之一) |
| `top3_categories[].confidence_score` | `float` | 置信度 (0-1)，三者之和为 1 |
| `top3_categories[].reason` | `string` | 分类理由 |

### 3. CQIA 指令-回答数据 (`data/CQIA/`)

来源: [COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) 弱智吧子集。

**原始 JSONL** (`ruozhiba_ruozhiba.jsonl`)，每行一个 JSON 对象：

```jsonl
{"instruction": "咖啡严格来说是不是也可以叫豆浆？", "input": "", "output": "不，咖啡和豆浆是两种完全不同的饮品...", "task_type": {"major": ["问答"], "minor": ["逻辑问答", "隐喻理解"]}, "domain": ["通用"], "metadata": "...", "answer_from": "llm", "human_verified": true, "copyright": "..."}
```

**清洗后 JSON** (`ruozhiba_cqia_cleaned.json`)，仅保留核心字段：

```json
[
  {
    "instruction": "咖啡严格来说是不是也可以叫豆浆？",
    "output": "不，咖啡和豆浆是两种完全不同的饮品..."
  }
]
```

### 4. GitHub 弱智吧语料 (`data/ruozhiba/data/`)

来源: GitHub 上已有的弱智吧数据集 (2018-2021 年)。

**年度精选** (`ruozhiba-post-annual.json`):

```json
[
  {
    "author": "公孙闬",
    "content": "151、家电下乡送出了杨永信保证每家都能电一下。",
    "l_num": 2,
    "ctime": "2018-12-24 00:02"
  }
]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `author` | `string` | 作者用户名 |
| `content` | `string` | 帖子内容（含序号前缀） |
| `l_num` | `int` | 楼层号 |
| `ctime` | `string` | 发布时间 (`YYYY-MM-DD HH:MM`) |

### 5. 经典 100 条 — 按类别分组 (`data/tieba/best100_all.json`)

```json
[
  {
    "category": "古典弱智",
    "description": "古典弱智，是弱智吧建吧最初期的风格...",
    "jokes": [
      "院长吃了麻婆豆腐之后，当场被麻婆砍了二十多刀。",
      "小明背井离乡，乡里人再也没能喝上一口井水。"
    ]
  }
]
```

### 6. 爬虫原始数据 (`crawler/threads/`)

每个帖子按 `{帖子ID}_{帖子标题}/` 存储：

```
crawler/threads/10354221105_弱智吧2025年度365佳贴/
├── 10354221105_dump.jsonl     # 完整导出 (每行一个楼层)
├── 10354221105_dump.txt       # 纯文本导出
├── checkpoint.json            # 爬取断点信息
├── extracted.json             # 提取后的结构化数据
├── images/                    # 下载的图片
│   └── 楼层_1_153043158018_图片1.jpg
└── posts/                     # 分页原始 JSON
    ├── page_0001.json
    ├── page_0002.json
    └── ...
```

---

## 数据统计

| 数据源 | 文件 | 条数 |
|--------|------|------|
| 贴吧 2025 年度 | `best365_2025.json` | 361 |
| 贴吧 2024 年度 | `best365_2024.json` | 352 |
| 贴吧 2023 年度 | `best365_2023.json` | 322 |
| 贴吧 2022 年度 | `best365_2022.json` | 330 |
| 贴吧 2021 下半年 | `best306_2021_2H.json` | 306 |
| 贴吧 2021 上半年 | `best295_2021_1H.json` | 286 |
| 贴吧 2020 年度 | `best365_2020.json` | 353 |
| 贴吧 2019 年度 | `best336_2019.json` | 333 |
| 贴吧 2018 年度 | `best176_2018.json` | 170 |
| 经典 100 条 | `best100_all.json` | 8 类 |
| CQIA 弱智吧 | `ruozhiba_ruozhiba.jsonl` | 240 |
| GitHub 年度精选 | `ruozhiba-post-annual.json` | 1361 |

> 注: 实际条数可能少于标题中的数字，因为部分帖子已被删除。

---

## 处理流程

```
                             ┌──────────────────────┐
                             │  Baidu Tieba Crawl   │
                             │  (aiotieba crawler)  │
                             └──────────┬───────────┘
                                        │
┌────────────────────┐    ┌─────────────▼──────────────┐    ┌──────────────────┐
│  COIG-CQIA (HF)   │    │  extract / process scripts │    │  GitHub corpus   │
│  240 instruction-  │───▶│  extract_annual_data.py    │◀───│  1361 posts      │
│  output pairs      │    │  extract_cqia_data.py      │    │  (2018-2021)     │
└────────────────────┘    │  process_ruozhiba_past_    │    └──────────────────┘
                          │    annual.py               │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  filter_duplicates.py      │
                          │  (精确 + 模糊去重)           │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  classify_jokes.py         │
                          │  classify_cqia_updated.py  │
                          │  (LLM 多线程分类 + 导师蒸馏)  │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  check_and_repair.py       │
                          │  dedup_test_vs_train.py    │
                          │  (数据校验 + 去重防污染)      │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  build_sft_data.py         │
                          │  (ShareGPT 格式化)          │
                          │  → ruozhiba_all.json       │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  LLaMA-Factory LoRA SFT    │
                          │  run_training.sh           │
                          │  R8 + R16 双卡并行 7 epochs  │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  llamafactory-cli export   │
                          │  (LoRA → 合并模型)           │
                          │  → Qwen3-4B-Ruozhiba-Merged│
                          └────────────────────────────┘
```

---

## 技术栈

- **语言**: Python 3.12
- **环境管理**: [uv](https://docs.astral.sh/uv/) (虚拟环境 `env_sft`)
- **爬虫**: [aiotieba](https://github.com/Starry-OvO/aiotieba) (异步贴吧库)
- **SFT 微调**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) + LoRA + [accelerate](https://github.com/huggingface/accelerate)
- **基座模型**: Qwen3-4B-Instruct-2507
- **分类标注模型**: Claude-Opus-4-6（导师蒸馏，生成 `thought_process`）
- **硬件**: 2× NVIDIA L20Z (80GB VRAM each)

### 数据处理 & 分类脚本依赖

| 包名 | 用途 |
|------|------|
| `openai` | LLM API 调用 (兼容 OpenAI 接口) |
| `tenacity` | API 调用重试 (指数退避) |
| `tqdm` | 进度条显示 |
| `python-dotenv` | `.env` 环境变量加载 |
| `pyyaml` | YAML 配置文件解析 |

标准库: `json`, `re`, `os`, `glob`, `time`, `pathlib`, `typing`, `datetime`, `difflib`, `concurrent.futures`, `multiprocessing`

### 环境搭建

```bash
uv venv env_sft python 3.12
source env_sft/bin/activate
uv pip install 'llamafactory[metrics]' accelerate
uv pip install openai tenacity tqdm python-dotenv pyyaml
```

---

## 训练结果摘要

### 实验矩阵

| Run | GPU | LoRA Rank | Alpha | Epochs | 最优 Eval Loss | 最优 Checkpoint |
|-----|-----|-----------|-------|--------|----------------|-----------------|
| MVP | 0 | 8 | 16 | 3 | 0.8820 | step 993 |
| A (R8) | 0 | 8 | 16 | 7 | 0.8870 | step 500 |
| **B (R16)** | **1** | **16** | **32** | **7** | **0.8859** | **checkpoint-415 (epoch 5)** |

### 正式训练 Loss 对比

| Eval Step (~Epoch) | R8 Train Loss | R8 Eval Loss | R16 Train Loss | R16 Eval Loss |
|---|---|---|---|---|
| 100 (~1.2) | 1.0656 | 1.0295 | 1.0108 | 0.9842 |
| 200 (~2.4) | 0.9081 | 0.9258 | 0.8634 | 0.9034 |
| 300 (~3.6) | 0.8327 | 0.8988 | 0.7801 | 0.8886 |
| 400 (~4.8) | 0.7915 | 0.8885 | 0.7257 | **0.8859** |
| 500 (~6.0) | 0.7848 | 0.8870 | 0.7035 | 0.8915 |

### 关键发现

- **最优模型**: R16 checkpoint-415 (epoch 5)，全局最低 eval_loss = **0.8859**
- R16 学习速度更快（train loss 更低），但 epoch 5 后出现轻微过拟合
- R8 未观察到过拟合，eval loss 持续缓慢下降
- 合并后模型: `models/Qwen3-4B-Ruozhiba-Merged/` (7.6 GB, 参数量 4,022,468,096)

> 详细训练分析见 `doc/train_analysis1.md`
