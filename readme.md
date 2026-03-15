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
│   │   ├── readme.md
│   │   └── todo.md
│   ├── processing_scripts/            # 爬取后处理脚本
│   │   ├── check_missing.py           #   检查缺失帖子
│   │   ├── extract_all_years.py       #   批量提取所有年份数据
│   │   ├── extract_posts.py           #   从原始页面提取帖子内容
│   │   └── README.md
│   └── threads/                       # 爬取的原始帖子数据
│       ├── index.json                 #   帖子索引
│       ├── 目录.txt                    #   帖子目录说明
│       ├── tieba.baidu.com_cookies.txt
│       ├── 10354221105_弱智吧2025年度365佳贴/
│       ├── 9354404050_弱智吧2024年度365佳贴/
│       ├── 8807455743_弱智吧2023年度365佳贴/
│       ├── 8202002333_弱智吧2022年度365佳贴/
│       ├── 7708339765_弱智吧2021年度306佳贴/
│       ├── 7339397421_弱智吧2020年度365佳贴/
│       └── 10130417881_盘点弱智吧最出圈的100条段子/
│
├── data/                              # 📦 整理后的数据集
│   ├── readme.md                      #   数据说明
│   ├── catogory_ideas.md              #   分类体系设计笔记
│   ├── init_ideas.md                  #   初期构思
│   ├── tieba/                         #   贴吧年度佳帖数据 (2018-2025)
│   │   ├── best365_2025.json          #     原始提取数据
│   │   ├── best365_2025_classified.json #   LLM 分类后数据
│   │   ├── best365_2024.json / _classified.json
│   │   ├── best365_2023.json / _classified.json
│   │   ├── best365_2022.json / _classified.json
│   │   ├── best365_2020.json / _classified.json
│   │   ├── best306_2021_2H.json / _classified.json
│   │   ├── best295_2021_1H.json / _classified.json
│   │   ├── best336_2019.json          #     (2018-2019 年暂未分类)
│   │   ├── best176_2018.json
│   │   ├── best100_all.json           #     按类别组织的经典 100 条
│   │   └── readme.md
│   ├── CQIA/                          #   COIG-CQIA 弱智吧子集
│   │   ├── ruozhiba_ruozhiba.jsonl    #     原始 JSONL (240 条)
│   │   ├── ruozhiba_cqia_cleaned.json #     清洗后 JSON
│   │   ├── ruozhiba_cqia_classified.json #  LLM 分类后
│   │   ├── ruozhiba_cqia_test.json    #     测试集
│   │   └── ruozhiba_cqia_test_classified.json
│   └── ruozhiba/                      #   GitHub 弱智吧语料
│       ├── README.md
│       └── data/
│           ├── ruozhiba-post-annual.json           # 1361 条年度精选
│           ├── ruozhiba-post-annual-processed.json  # 处理后
│           ├── ruozhiba-post-annual-processed_filtered.json
│           ├── ruozhiba-post-annual-processed_exact_matched.json
│           ├── ruozhiba-post-annual-processed_fuzzy_matched.json
│           ├── ruozhiba-title-good.json             # 吧主推荐帖
│           └── ruozhiba-title-norm.json             # 普通帖标题
│
├── scripts/                           # ⚙️ 数据处理与分类脚本
│   ├── readme.md                      #   脚本说明
│   ├── classify_jokes.py              #   LLM 批量分类 (多线程, 断点续传)
│   ├── classify_cqia.py               #   CQIA 数据 LLM 分类
│   ├── classify_config.yaml           #   分类 API 配置
│   ├── classify_cqia_config.yaml      #   CQIA 分类配置
│   ├── extract_annual_data.py         #   按时间段提取年度数据
│   ├── extract_cqia_data.py           #   CQIA JSONL → JSON 转换
│   ├── process_ruozhiba_past_annual.py #  GitHub 语料预处理
│   ├── filter_duplicates.py           #   精确 + 模糊去重
│   ├── check_and_repair.py            #   数据完整性校验与修复
│   ├── check_and_repair_cqia.py       #   CQIA 数据校验与修复
│   ├── check_escape.py                #   转义字符检查
│   ├── fix_quotes.py                  #   引号修复
│   ├── fix_double_escapes.py          #   双重转义修复
│   ├── debug_quotes.py                #   引号问题调试
│   ├── test_*.py                      #   单元测试
│   └── ...
│
└── doc/                               # 📝 文档
    ├── assignment.md                  #   作业要求
    ├── howto.md                       #   操作指南
    ├── Lab3_SFT.pdf                   #   实验指导书
    └── proposal/                      #   项目方案
        ├── final_proposal_gemini_update.md  # 最终方案
        ├── final_proposal_gemini.md
        ├── deepseek.md / gemini.md / qwen.md  # 各模型生成的方案草稿
        └── my_ideas_qa.md
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
                          │  (LLM 多线程分类，断点续传)   │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  check_and_repair.py       │
                          │  (数据完整性校验)             │
                          └─────────────┬──────────────┘
                                        │
                          ┌─────────────▼──────────────┐
                          │  SFT Training Data         │
                          │  (ShareGPT format)         │
                          └────────────────────────────┘
```

---

## 技术栈

- **语言**: Python 3.12
- **环境管理**: [uv](https://docs.astral.sh/uv/) (虚拟环境 `env_sft`)
- **爬虫**: [aiotieba](https://github.com/Starry-OvO/aiotieba) (异步贴吧库)
- **SFT 微调**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) + LoRA + [accelerate](https://github.com/huggingface/accelerate)
- **基座模型**: Qwen3-8B-Instruct / Qwen3-1.5B-Instruct

### 数据处理 & 分类脚本依赖

| 包名 | 用途 |
|------|------|
| `openai` | LLM API 调用 (兼容 OpenAI 接口) |
| `tenacity` | API 调用重试 (指数退避) |
| `tqdm` | 进度条显示 |
| `python-dotenv` | `.env` 环境变量加载 |
| `pyyaml` | YAML 配置文件解析 |

标准库: `json`, `re`, `os`, `glob`, `time`, `pathlib`, `typing`, `datetime`, `difflib`, `concurrent.futures`, `multiprocessing`

```
uv venv env_sft python 3.12
source env_sft/bin/activate
uv pip install 'llamafactory[metrics]' accelerate
uv pip install openai tenacity tqdm python-dotenv pyyaml
```
