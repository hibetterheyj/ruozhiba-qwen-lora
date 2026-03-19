# 脚本说明文档

> 本目录按流水线阶段分子文件夹，路径均相对**仓库根目录**。Python 脚本通过 `Path(__file__).resolve().parents[2]` 定位根目录，以正确读写 `configs/`、`data/`、`logs/`、`models/`、`results/`、`LLaMA-Factory/`、`crawler/` 等。

**目录重构与 `upload/` 同步说明** → [`changelog.md`](changelog.md)。

## 目录一览

| 子目录 | 用途 |
|--------|------|
| **`crawl/`** | 抓取/抽取：从 HF CQIA、GitHub 年度语料等原始来源抽取字段 |
| **`data/`** | 数据处理：去重、LLM 分类、校验修复、测试集去污染、ShareGPT 构建；YAML 配置与分类脚本同目录 |
| **`train/`** | 训练与合并：LoRA 启动、Batch 压测、`llamafactory-cli export` 批量合并 |
| **`inference/`** | 推理：`vLLM` 离线批量推理及 21 模型串行封装 |
| **`viz/`** | 评估与可视化：两阶段 JSON 指标、图表、Before/After 样本选取 |
| **`tests/`** | 单元/调试向的小脚本（引号、JSON 解析等） |

---

## `crawl/` — 抓取与抽取

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `extract_cqia_data.py` | 从 CQIA 原始 JSONL 中提取 `instruction` 和 `output` | `data/CQIA/ruozhiba_ruozhiba.jsonl` | `data/CQIA/ruozhiba_cqia_cleaned.json` |
| `process_ruozhiba_past_annual.py` | GitHub 弱智吧年度精选预处理：序号与正文、`l_num`/`ctime`、按时间排序 | `data/ruozhiba/data/ruozhiba-post-annual.json` | `data/ruozhiba/data/ruozhiba-post-annual-processed.json` |
| `extract_annual_data.py` | 从去重后 GitHub 语料按时间段提取 2018/2019，转贴吧格式 | `data/ruozhiba/data/ruozhiba-post-annual-processed_filtered.json` | `data/tieba/best176_2018.json` 等 |

贴吧爬虫原始页数据见仓库 `crawler/threads/`（aiotieba 等），不在 `scripts/crawl/` 内。

---

## `data/` — 数据处理流水线

以下按推荐执行顺序排列。

### 1. 交叉去重

| 脚本 | 功能 |
|------|------|
| `filter_duplicates.py` | GitHub 语料与贴吧数据去重（精确 + 模糊 SequenceMatcher 0.5） |

### 2. LLM 分类标注

| 脚本 | 配置文件 |
|------|----------|
| `classify_jokes.py` | `classify_config.yaml` |
| `classify_cqia.py` | `classify_cqia_config.yaml` |
| `classify_cqia_updated.py` | `classify_cqia_updated_config.yaml` |

#### `classify_jokes.py` 工程优化要点（供报告参考）

1. **JSON 解析鲁棒性** — `fix_double_escaped_quotes` / `fix_unescaped_quotes` / `extract_json_from_response` 等多层回退  
2. **并发** — `ThreadPoolExecutor` + 共享 OpenAI 客户端  
3. **断点续传** — 按 `no` 跳过已完成；原子写入（`.tmp` + `replace()`）  
4. **边界** — 空 `choices`、Content Filter；`tqdm.write` 保持进度条

#### `classify_cqia_updated.py` 设计要点

- System Prompt 与贴吧版 `classify_config.yaml` 对齐，补全 `thought_process`  
- 仅 `instruction` 作为 LLM 输入；断点续传与原子写入与 `classify_jokes.py` 一致  

### 3. 质量校验与修复

| 脚本 | 功能 |
|------|------|
| `check_and_repair.py` | 贴吧分类字段完整性 |
| `check_and_repair_cqia.py` | CQIA schema |
| `check_escape.py` | 转义检查 |
| `fix_quotes.py` / `fix_double_escapes.py` / `debug_quotes.py` | 引号相关修复与调试 |

### 4. 去重防污染

| 脚本 | 产出 |
|------|------|
| `dedup_test_vs_train.py` | `data/dedup_report.json`、`data/tieba/*_classified_dedup.json` |

### 5. SFT 数据构建

| 脚本 | 产出 |
|------|------|
| `build_sft_data.py` | `LLaMA-Factory/data/ruozhiba_all.json`、`ruozhiba_last3.json`（及 `dataset_info` 注册说明） |

---

## `train/` — 训练与合并

| 脚本 | 功能 |
|------|------|
| `run_training.sh` | 注入 `CUDA_VISIBLE_DEVICES`、`lora_rank`/`alpha`、`output_dir`，调用 `llamafactory-cli train` |
| `probe_batch_size.sh` | 小步数压测最大 `per_device_train_batch_size` |
| `batch_merge.sh` | 串行合并多组 checkpoint → `models/merged/` |

```bash
# 用法: bash scripts/train/run_training.sh <GPU_ID> <RANK> [CONFIG] [TAG]
bash scripts/train/run_training.sh 0 8
bash scripts/train/run_training.sh 1 16
```

---

## `inference/` — 批量推理

| 脚本 | 功能 |
|------|------|
| `inference_eval.py` | vLLM 单模型 / 多模型 / 扫描目录 |
| `batch_inference.sh` | 21 个模型串行推理 |

```bash
bash scripts/inference/batch_inference.sh 0    # GPU 0

python scripts/inference/inference_eval.py \
    --model_path models/Qwen3-4B-Instruct-2507 --tag baseline --gpu 0
```

默认测试集：`data/CQIA/ruozhiba_cqia_classified_v2.json`；prompt：`configs/prompts.yaml`；输出目录：`results/`。

---

## `viz/` — 评估与可视化

| 脚本 | 产出 |
|------|------|
| `eval_metrics.py` | `results/json/`、`heatmaps/`、`confusion_matrices/`、`charts/` 等 |
| `gen_before_after.py` | `results/before_after_samples.json` |

```bash
python scripts/viz/eval_metrics.py \
    --results_dir results/ \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json \
    --comparison

python scripts/viz/gen_before_after.py

# 自定义对比输入 / 输出
python scripts/viz/gen_before_after.py \
    --baseline results/results_baseline.json \
    --candidate results/results_r16_e5.json \
    --output results/before_after_samples.json
```

---

## `tests/` — 调试 / 诊断脚本

| 脚本 | 说明 |
|------|------|
| `test_fix_function.py` | JSON 修复函数调试（`main()` + CLI，可指定文件/样本） |
| `test_fix_quotes.py` | ASCII → 中文引号转换调试（复用 `scripts/data/fix_quotes.py` 实现） |
| `test_json_parse.py` | JSON 解析容错诊断（清理损坏替换逻辑，使用 `logging`） |
| `test_actual_file.py` | 指定样本检视（CLI 指定路径与 `no`，默认相对仓库根 `data/tieba/`） |

> `scripts/tests/` 当前定位为 **debug / adhoc 诊断脚本**，用于手工复核引号修复、JSON 提取与真实样本排查；已统一补充 `main()`、类型标注与 `logging`，但不作为正式自动化测试套件。

---

## 依赖（摘录）

| 包名 | 用途 |
|------|------|
| `openai` / `tenacity` / `tqdm` / `pyyaml` / `python-dotenv` | 分类与配置 |
| `vllm` | 推理 |
| `json-repair` / `matplotlib` / `seaborn` / `numpy` | 评估与作图 |

---

## 2026-03-19 收尾说明

- `scripts/tests/` 下 4 个脚本已完成一轮工程化收尾：不再“导入即执行”，统一采用 `main()` + CLI / `logging`。
- `scripts/crawl/extract_annual_data.py`、`scripts/data/fix_double_escapes.py`、`scripts/viz/gen_before_after.py` 已将残留 `print` 迁移为 `logging`。
- `scripts/viz/gen_before_after.py` 已支持通过 CLI 参数覆盖 baseline / candidate / output，默认行为保持兼容。
- `upload/scripts/` 与主仓库在**保留的核心脚本子目录**上保持同构；`upload/` 不包含 `scripts/tests/`。

---

## 最小提交包 `upload/`

精简复现脚本与主仓库**保留的核心脚本子路径同构**（例如 `upload/scripts/train/run_training.sh`），默认在**完整克隆的仓库根目录**下执行；说明见 [`upload/readme.md`](../upload/readme.md)。
