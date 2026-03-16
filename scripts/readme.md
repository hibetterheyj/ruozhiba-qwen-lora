# 脚本说明文档

> 本目录包含项目全流程中的数据处理、LLM 分类、SFT 数据构建、去重防污染、训练、推理、评估等脚本。

---

## 数据处理流水线脚本

以下脚本按项目实际执行顺序排列：

### 1. 数据提取与预处理

| 脚本 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `extract_cqia_data.py` | 从 CQIA 原始 JSONL 中提取 `instruction` 和 `output` 字段 | `data/CQIA/ruozhiba_ruozhiba.jsonl` | `data/CQIA/ruozhiba_cqia_cleaned.json` |
| `process_ruozhiba_past_annual.py` | GitHub 弱智吧年度精选数据预处理：提取序号和正文，附加元数据 (`l_num`, `ctime`)，按时间排序 | `data/ruozhiba/data/ruozhiba-post-annual.json` | `data/ruozhiba/data/ruozhiba-post-annual-processed.json` |
| `filter_duplicates.py` | GitHub 语料与贴吧数据交叉去重（精确匹配 + 模糊匹配，SequenceMatcher 阈值 0.5） | 处理后的 GitHub 语料 + 贴吧数据 | `ruozhiba-post-annual-processed_filtered.json` + 匹配记录 |
| `extract_annual_data.py` | 从去重后的 GitHub 语料中按时间段提取 2018/2019 年度数据，转换为贴吧格式 | `ruozhiba-post-annual-processed_filtered.json` | `data/tieba/best176_2018.json`, `data/tieba/best336_2019.json` |

### 2. LLM 分类标注

| 脚本 | 功能 | 配置文件 | 说明 |
|------|------|----------|------|
| `classify_jokes.py` | 贴吧段子 LLM 批量分类（8 类 Top-3 + 思考过程） | `classify_config.yaml` | 多线程、断点续传、原子写入 |
| `classify_cqia.py` | CQIA 数据集 LLM 分类 | `classify_cqia_config.yaml` | 处理 instruction/output 对 |
| `classify_cqia_updated.py` | CQIA 数据补全：为 240 条已分类数据补充 `thought_process` 字段（导师蒸馏） | `classify_cqia_updated_config.yaml` | 使用 Claude-Opus-4-6 生成深度分析 |

#### `classify_jokes.py` 工程优化历史（供报告参考）

1. **JSON 解析鲁棒性** — 应对 LLM 生成的非标准 JSON：
   - `fix_double_escaped_quotes()`: 修复 `\\"` → `"` 双重转义
   - `fix_unescaped_quotes()`: 修复字符串内未转义引号
   - `extract_json_from_response()`: 多层回退提取（Markdown 剥离 → 直接解析 → 引号修复 → 正则匹配）
   - 使用 `{**item, ...}` 展开保留所有原始字段

2. **并发优化** — 提升 I/O 密集型任务性能：
   - `multiprocessing.Pool` → `ThreadPoolExecutor`（更适合 API 调用场景）
   - 全线程共享单一 OpenAI 客户端实例（避免万次级冗余初始化）

3. **断点续传** — 防止崩溃/中断导致数据丢失：
   - `load_existing_results()`: 按 `no` 字段跳过已处理条目
   - `try-except-finally`: 确保 `KeyboardInterrupt` 或异常时保存进度
   - 全局预分配 `no` 索引，防止 ID 冲突

4. **边界情况处理** — 生产级可靠性：
   - 原子写入：先写 `.json.tmp` 再 `replace()`，防止文件损坏
   - 防御性 API 响应提取：检查空 `choices` 或 `content`（Content Filter 情况）
   - `tqdm.write()` 替代 `print()`：保持进度条渲染完整
   - `num_processes` → `max_workers` 语义重命名

#### `classify_cqia_updated.py` 设计要点

- System Prompt 与 `classify_config.yaml`（贴吧版）对齐，要求输出 `thought_process` + `top3_categories`
- 仅使用 `instruction` 字段作为 LLM 输入（不使用 CQIA 的 `output` 字段）
- 保留原有 `output`、`top3_categories` 不变，在 `classification` 中新增 `thought_process`
- 复用 `classify_jokes.py` 的鲁棒性优化：ThreadPoolExecutor、断点续传（JSONL checkpoint）、原子写入、多层 JSON 解析容错
- 新旧 `top3_categories` 对比记录日志（仅记录 category drift，不覆盖原有分类）

### 3. 数据质量校验与修复

| 脚本 | 功能 |
|------|------|
| `check_and_repair.py` | 贴吧分类数据完整性校验与修复（检查 `thought_process` + `top3_categories` 字段完整性） |
| `check_and_repair_cqia.py` | CQIA 数据 schema 校验与修复 |
| `check_escape.py` | JSON 转义字符检查 |
| `fix_quotes.py` | 引号修复（处理 LLM 输出中的引号异常） |
| `fix_double_escapes.py` | 双重转义修复 |
| `debug_quotes.py` | 引号问题调试辅助 |

### 4. 去重防污染

| 脚本 | 功能 | 产出 |
|------|------|------|
| `dedup_test_vs_train.py` | CQIA 测试集 vs 贴吧训练集去重（MD5 精确匹配 + SequenceMatcher ≥ 0.9 模糊匹配） | `data/dedup_report.json` + 9 个 `*_classified_dedup.json` |

去重结果：2813 → **2786** 条（移除 27 条重复：精确 18 条 + 模糊 9 条）

### 5. SFT 训练数据构建

| 脚本 | 功能 | 产出 |
|------|------|------|
| `build_sft_data.py` | 将去重后贴吧分类数据转换为 LLaMA-Factory ShareGPT 格式（system/human/gpt 三轮对话） | `LLaMA-Factory/data/ruozhiba_all.json` (2785 条), `ruozhiba_last3.json` (1025 条) |

### 6. 训练与合并

| 脚本 | 功能 | 说明 |
|------|------|------|
| `run_training.sh` | 训练启动脚本 | 通过 CLI 参数注入 `CUDA_VISIBLE_DEVICES`、`lora_rank`、`lora_alpha`、`output_dir` |
| `probe_batch_size.sh` | Batch Size 动态压测 | `max_steps: 15` 小步快跑，自动激活 venv，自动清理临时文件 |
| `batch_merge.sh` | 批量 LoRA 权重合并 | 串行合并 4 组训练 × Epoch 3-7 共 20 个 checkpoint 至 `models/merged/` |

**`run_training.sh` 使用方式:**

```bash
# 用法: bash scripts/run_training.sh <GPU_ID> <RANK>
bash scripts/run_training.sh 0 8    # GPU 0, LoRA rank=8, alpha=16
bash scripts/run_training.sh 1 16   # GPU 1, LoRA rank=16, alpha=32
```

### 7. 推理与评估

| 脚本 | 功能 | 产出 |
|------|------|------|
| `inference_eval.py` | vLLM 离线批量推理（单模型/批量/多模型三种模式） | `results/results_{tag}.json` |
| `batch_inference.sh` | 批量推理封装（21 个模型串行） | 21 个推理结果文件 |
| `eval_metrics.py` | 两阶段 JSON 评估（格式遵循 + 逻辑准确率）+ 可视化 | `results/json/` + `results/heatmaps/` + `results/confusion_matrices/` + `results/charts/` |
| `gen_before_after.py` | Before/After 对比样本自动选取 | `results/before_after_samples.json` |

**推理用法:**

```bash
# 批量推理 21 个模型
bash scripts/batch_inference.sh 0    # GPU 0

# 单模型推理
python scripts/inference_eval.py \
    --model_path models/Qwen3-4B-Instruct-2507 --tag baseline --gpu 0
```

**评估用法:**

```bash
# 批量评估 + 对比总表 + 可视化
python scripts/eval_metrics.py \
    --results_dir results/ \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json \
    --comparison
```

### 8. 单元测试

| 脚本 | 测试对象 |
|------|----------|
| `test_fix_function.py` | 修复函数单元测试 |
| `test_fix_quotes.py` | 引号修复逻辑测试 |
| `test_json_parse.py` | JSON 解析容错测试 |
| `test_actual_file.py` | 实际文件处理测试 |

---

## 配置文件

| 文件 | 用途 | 关键参数 |
|------|------|----------|
| `classify_config.yaml` | `classify_jokes.py` 配置 | system_prompt（8 类分类 + 思考过程）、files_to_process、API 参数 |
| `classify_cqia_config.yaml` | `classify_cqia.py` 配置 | 简化输出格式，适配 CQIA 数据结构 |
| `classify_cqia_updated_config.yaml` | `classify_cqia_updated.py` 配置 | system_prompt 对齐贴吧版、`max_workers: 4`、`temperature: 0.3`、`max_tokens: 1500` |

---

## 依赖

| 包名 | 用途 |
|------|------|
| `openai` | LLM API 调用（兼容 OpenAI 接口） |
| `tenacity` | API 调用重试（指数退避） |
| `tqdm` | 进度条显示 |
| `pyyaml` | YAML 配置文件解析 |
| `python-dotenv` | `.env` 环境变量加载 |
| `vllm` | 离线批量推理引擎 |
| `json-repair` | JSON 修复解析（容错评估） |
| `matplotlib` | 图表可视化 |
| `seaborn` | 热力图/混淆矩阵可视化 |
| `numpy` | 数值计算 |
