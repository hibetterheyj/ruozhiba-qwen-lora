# Scripts Documentation

## Python Scripts

### classify_jokes.py
Classifies tieba joke data using LLM API. Reads input files specified in `classify_config.yaml`, sends text to LLM for multi-label classification (8 categories including 古典弱智, 奇怪提问, 弱智科学家, etc.), and outputs classification results with confidence scores.

**Optimization History (for report reference):**

1. **JSON Parsing Robustness** - Addressed LLM-generated JSON format errors:
   - `fix_double_escaped_quotes()`: Fixes `\\"` → `"` double-escaped quotes
   - `fix_unescaped_quotes()`: Fixes unescaped quotes inside JSON string values
   - `extract_json_from_response()`: Multi-layer fallback extraction (Markdown stripping → direct parse → quote fixes → regex pattern matching)
   - Preserves all original data fields using `{**item, ...}` unpacking

2. **Concurrency Optimization** - Improved I/O-bound task performance:
   - Replaced `multiprocessing.Pool` with `ThreadPoolExecutor` (better suited for API calls)
   - Reused single OpenAI client instance across all threads (avoided 10000x redundant instantiation)

3. **Checkpoint & Resume** - Prevented data loss on crash/interruption:
   - `load_existing_results()`: Skips already-processed items by `no` field
   - `try-except-finally`: Ensures progress saved on `KeyboardInterrupt` or exceptions
   - Pre-assigned global `no` indices to prevent ID conflicts

4. **Edge Case Handling** - Production-grade reliability:
   - Atomic write: Write to `.json.tmp` then `replace()` to prevent file corruption
   - Defensive API response extraction: Checks for empty `choices` or `content` (Content Filter cases)
   - `tqdm.write()` instead of `print()`: Preserves progress bar rendering
   - Renamed `num_processes` → `max_workers` for semantic accuracy

### classify_cqia.py
Classifies CQIA dataset using LLM API. Similar to `classify_jokes.py` but processes CQIA format data with instruction/output pairs. Configuration in `classify_cqia_config.yaml`.

### classify_cqia_updated.py
CQIA 数据补全：为已分类的 240 条 CQIA 数据补充 `thought_process` 字段（导师蒸馏）。使用 Claude-Opus-4-6 对每条 `instruction` 生成深度分析的思考过程，合并到已有 `classification` 中。配置文件 `classify_cqia_updated_config.yaml`。

- System Prompt 对齐 `classify_config.yaml`（贴吧版），要求输出 `thought_process` + `top3_categories`
- 仅使用 `instruction` 字段作为 LLM 输入（不使用 CQIA 的 `output` 字段）
- 保留原有 `output`、`top3_categories` 不变，在 `classification` 中新增 `thought_process`
- 复用 `classify_jokes.py` 的鲁棒性优化：ThreadPoolExecutor、断点续传（JSONL checkpoint）、原子写入、多层 JSON 解析容错
- 新旧 `top3_categories` 对比记录日志（仅记录 category drift，不覆盖原有分类）

Input: `data/CQIA/ruozhiba_cqia_classified.json`, Output: `data/CQIA/ruozhiba_cqia_classified_v2.json`.

### extract_cqia_data.py
Extracts `instruction` and `output` fields from JSONL format CQIA dataset and saves to JSON format. Input: `data/CQIA/ruozhiba_ruozhiba.jsonl`, Output: `data/CQIA/ruozhiba_cqia_cleaned.json`.

### process_ruozhiba_past_annual.py
Processes raw ruozhiba annual post data. Extracts post number and text content from raw content field, adds metadata (l_num, ctime), and sorts by creation time. Input: `data/ruozhiba/data/ruozhiba-post-annual.json`, Output: `data/ruozhiba/data/ruozhiba-post-annual-processed.json`.

### filter_duplicates.py
Finds and removes duplicate content between ruozhiba and tieba datasets. Performs exact matching and fuzzy matching (threshold=0.5) using SequenceMatcher. Outputs filtered ruozhiba data and match records.

### extract_annual_data.py
Extracts annual post data from filtered ruozhiba dataset and saves to tieba format. Filters by creation time ranges:
- 2018 data: ctime <= "2019-01-01 13:37" → `best176_2018.json`
- 2019 data: "2019-12-15 23:00" <= ctime <= "2020-01-04 19:32" → `best336_2019.json`

Input: `data/ruozhiba/data/ruozhiba-post-annual-processed_filtered.json`, Output: `data/tieba/best176_2018.json`, `data/tieba/best336_2019.json`. Data source: https://github.com/Leymore/ruozhiba/blob/main/data/ruozhiba-post-annual.json

## Configuration Files

### classify_config.yaml
Configuration for `classify_jokes.py`. Contains:
- `system_prompt`: LLM classification prompt with 8 category definitions
- `files_to_process`: Input/output file mappings for tieba data
- `processing`: API parameters (num_processes, temperature, max_tokens, sleep_time)

### classify_cqia_config.yaml
Configuration for `classify_cqia.py`. Similar structure to `classify_config.yaml` but optimized for CQIA dataset processing with simplified output format.

### classify_cqia_updated_config.yaml
Configuration for `classify_cqia_updated.py`. Uses the same system prompt as `classify_config.yaml` (with `thought_process` + `top3_categories` output format). Processing parameters: `max_workers: 4`, `temperature: 0.3`, `max_tokens: 1500`.

## Dependencies
- openai: LLM API client
- tenacity: Retry mechanism
- tqdm: Progress bar
- PyYAML: Config parsing
- python-dotenv: Environment variables
