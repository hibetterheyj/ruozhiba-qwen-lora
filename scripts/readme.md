# Scripts Documentation

## Python Scripts

### classify_jokes.py
Classifies tieba joke data using LLM API. Reads input files specified in `classify_config.yaml`, sends text to LLM for multi-label classification (8 categories including 古典弱智, 奇怪提问, 弱智科学家, etc.), and outputs classification results with confidence scores.

### classify_cqia.py
Classifies CQIA dataset using LLM API. Similar to `classify_jokes.py` but processes CQIA format data with instruction/output pairs. Configuration in `classify_cqia_config.yaml`.

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

## Dependencies
- openai: LLM API client
- tenacity: Retry mechanism
- tqdm: Progress bar
- PyYAML: Config parsing
- python-dotenv: Environment variables
