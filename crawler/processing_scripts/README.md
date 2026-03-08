# Processing Scripts

Scripts for extracting and processing Tieba (贴吧) thread data.

## Files

### `extract_posts.py`

Extract posts from 2025 annual best 365 posts thread.

**Input:** `crawler/threads/10354221105_弱智吧2025年度365佳贴/extracted.json`

**Output:** `data/tieba/best365_2025.json`

**Usage:**
```bash
python extract_posts.py
```

### `extract_all_years.py`

Batch extract posts from multiple years (2023, 2024).

**Input:** `crawler/threads/{thread_id}_弱智吧{year}年度365佳贴/{thread_id}_dump.jsonl`

**Output:** `data/tieba/best365_{year}.json`

**Usage:**
```bash
python extract_all_years.py
```

### `check_missing.py`

Check for missing post numbers in the raw dump file.

**Input:** `crawler/threads/10354221105_弱智吧2025年度365佳贴/10354221105_dump.jsonl`

**Usage:**
```bash
python check_missing.py
```

## Output Format

All extraction scripts produce JSON files with the following structure:

```json
[
  {
    "no": 1,
    "text": "Post content here...",
    "score": 4.31
  },
  ...
]
```

| Field | Description |
|-------|-------------|
| `no` | Post ranking number (1-365) |
| `text` | Post content |
| `score` | Average rating score |

## Post Format Pattern

The scripts match posts with the following pattern:

```
{number}.{content}
评分：{score}
```

Example:
```
1."太好了还有气！"王警官看着尸体手里的可乐高兴道。
评分：4.31
```
