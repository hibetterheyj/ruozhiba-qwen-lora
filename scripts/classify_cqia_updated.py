"""CQIA 数据补全：为已分类的 CQIA 数据补充 thought_process 字段。

读取 ruozhiba_cqia_classified.json（240 条，已有 top3_categories），
使用 LLM 对每条 instruction 生成 thought_process，合并到 classification 中。

复用 classify_jokes.py 的鲁棒性优化：
- ThreadPoolExecutor 并发控制
- 断点续传（JSONL checkpoint）
- 原子写入（.tmp → rename）
- 多层 JSON 解析容错

Usage:
    python classify_cqia_updated.py
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_client() -> Tuple[OpenAI, str]:
    api_key = os.getenv("ZYAI_API_KEY")
    base_url = os.getenv("ZYAI_BASE_URL")
    model_id = os.getenv("ZYAI_MODEL_ID", "claude-opus-4-6")

    if not api_key or not base_url:
        raise ValueError("ZYAI_API_KEY and ZYAI_BASE_URL must be set in .env file")

    return OpenAI(api_key=api_key, base_url=base_url), model_id


# ---------------------------------------------------------------------------
# JSON Parsing Robustness (reused from classify_jokes.py)
# ---------------------------------------------------------------------------

def fix_double_escaped_quotes(s: str) -> str:
    while '\\"' in s:
        s = s.replace('\\"', '"')
    return s


def fix_unescaped_quotes(s: str) -> str:
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(s):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            result.append(char)
            continue

        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                if i + 1 < len(s) and s[i + 1] in [",", "}", "]", ":", " ", "\n", "\t"]:
                    in_string = False
                    result.append(char)
                else:
                    result.append('\\"')
        else:
            result.append(char)

    return "".join(result)


def extract_json_from_response(content: str) -> Optional[Dict[str, Any]]:
    if not content:
        return None

    content = content.strip()

    # Strip markdown code fences
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # Direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Fix double-escaped quotes
    try:
        fixed = fix_double_escaped_quotes(content)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Fix unescaped quotes
    try:
        fixed = fix_unescaped_quotes(content)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Regex fallback
    json_patterns = [
        r'\{[\s\S]*"thought_process"[\s\S]*"top3_categories"[\s\S]*\}',
        r'\{[\s\S]*\}',
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, content)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                try:
                    return json.loads(fix_double_escaped_quotes(match))
                except json.JSONDecodeError:
                    try:
                        return json.loads(fix_unescaped_quotes(match))
                    except json.JSONDecodeError:
                        continue

    return None


# ---------------------------------------------------------------------------
# LLM API Call
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=60))
def classify_text(
    client: OpenAI,
    model_id: str,
    instruction: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """Send instruction to LLM and get thought_process + top3_categories."""
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    if not response.choices or not response.choices[0].message.content:
        return {"error": "API returned empty content or triggered safety filters."}

    content = response.choices[0].message.content
    result = extract_json_from_response(content)
    if result:
        return result

    return {"error": "JSON decode error: Failed to parse response", "raw_response": content}


# ---------------------------------------------------------------------------
# Category Consistency Check
# ---------------------------------------------------------------------------

def compare_categories(
    old_categories: List[Dict[str, Any]],
    new_categories: List[Dict[str, Any]],
    index: int,
    instruction: str,
) -> None:
    """Log differences between old and new top3_categories (informational only)."""
    old_top1 = old_categories[0]["category"] if old_categories else "N/A"
    new_top1 = new_categories[0]["category"] if new_categories else "N/A"

    if old_top1 != new_top1:
        logger.info(
            "Category drift [#%d]: Top-1 changed '%s' -> '%s' | text: %.40s...",
            index, old_top1, new_top1, instruction,
        )


# ---------------------------------------------------------------------------
# Per-Item Processing
# ---------------------------------------------------------------------------

def process_item(
    args: Tuple[OpenAI, Dict[str, Any], int, str, str, float, int, float],
) -> Dict[str, Any]:
    """Process a single CQIA item: call LLM, merge thought_process into classification."""
    client, item, index, model_id, system_prompt, temperature, max_tokens, sleep_time = args

    instruction = item.get("instruction", "")

    try:
        new_classification = classify_text(
            client, model_id, instruction, system_prompt, temperature, max_tokens,
        )
    except Exception as e:
        new_classification = {"error": f"API Request failed after retries: {str(e)}"}

    # Build result: preserve original fields, merge thought_process
    existing_classification = item.get("classification", {})
    old_categories = existing_classification.get("top3_categories", [])

    if "error" not in new_classification:
        # Compare old vs new categories (log only, no overwrite)
        new_categories = new_classification.get("top3_categories", [])
        compare_categories(old_categories, new_categories, index, instruction)

        # Merge: add thought_process to existing classification
        merged_classification = {
            **existing_classification,
            "thought_process": new_classification.get("thought_process", ""),
        }
    else:
        # On error, preserve original classification, mark thought_process as failed
        merged_classification = {
            **existing_classification,
            "thought_process": None,
            "_error": new_classification,
        }
        tqdm.write(f"  [Error] Item #{index}: {new_classification.get('error', 'unknown')}")

    result = {
        "index": index,
        "instruction": instruction,
        "output": item.get("output", ""),
        "classification": merged_classification,
    }

    time.sleep(sleep_time)
    return result


# ---------------------------------------------------------------------------
# Checkpoint & Resume
# ---------------------------------------------------------------------------

def load_checkpoint(output_path: Path) -> Dict[int, Dict[str, Any]]:
    """Load existing results from JSONL checkpoint file."""
    jsonl_path = output_path.with_suffix(".jsonl")
    if not jsonl_path.exists():
        return {}

    existing: Dict[int, Dict[str, Any]] = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if "index" in item:
                        existing[item["index"]] = item
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("Failed to load checkpoint: %s", e)

    return existing


def save_final_output(all_results: List[Dict[str, Any]], output_path: Path) -> None:
    """Atomic write: write to .tmp then rename to prevent corruption."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove internal 'index' field from final output
    clean_results = []
    for r in all_results:
        clean = {k: v for k, v in r.items() if k != "index"}
        # Also remove _error marker from classification if present
        if "classification" in clean and "_error" in clean["classification"]:
            clean["classification"] = {
                k: v for k, v in clean["classification"].items() if k != "_error"
            }
        clean_results.append(clean)

    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, ensure_ascii=False, indent=2)
    tmp_path.replace(output_path)


# ---------------------------------------------------------------------------
# Main Processing
# ---------------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    max_workers: int,
    temperature: float,
    max_tokens: int,
    sleep_time: float,
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    # Assign stable indices
    for i, item in enumerate(data):
        item["_index"] = i

    # Load checkpoint
    existing_results = load_checkpoint(output_path)
    processed_indices = set(existing_results.keys())

    items_to_process = [item for item in data if item["_index"] not in processed_indices]

    total_items = len(data)
    already_processed = len(processed_indices)
    remaining = len(items_to_process)

    logger.info("Total: %d, Already processed: %d, Remaining: %d", total_items, already_processed, remaining)

    if remaining == 0:
        logger.info("All items already processed. Generating final output...")
        all_results = sorted(existing_results.values(), key=lambda x: x.get("index", 0))
        save_final_output(all_results, output_path)
        logger.info("Final output saved to: %s", output_path)
        return

    client, model_id = get_client()

    args_list = [
        (client, item, item["_index"], model_id, system_prompt, temperature, max_tokens, sleep_time)
        for item in items_to_process
    ]

    results: List[Dict[str, Any]] = []
    jsonl_path = output_path.with_suffix(".jsonl")

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, args): args for args in args_list}

            for future in tqdm(as_completed(futures), total=len(args_list), desc="Processing CQIA"):
                result = future.result()
                results.append(result)
                # Append to JSONL checkpoint immediately
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except KeyboardInterrupt:
        tqdm.write("\n  [Warn] Interrupted by user! Saving progress...")
    except Exception as e:
        tqdm.write(f"\n  [Error] Unexpected error: {e}. Saving progress...")
    finally:
        all_results = list(existing_results.values()) + results
        all_results.sort(key=lambda x: x.get("index", 0))

        save_final_output(all_results, output_path)

        # Count errors
        error_count = sum(
            1 for r in all_results
            if r.get("classification", {}).get("thought_process") is None
        )
        success_count = len(all_results) - error_count

        logger.info("Progress saved. Total: %d (success: %d, errors: %d)", len(all_results), success_count, error_count)
        logger.info("JSONL checkpoint: %s", jsonl_path)
        logger.info("JSON output: %s", output_path)


def main() -> None:
    script_dir = Path(__file__).parent
    config_path = script_dir / "classify_cqia_updated_config.yaml"
    data_dir = script_dir.parent / "data" / "CQIA"

    config = load_config(config_path)

    system_prompt = config["system_prompt"]
    files_to_process = config["files_to_process"]
    processing = config["processing"]

    for file_pair in files_to_process:
        input_path = data_dir / file_pair["input"]
        output_path = data_dir / file_pair["output"]

        if input_path.exists():
            logger.info("Processing: %s -> %s", file_pair["input"], file_pair["output"])
            process_file(
                input_path=input_path,
                output_path=output_path,
                system_prompt=system_prompt,
                max_workers=processing["max_workers"],
                temperature=processing["temperature"],
                max_tokens=processing["max_tokens"],
                sleep_time=processing["sleep_time"],
            )
        else:
            logger.error("File not found: %s", input_path)


if __name__ == "__main__":
    main()
