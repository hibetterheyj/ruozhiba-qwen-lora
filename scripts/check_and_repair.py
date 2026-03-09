import glob
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm


load_dotenv()


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_client() -> Tuple[OpenAI, str]:
    import os

    api_key = os.getenv("ZYAI_API_KEY")
    base_url = os.getenv("ZYAI_BASE_URL")
    model_id = os.getenv("ZYAI_MODEL_ID", "claude-opus-4-6")

    if not api_key or not base_url:
        raise ValueError("ZYAI_API_KEY and ZYAI_BASE_URL must be set in .env file")

    return OpenAI(api_key=api_key, base_url=base_url), model_id


def fix_double_escaped_quotes(s: str) -> str:
    """Fix double escaped quotes like \\" -> \"."""
    while '\\"' in s:
        s = s.replace('\\"', '"')
    return s


def fix_unescaped_quotes(s: str) -> str:
    """Fix unescaped quotes inside a JSON string."""
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
    """Try multiple methods to extract valid JSON from response."""
    if not content:
        return None

    content = content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        fixed = fix_double_escaped_quotes(content)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    try:
        fixed = fix_unescaped_quotes(content)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

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
                    fixed = fix_double_escaped_quotes(match)
                    return json.loads(fixed)
                except json.JSONDecodeError:
                    try:
                        fixed = fix_unescaped_quotes(match)
                        return json.loads(fixed)
                    except json.JSONDecodeError:
                        continue

    return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def classify_text(
    client: OpenAI, model_id: str, text: str, system_prompt: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    result = extract_json_from_response(content)
    if result:
        return result

    return {"error": f"JSON decode error: Failed to parse response", "raw_response": content}


def is_error_classification(classification: Dict[str, Any]) -> bool:
    if "error" not in classification:
        return False

    error_msg = classification["error"]
    error_patterns = [
        "JSON decode error",
        "API Request failed",
        "Request timed out",
        "APITimeoutError",
        "No JSON found"
    ]

    return any(pattern in error_msg for pattern in error_patterns)


def try_repair_from_raw_response(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to repair classification from existing raw_response."""
    classification = item.get("classification", {})
    raw_response = classification.get("raw_response")

    if not raw_response:
        return None

    result = extract_json_from_response(raw_response)
    if result and "thought_process" in result and "top3_categories" in result:
        return result

    return None


def check_file(file_path: Path) -> Tuple[int, int, List[Dict[str, Any]]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    total = len(data)
    error_items = []

    for item in data:
        classification = item.get("classification", {})
        if is_error_classification(classification):
            error_items.append(item)

    return total, len(error_items), error_items


def repair_items(
    items: List[Dict[str, Any]],
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    sleep_time: float,
    max_retries: int = 2
) -> List[Dict[str, Any]]:
    client, model_id = get_client()
    repaired = []

    for item in tqdm(items, desc="Repairing"):
        repaired_classification = try_repair_from_raw_response(item)

        if repaired_classification:
            repaired_item = {
                "no": item.get("no"),
                "text": item.get("text", ""),
                "score": item.get("score"),
                "classification": repaired_classification
            }
            repaired.append(repaired_item)
            continue

        for attempt in range(max_retries):
            try:
                classification = classify_text(
                    client, model_id, item.get("text", ""), system_prompt, temperature, max_tokens
                )

                if not is_error_classification(classification):
                    break

            except Exception as e:
                classification = {"error": f"API Request failed after retries: {str(e)}"}

            if attempt < max_retries - 1:
                time.sleep(sleep_time * 2)

        repaired_item = {
            "no": item.get("no"),
            "text": item.get("text", ""),
            "score": item.get("score"),
            "classification": classification
        }
        repaired.append(repaired_item)
        time.sleep(sleep_time)

    return repaired


def update_file(file_path: Path, repaired_items: List[Dict[str, Any]]) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    repaired_map = {item["no"]: item for item in repaired_items}

    for i, item in enumerate(data):
        no = item.get("no")
        if no in repaired_map:
            data[i] = repaired_map[no]

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_classified_files(data_dir: Path) -> List[str]:
    """Find all files ending with classified.json in the data directory."""
    pattern = str(data_dir / "*classified.json")
    files = glob.glob(pattern)
    return sorted([Path(f).name for f in files])


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "classify_config.yaml"
    data_dir = script_dir.parent / "data" / "tieba"

    config = load_config(config_path)

    system_prompt = config["system_prompt"]
    processing = config["processing"]

    files_to_check = get_classified_files(data_dir)

    print("=" * 60)
    print("Classification File Check Report")
    print("=" * 60)

    total_errors = 0

    for filename in files_to_check:
        file_path = data_dir / filename

        if not file_path.exists():
            print(f"\n[NOT FOUND] {filename}")
            continue

        total, error_count, error_items = check_file(file_path)

        print(f"\n[FILE] {filename}")
        print(f"  Total items: {total}")
        print(f"  Error items: {error_count}")

        if error_count > 0:
            total_errors += error_count
            error_types = {}
            for item in error_items:
                error_msg = item["classification"]["error"]
                error_type = error_msg.split(":")[0] if ":" in error_msg else error_msg[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1

            print("  Error breakdown:")
            for etype, count in error_types.items():
                print(f"    - {etype}: {count}")

    print("\n" + "=" * 60)
    print(f"Total errors across all files: {total_errors}")
    print("=" * 60)

    if total_errors > 0:
        print("\nStarting repair process...")
        print(f"Settings: temperature={processing['temperature']}, max_tokens={processing['max_tokens']}, sleep_time={processing['sleep_time']}")

        for filename in files_to_check:
            file_path = data_dir / filename

            if not file_path.exists():
                continue

            total, error_count, error_items = check_file(file_path)

            if error_count == 0:
                continue

            print(f"\n[REPAIRING] {filename} ({error_count} items)")

            repaired = repair_items(
                items=error_items,
                system_prompt=system_prompt,
                temperature=processing["temperature"],
                max_tokens=processing["max_tokens"],
                sleep_time=processing["sleep_time"]
            )

            update_file(file_path, repaired)
            print(f"  Saved {len(repaired)} repaired items")

        print("\n" + "=" * 60)
        print("Repair completed. Running verification...")

        for filename in files_to_check:
            file_path = data_dir / filename
            if file_path.exists():
                total, error_count, _ = check_file(file_path)
                print(f"  {filename}: {error_count} remaining errors")

    else:
        print("\nNo errors found. All classifications are valid.")


if __name__ == "__main__":
    main()
