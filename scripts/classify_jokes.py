import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    if not response.choices or not response.choices[0].message.content:
        return {"error": "API returned empty content or triggered safety filters."}

    content = response.choices[0].message.content

    result = extract_json_from_response(content)
    if result:
        return result

    return {"error": "JSON decode error: Failed to parse response", "raw_response": content}


def process_item(
    args: Tuple[OpenAI, Dict[str, Any], str, str, float, int, float]
) -> Dict[str, Any]:
    client, item, model_id, system_prompt, temperature, max_tokens, sleep_time = args

    try:
        classification = classify_text(client, model_id, item.get("text", ""), system_prompt, temperature, max_tokens)
    except Exception as e:
        classification = {"error": f"API Request failed after retries: {str(e)}"}

    result = {
        **item,
        "no": item.get("no"),
        "text": item.get("text", ""),
        "classification": classification
    }

    time.sleep(sleep_time)

    return result


def load_existing_results(output_path: Path) -> Dict[int, Dict[str, Any]]:
    if not output_path.exists():
        return {}

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            existing_data: List[Dict[str, Any]] = json.load(f)
        return {item.get("no"): item for item in existing_data if "no" in item}
    except (json.JSONDecodeError, Exception):
        return {}


def process_file(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    max_workers: int,
    temperature: float,
    max_tokens: int,
    sleep_time: float
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for i, item in enumerate(data):
        if "no" not in item:
            item["no"] = i

    existing_results = load_existing_results(output_path)
    processed_nos = set(existing_results.keys())

    items_to_process = [
        item for item in data
        if item.get("no") not in processed_nos
    ]

    total_items = len(data)
    already_processed = len(processed_nos)
    remaining = len(items_to_process)

    print(f"  Total: {total_items}, Already processed: {already_processed}, Remaining: {remaining}")

    if remaining == 0:
        print(f"  All items already processed. Skipping.")
        return

    client, model_id = get_client()

    args_list = [
        (client, item, model_id, system_prompt, temperature, max_tokens, sleep_time)
        for item in items_to_process
    ]

    results: List[Dict[str, Any]] = []

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_item, args): args for args in args_list}

            for future in tqdm(as_completed(futures), total=len(args_list), desc="Classifying"):
                results.append(future.result())
    except KeyboardInterrupt:
        tqdm.write("\n  [Warn] Task interrupted by user! Saving progress...")
    except Exception as e:
        tqdm.write(f"\n  [Error] Unexpected error: {e}. Saving progress...")
    finally:
        if results:
            all_results = list(existing_results.values()) + results
            all_results.sort(key=lambda x: x.get("no", 0))

            output_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_output_path = output_path.with_suffix('.json.tmp')

            with open(tmp_output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)

            tmp_output_path.replace(output_path)

            tqdm.write(f"  Progress saved. Total completed: {len(all_results)}")


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "classify_config.yaml"
    data_dir = script_dir.parent / "data" / "tieba"

    config = load_config(config_path)

    system_prompt = config["system_prompt"]
    files_to_process = config["files_to_process"]
    processing = config["processing"]

    for file_pair in files_to_process:
        input_path = data_dir / file_pair["input"]
        output_path = data_dir / file_pair["output"]

        if input_path.exists():
            print(f"\nProcessing: {file_pair['input']}")
            process_file(
                input_path=input_path,
                output_path=output_path,
                system_prompt=system_prompt,
                max_workers=processing["num_processes"],
                temperature=processing["temperature"],
                max_tokens=processing["max_tokens"],
                sleep_time=processing["sleep_time"]
            )
        else:
            print(f"File not found: {input_path}")


if __name__ == "__main__":
    main()
