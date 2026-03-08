import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI
from dotenv import load_dotenv


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


def classify_text(
    client: OpenAI, model_id: str, text: str, system_prompt: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )

        content = response.choices[0].message.content

        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            return json.loads(json_str)
        else:
            return {"error": "No JSON found in response", "raw_response": content}

    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}", "raw_response": content if 'content' in dir() else None}
    except Exception as e:
        return {"error": str(e)}


def process_item(args: Tuple[Dict[str, Any], str, str, float, int, float, int]) -> Dict[str, Any]:
    item, model_id, system_prompt, temperature, max_tokens, sleep_time, index = args
    client, _ = get_client()

    print(f"Processing item {index + 1}: {item['text'][:30]}...")

    classification = classify_text(client, model_id, item["text"], system_prompt, temperature, max_tokens)

    result = {
        "no": item["no"],
        "text": item["text"],
        "score": item["score"],
        "classification": classification
    }

    time.sleep(sleep_time)

    return result


def process_file(
    input_path: Path,
    output_path: Path,
    system_prompt: str,
    num_processes: int,
    temperature: float,
    max_tokens: int,
    sleep_time: float
) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    _, model_id = get_client()

    args_list = [
        (item, model_id, system_prompt, temperature, max_tokens, sleep_time, i)
        for i, item in enumerate(data)
    ]

    with Pool(processes=num_processes) as pool:
        results: List[Dict[str, Any]] = pool.map(process_item, args_list)

    results.sort(key=lambda x: x["no"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Classification completed. Results saved to: {output_path}")


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
                num_processes=processing["num_processes"],
                temperature=processing["temperature"],
                max_tokens=processing["max_tokens"],
                sleep_time=processing["sleep_time"]
            )
        else:
            print(f"File not found: {input_path}")


if __name__ == "__main__":
    main()
