import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def classify_text(
    client: OpenAI, model_id: str, instruction: str, output: str, system_prompt: str, temperature: float, max_tokens: int
) -> Dict[str, Any]:
    user_content = f"【问题/言论】\n{instruction}\n\n【思考分析】\n{output}"

    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content

    if content.strip().startswith("```"):
        lines = content.strip().split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}", "raw_response": content}


def process_item(args: Tuple[Dict[str, Any], str, str, float, int, float, int]) -> Dict[str, Any]:
    item, model_id, system_prompt, temperature, max_tokens, sleep_time, index = args
    client, _ = get_client()

    instruction = item.get("instruction", "")
    output = item.get("output", "")

    classification = classify_text(client, model_id, instruction, output, system_prompt, temperature, max_tokens)

    result = {
        "instruction": instruction,
        "output": output,
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

    results: List[Dict[str, Any]] = []
    with Pool(processes=num_processes) as pool:
        for result in tqdm(
            pool.imap_unordered(process_item, args_list),
            total=len(args_list),
            desc="Classifying"
        ):
            results.append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Classification completed. Results saved to: {output_path}")


def main():
    script_dir = Path(__file__).parent
    config_path = script_dir / "classify_cqia_config.yaml"
    data_dir = Path(__file__).resolve().parents[2] / "data" / "CQIA"

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
