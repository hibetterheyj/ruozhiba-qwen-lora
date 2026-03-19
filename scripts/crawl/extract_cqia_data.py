import json
from pathlib import Path
from typing import List, Dict


def extract_instruction_output(input_path: Path, output_path: Path) -> None:
    extracted_data: List[Dict[str, str]] = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            if 'instruction' in data and 'output' in data:
                extracted_data.append({
                    'instruction': data['instruction'],
                    'output': data['output']
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(extracted_data)} records")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    repo_root = Path(__file__).resolve().parents[2]
    input_file = repo_root / 'data' / 'CQIA' / 'ruozhiba_ruozhiba.jsonl'
    output_file = repo_root / 'data' / 'CQIA' / 'ruozhiba_cqia_cleaned.json'

    extract_instruction_output(input_file, output_file)
