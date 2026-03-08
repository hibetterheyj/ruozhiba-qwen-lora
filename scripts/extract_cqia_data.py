import json
from typing import List, Dict


def extract_instruction_output(input_path: str, output_path: str) -> None:
    """
    Extract instruction and output fields from JSONL file and save to JSON file.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSON file
    """
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

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=2)

    print(f"Extracted {len(extracted_data)} records")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    input_file = '/Users/heyujie/Documents/cuhksz-all-sync/course_materials/CSS 5120 - Computational Linguistics/Lab3_SFT/data/CQIA/ruozhiba_ruozhiba.jsonl'
    output_file = '/Users/heyujie/Documents/cuhksz-all-sync/course_materials/CSS 5120 - Computational Linguistics/Lab3_SFT/data/CQIA/ruozhiba_cqia_cleaned.json'

    extract_instruction_output(input_file, output_file)
