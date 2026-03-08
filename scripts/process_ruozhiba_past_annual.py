"""
Process ruozhiba-post-annual.json to extract no and text fields from content.
"""

import json
import re
from typing import Dict, List, Any


def extract_no_and_text(content: str) -> tuple:
    """
    Extract number and text from content string.
    Pattern: 1-3 digits followed by '.' or '、'
    """
    pattern = r'^(\d{1,3})[.、](.+)$'
    match = re.match(pattern, content.strip())
    if match:
        no = int(match.group(1))
        text = match.group(2).strip()
        return no, text
    return None, content.strip()


def process_ruozhiba_data(input_path: str, output_path: str) -> None:
    """
    Process ruozhiba data file and extract structured fields.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data: List[Dict[str, Any]] = json.load(f)

    result: List[Dict[str, Any]] = []

    for item in data:
        content = item.get('content', '')
        no, text = extract_no_and_text(content)

        new_item = {
            'no': no,
            'text': text,
            'l_num': item.get('l_num'),
            'ctime': item.get('ctime')
        }
        result.append(new_item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(result)} items")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    input_file = '../data/ruozhiba/data/ruozhiba-post-annual.json'
    output_file = '../data/ruozhiba/data/ruozhiba-post-annual-processed.json'

    process_ruozhiba_data(input_file, output_file)
