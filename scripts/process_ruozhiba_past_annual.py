import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple


def extract_no_and_text(content: str) -> Tuple[int, str]:
    pattern = r'^(\d{1,3})[.、](.+)$'
    match = re.match(pattern, content.strip())
    if match:
        no = int(match.group(1))
        text = match.group(2).strip()
        return no, text
    return None, content.strip()


def process_ruozhiba_data(input_path: Path, output_path: Path) -> None:
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

    result.sort(key=lambda x: x.get('ctime', ''))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(result)} items")
    print(f"Output saved to: {output_path}")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    input_file = script_dir.parent / 'data' / 'ruozhiba' / 'data' / 'ruozhiba-post-annual.json'
    output_file = script_dir.parent / 'data' / 'ruozhiba' / 'data' / 'ruozhiba-post-annual-processed.json'

    process_ruozhiba_data(input_file, output_file)
