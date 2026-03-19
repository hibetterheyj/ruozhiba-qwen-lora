import json
from pathlib import Path
from typing import Any

ASCII_QUOTE = chr(34)
CHINESE_LEFT_QUOTE = chr(8220)
CHINESE_RIGHT_QUOTE = chr(8221)

def convert_ascii_quotes_to_chinese(text: str) -> str:
    if not text or ASCII_QUOTE not in text:
        return text
    
    result = []
    in_quote = False
    
    for char in text:
        if char == ASCII_QUOTE:
            if not in_quote:
                result.append(CHINESE_LEFT_QUOTE)
                in_quote = True
            else:
                result.append(CHINESE_RIGHT_QUOTE)
                in_quote = False
        else:
            result.append(char)
    
    return ''.join(result)


def process_value(value: Any) -> Any:
    if isinstance(value, str):
        return convert_ascii_quotes_to_chinese(value)
    elif isinstance(value, dict):
        return {k: process_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [process_value(item) for item in value]
    else:
        return value


def count_ascii_quotes_in_strings(value: Any, fields_to_check: list = None) -> int:
    count = 0
    if fields_to_check is None:
        fields_to_check = ['thought_process', 'reason']
    
    if isinstance(value, dict):
        for k, v in value.items():
            if k in fields_to_check and isinstance(v, str):
                count += v.count(ASCII_QUOTE)
            else:
                count += count_ascii_quotes_in_strings(v, fields_to_check)
    elif isinstance(value, list):
        for item in value:
            count += count_ascii_quotes_in_strings(item, fields_to_check)
    
    return count


def process_file(file_path: Path) -> dict:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = count_ascii_quotes_in_strings(data)
    
    processed_data = process_value(data)
    
    processed_count = count_ascii_quotes_in_strings(processed_data)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    return {
        'original_ascii': original_count,
        'processed_ascii': processed_count,
        'converted': original_count - processed_count
    }


def main():
    script_dir = Path(__file__).parent
    
    repo_root = Path(__file__).resolve().parents[2]
    tieba_dir = repo_root / 'data' / 'tieba'
    cqia_dir = repo_root / 'data' / 'CQIA'
    
    files_to_process = [
        tieba_dir / 'best365_2025_classified.json',
        tieba_dir / 'best365_2024_classified.json',
        tieba_dir / 'best365_2023_classified.json',
        tieba_dir / 'best365_2022_classified.json',
        cqia_dir / 'ruozhiba_cqia_classified.json',
    ]
    
    print('=' * 60)
    print('ASCII Quote to Chinese Quote Conversion')
    print('=' * 60)
    
    total_converted = 0
    
    for file_path in files_to_process:
        if not file_path.exists():
            print(f'\n[NOT FOUND] {file_path.name}')
            continue
        
        print(f'\n[PROCESSING] {file_path.name}')
        stats = process_file(file_path)
        print(f'  ASCII quotes in content before: {stats["original_ascii"]}')
        print(f'  ASCII quotes in content after: {stats["processed_ascii"]}')
        print(f'  Converted to Chinese quotes: {stats["converted"]}')
        total_converted += stats['converted']
    
    print('\n' + '=' * 60)
    print(f'Total ASCII quotes converted: {total_converted}')
    print('=' * 60)


if __name__ == '__main__':
    main()
