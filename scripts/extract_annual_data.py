import json
from datetime import datetime
from typing import Dict, List, Tuple


def load_json(file_path: str) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_time(time_str: str) -> datetime:
    return datetime.strptime(time_str, "%Y-%m-%d %H:%M")


def extract_data(data: List[Dict], time_start: str, time_end: str) -> Tuple[List[Dict], List[int]]:
    start_dt = parse_time(time_start)
    end_dt = parse_time(time_end)
    
    filtered = []
    for item in data:
        if item.get('no') is None:
            continue
        ctime = item.get('ctime', '')
        try:
            item_dt = parse_time(ctime)
            if start_dt <= item_dt <= end_dt:
                filtered.append({
                    'no': item['no'],
                    'text': item['text']
                })
        except:
            continue
    
    filtered.sort(key=lambda x: x['no'])
    
    nos = [item['no'] for item in filtered]
    if not nos:
        return filtered, []
    
    min_no = min(nos)
    max_no = max(nos)
    expected = set(range(min_no, max_no + 1))
    actual = set(nos)
    missing = sorted(list(expected - actual))
    
    return filtered, missing


def main():
    input_path = '/Users/heyujie/Documents/cuhksz-all-sync/course_materials/CSS 5120 - Computational Linguistics/Lab3_SFT/data/ruozhiba/data/ruozhiba-post-annual-processed_filtered.json'
    output_dir = '/Users/heyujie/Documents/cuhksz-all-sync/course_materials/CSS 5120 - Computational Linguistics/Lab3_SFT/data/tieba/'
    
    data = load_json(input_path)
    
    print("2018 data:")
    data_2018, missing_2018 = extract_data(data, "2018-01-01 00:00", "2019-01-01 13:37")
    output_2018 = output_dir + 'best176_2018.json'
    save_json(data_2018, output_2018)
    print(f"- Matched {len(data_2018)} posts")
    print(f"- Missing numbers: {missing_2018}")
    print(f"- Saved to {output_2018}")
    
    print("\n2019 data:")
    data_2019, missing_2019 = extract_data(data, "2019-12-15 23:00", "2020-01-04 19:32")
    output_2019 = output_dir + 'best336_2019.json'
    save_json(data_2019, output_2019)
    print(f"- Matched {len(data_2019)} posts")
    print(f"- Missing numbers: {missing_2019}")
    print(f"- Saved to {output_2019}")


if __name__ == '__main__':
    main()
