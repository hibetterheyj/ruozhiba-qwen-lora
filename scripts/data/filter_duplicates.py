import json
from pathlib import Path
from typing import Dict, List, Tuple, Set
from difflib import SequenceMatcher


def load_json(file_path: Path) -> List[Dict]:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: List[Dict], file_path: Path) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def remove_special_chars(text: str) -> str:
    return text.replace('｜', '').replace('|', '')


def calculate_similarity(text1: str, text2: str) -> float:
    cleaned1 = remove_special_chars(text1)
    cleaned2 = remove_special_chars(text2)
    return SequenceMatcher(None, cleaned1, cleaned2).ratio()


def find_exact_matches(
    ruozhiba_data: List[Dict],
    tieba_files: List[Tuple[Path, List[Dict]]]
) -> Tuple[Set[str], List[Dict]]:
    ruozhiba_texts = {item['text'] for item in ruozhiba_data}

    matched_texts: Set[str] = set()
    exact_matched_records: List[Dict] = []

    for file_path, tieba_data in tieba_files:
        file_name = file_path.name
        for item in tieba_data:
            if item['text'] in ruozhiba_texts:
                matched_texts.add(item['text'])
                exact_matched_records.append({
                    'tieba_file': file_name,
                    'tieba_text': item['text'],
                    'ruozhiba_text': item['text'],
                    'similarity': 1.0
                })

    return matched_texts, exact_matched_records


def find_fuzzy_matches(
    ruozhiba_data: List[Dict],
    tieba_files: List[Tuple[Path, List[Dict]]],
    threshold: float = 0.5
) -> List[Dict]:
    fuzzy_matched_records: List[Dict] = []

    for file_path, tieba_data in tieba_files:
        file_name = file_path.name
        for tieba_item in tieba_data:
            for ruozhiba_item in ruozhiba_data:
                similarity = calculate_similarity(tieba_item['text'], ruozhiba_item['text'])

                if similarity >= threshold and similarity < 1.0:
                    fuzzy_matched_records.append({
                        'tieba_file': file_name,
                        'tieba_text': tieba_item['text'],
                        'ruozhiba_text': ruozhiba_item['text'],
                        'similarity': round(similarity, 4)
                    })

    return fuzzy_matched_records


def main():
    base_dir = Path(__file__).resolve().parents[2]

    ruozhiba_path = base_dir / 'data' / 'ruozhiba' / 'data' / 'ruozhiba-post-annual-processed.json'

    tieba_paths = [
        base_dir / 'data' / 'tieba' / 'best365_2020.json',
        base_dir / 'data' / 'tieba' / 'best295_2021_1H.json',
        base_dir / 'data' / 'tieba' / 'best306_2021_2H.json'
    ]

    print("Loading ruozhiba data...")
    ruozhiba_data = load_json(ruozhiba_path)
    print(f"Loaded {len(ruozhiba_data)} records from ruozhiba")

    tieba_files: List[Tuple[Path, List[Dict]]] = []
    for path in tieba_paths:
        print(f"Loading {path.name}...")
        data = load_json(path)
        tieba_files.append((path, data))
        print(f"Loaded {len(data)} records")

    print("\nFinding exact matches...")
    matched_texts, exact_matched_records = find_exact_matches(ruozhiba_data, tieba_files)
    print(f"Found {len(matched_texts)} unique matched texts")
    print(f"Total exact match records: {len(exact_matched_records)}")

    print("\nFinding fuzzy matches (threshold=0.5)...")
    fuzzy_matched_records = find_fuzzy_matches(ruozhiba_data, tieba_files, threshold=0.5)
    print(f"Found {len(fuzzy_matched_records)} fuzzy matches")

    print("\nGenerating output files...")

    output_dir = ruozhiba_path.parent
    base_name = ruozhiba_path.stem

    filtered_data = [item for item in ruozhiba_data if item['text'] not in matched_texts]
    filtered_path = output_dir / f"{base_name}_filtered.json"
    save_json(filtered_data, filtered_path)
    print(f"Saved {len(filtered_data)} records to {filtered_path}")

    exact_matched_path = output_dir / f"{base_name}_exact_matched.json"
    save_json(exact_matched_records, exact_matched_path)
    print(f"Saved {len(exact_matched_records)} exact matches to {exact_matched_path}")

    fuzzy_matched_path = output_dir / f"{base_name}_fuzzy_matched.json"
    save_json(fuzzy_matched_records, fuzzy_matched_path)
    print(f"Saved {len(fuzzy_matched_records)} fuzzy matches to {fuzzy_matched_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
