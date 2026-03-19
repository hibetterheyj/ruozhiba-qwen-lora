import json
import logging
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def fix_double_escaped_quotes_in_file(file_path: Path) -> None:
    """Fix double escaped quotes in a JSON file."""
    LOGGER.info("Processing: %s", file_path.name)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fixed_count = 0

    for item in data:
        classification = item.get("classification", {})

        if "thought_process" in classification:
            thought = classification["thought_process"]
            if '\\"' in thought:
                original = thought
                classification["thought_process"] = original.replace('\\"', '"')
                fixed_count += 1

        if "top3_categories" in classification:
            for cat in classification["top3_categories"]:
                if "reason" in cat:
                    reason = cat["reason"]
                    if '\\"' in reason:
                        original = reason
                        cat["reason"] = original.replace('\\"', '"')
                        fixed_count += 1
                if "category" in cat:
                    category = cat["category"]
                    if '\\"' in category:
                        original = category
                        cat["category"] = original.replace('\\"', '"')
                        fixed_count += 1

    if fixed_count > 0:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        LOGGER.info("  Fixed %s double-escaped quotes", fixed_count)
    else:
        LOGGER.info("  No double-escaped quotes found")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_dir = Path(__file__).resolve().parents[2] / "data" / "tieba"

    files = [
        "best365_2025_classified.json",
        "best365_2024_classified.json",
        "best365_2023_classified.json",
        "best365_2022_classified.json",
        "best295_2021_1H_classified.json",
    ]

    for filename in files:
        file_path = data_dir / filename
        if file_path.exists():
            fix_double_escaped_quotes_in_file(file_path)


if __name__ == "__main__":
    main()
