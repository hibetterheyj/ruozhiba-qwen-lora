"""Adhoc inspector for a single item inside a classified JSON file."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO_ROOT / "data" / "tieba" / "best365_2025_classified.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting a file and item number."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="JSON file to inspect")
    parser.add_argument("--no", type=int, default=57, help="Value of the `no` field to inspect")
    return parser.parse_args()


def load_json(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSON array from disk with an actionable error on failure."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}. Use --file to point to an existing classified JSON file."
        )
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """Print a selected item and its raw response for manual debugging."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    data = load_json(args.file)

    for item in data:
        if item.get("no") == args.no:
            LOGGER.info("Found item no. %s:", args.no)
            LOGGER.info("%s", json.dumps(item, ensure_ascii=False, indent=2))

            classification = item.get("classification", {})
            raw_response = classification.get("raw_response")
            if raw_response:
                LOGGER.info("%s", "=" * 60)
                LOGGER.info("RAW RESPONSE:")
                LOGGER.info("%r", raw_response)
            return

    LOGGER.warning("Item with no=%s not found in %s", args.no, args.file)


if __name__ == "__main__":
    main()
