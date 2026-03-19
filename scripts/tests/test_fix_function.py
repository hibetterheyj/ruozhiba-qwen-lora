"""Adhoc debug script for testing raw-response JSON repair helpers."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO_ROOT / "data" / "tieba" / "best365_2025_classified.json"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for debug file and sample selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE, help="JSON file to inspect")
    parser.add_argument("--no", type=int, default=57, help="Value of the `no` field to inspect")
    return parser.parse_args()


def load_json(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSON array from disk with a clear error if missing."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}. Use --file to point to an existing classified JSON file."
        )
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def fix_unescaped_quotes(s: str) -> str:
    """Fix unescaped quotes inside a JSON string."""
    result = []
    in_string = False
    escape_next = False

    for i, char in enumerate(s):
        if escape_next:
            result.append(char)
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            result.append(char)
            continue

        if char == '"':
            if not in_string:
                in_string = True
                result.append(char)
            else:
                if i + 1 < len(s) and s[i + 1] in [",", "}", "]", ":", " ", "\n", "\t"]:
                    in_string = False
                    result.append(char)
                else:
                    result.append("\\\"")
        else:
            result.append(char)

    return "".join(result)


def extract_json_from_response(content: str) -> dict[str, Any] | None:
    """Try multiple methods to extract valid JSON from response."""
    if not content:
        return None

    content = content.strip()

    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    try:
        fixed = fix_unescaped_quotes(content)
        LOGGER.info("After fix_unescaped_quotes: %r", fixed[:300])
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        LOGGER.info("Still failed: %s", e)

    return None

def main() -> None:
    """Attempt to repair one selected raw response and print the parsed object."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    data = load_json(args.file)

    for item in data:
        if item.get("no") == args.no:
            LOGGER.info("Found item no. %s", args.no)
            classification = item.get("classification", {})
            raw_response = classification.get("raw_response")

            if raw_response:
                LOGGER.info("%s", "=" * 60)
                LOGGER.info("Attempting to parse raw_response")
                result = extract_json_from_response(raw_response)

                if result:
                    LOGGER.info("SUCCESS")
                    LOGGER.info("%s", json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    LOGGER.info("FAILED")
            return

    LOGGER.warning("Item with no=%s not found in %s", args.no, args.file)


if __name__ == "__main__":
    main()
