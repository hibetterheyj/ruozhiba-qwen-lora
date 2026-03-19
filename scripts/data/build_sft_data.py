#!/usr/bin/env python3
"""Convert dedup tieba classified data to LLaMA-Factory ShareGPT format.

Reads Phase 1.2 dedup output (data/tieba/best*_classified_dedup.json) and produces
two ShareGPT datasets for LLaMA-Factory SFT:
  - LLaMA-Factory/data/ruozhiba_last3.json  (2023-2025, ~1025 entries)
  - LLaMA-Factory/data/ruozhiba_all.json    (2018-2025, ~2786 entries)

Usage (from repo root):
    python scripts/data/build_sft_data.py
    python scripts/data/build_sft_data.py --dry-run   # preview counts without writing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIEBA_DATA_DIR = PROJECT_ROOT / "data" / "tieba"
PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts.yaml"
OUTPUT_DIR = PROJECT_ROOT / "LLaMA-Factory" / "data"

LAST3_OUTPUT = OUTPUT_DIR / "ruozhiba_last3.json"
ALL_OUTPUT = OUTPUT_DIR / "ruozhiba_all.json"

LAST3_YEARS = {2023, 2024, 2025}

# Filename pattern: best<N>_<YYYY>[_suffix]_classified_dedup.json
YEAR_RE = re.compile(r"best\d+_(\d{4})")


def load_system_prompt(config_path: Path) -> str:
    """Load centralized system prompt from configs/prompts.yaml."""
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prompt = cfg.get("system_prompt", "")
    if not prompt:
        raise ValueError(f"system_prompt is empty in {config_path}")
    return prompt.rstrip("\n")


def extract_year(filename: str) -> int:
    """Extract 4-digit year from a dedup filename."""
    m = YEAR_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot extract year from filename: {filename}")
    return int(m.group(1))


def to_sharegpt(
    item: dict[str, Any],
    system_prompt: str,
) -> dict[str, list[dict[str, str]]]:
    """Convert a single classified tieba entry to ShareGPT conversation format.

    Field mapping:
        system  ← centralized system_prompt (configs/prompts.yaml)
        human   ← item["text"]
        gpt     ← json.dumps(item["classification"], ensure_ascii=False)
    """
    classification = item["classification"]
    gpt_value = json.dumps(classification, ensure_ascii=False, indent=None)

    return {
        "conversations": [
            {"from": "system", "value": system_prompt},
            {"from": "human", "value": item["text"]},
            {"from": "gpt", "value": gpt_value},
        ]
    }


def load_dedup_files(
    data_dir: Path,
) -> list[tuple[str, int, list[dict[str, Any]]]]:
    """Load all *_classified_dedup.json files, returning (filename, year, data)."""
    results: list[tuple[str, int, list[dict[str, Any]]]] = []
    dedup_files = sorted(
        f for f in os.listdir(data_dir) if f.endswith("_classified_dedup.json")
    )
    if not dedup_files:
        raise FileNotFoundError(
            f"No *_classified_dedup.json files found in {data_dir}"
        )

    for fname in dedup_files:
        year = extract_year(fname)
        filepath = data_dir / fname
        with open(filepath, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded %s: %d entries (year=%d)", fname, len(data), year)
        results.append((fname, year, data))

    return results


def validate_entry(item: dict[str, Any], idx: int, filename: str) -> bool:
    """Validate a single entry has required fields."""
    if "text" not in item:
        logger.warning("Missing 'text' in %s entry #%d, skipping", filename, idx)
        return False
    if "classification" not in item:
        logger.warning(
            "Missing 'classification' in %s entry #%d, skipping", filename, idx
        )
        return False
    cls = item["classification"]
    if "thought_process" not in cls:
        logger.warning(
            "Missing 'thought_process' in %s entry #%d, skipping", filename, idx
        )
        return False
    if "top3_categories" not in cls:
        logger.warning(
            "Missing 'top3_categories' in %s entry #%d, skipping", filename, idx
        )
        return False
    return True


def build_datasets(
    data_dir: Path, system_prompt: str
) -> tuple[list[dict], list[dict]]:
    """Build last3 and all ShareGPT datasets from dedup files."""
    all_items: list[dict] = []
    last3_items: list[dict] = []
    skipped = 0

    file_data = load_dedup_files(data_dir)

    for fname, year, entries in file_data:
        for idx, item in enumerate(entries):
            if not validate_entry(item, idx, fname):
                skipped += 1
                continue
            conv = to_sharegpt(item, system_prompt)
            all_items.append(conv)
            if year in LAST3_YEARS:
                last3_items.append(conv)

    if skipped > 0:
        logger.warning("Skipped %d entries with missing fields", skipped)

    return last3_items, all_items


def write_output(data: list[dict], output_path: Path, dry_run: bool = False) -> None:
    """Write ShareGPT JSON to output path."""
    if dry_run:
        logger.info("[DRY-RUN] Would write %d conversations to %s", len(data), output_path)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info("Wrote %d conversations to %s", len(data), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ShareGPT SFT datasets from dedup tieba data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview counts without writing output files",
    )
    args = parser.parse_args()

    # Load centralized prompt
    logger.info("Loading system prompt from %s", PROMPT_CONFIG)
    system_prompt = load_system_prompt(PROMPT_CONFIG)
    logger.info("System prompt loaded (%d chars)", len(system_prompt))

    # Build datasets
    last3, all_data = build_datasets(TIEBA_DATA_DIR, system_prompt)

    logger.info("--- Dataset Summary ---")
    logger.info("ruozhiba_all:   %d conversations (2018-2025)", len(all_data))
    logger.info("ruozhiba_last3: %d conversations (2023-2025)", len(last3))

    # Write output
    write_output(all_data, ALL_OUTPUT, dry_run=args.dry_run)
    write_output(last3, LAST3_OUTPUT, dry_run=args.dry_run)

    if not args.dry_run:
        logger.info("Done. Register datasets in LLaMA-Factory/data/dataset_info.json")


if __name__ == "__main__":
    main()
