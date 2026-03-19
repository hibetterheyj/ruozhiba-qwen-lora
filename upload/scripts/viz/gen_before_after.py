"""Generate before/after comparison samples for lab report Section 3.5."""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

# upload/scripts/viz -> 仓库根目录（与推理输出 results/ 一致）
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOGGER = logging.getLogger(__name__)


def _extract_format_case(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    preferred_indices: list[int],
) -> dict[str, Any] | None:
    """Pick one case that best illustrates baseline-vs-SFT output formatting."""
    def is_json_like(text: str) -> bool:
        return text.lstrip().startswith("{") and text.rstrip().endswith("}")

    def has_structured_fields(text: str) -> bool:
        return all(key in text for key in ["\"rank\"", "\"confidence_score\"", "\"reason\""])

    def has_string_list_top3(text: str) -> bool:
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                import json_repair
                parsed = json_repair.loads(text)
            except Exception:
                return False
        cats = parsed.get("top3_categories", []) if isinstance(parsed, dict) else []
        return bool(cats) and isinstance(cats[0], str)

    for idx in preferred_indices:
        base_output = str(baseline[idx].get("model_output", "")).strip()
        sft_output = str(candidate[idx].get("model_output", "")).strip()
        if not base_output or not sft_output:
            continue
        if (not is_json_like(base_output) and is_json_like(sft_output)) or (
            has_string_list_top3(base_output) and has_structured_fields(sft_output)
        ):
            return {
                "sample_index": idx,
                "instruction": baseline[idx].get("instruction", ""),
                "gold_top1": get_gold_top1(baseline[idx].get("gold_classification", {})),
                "baseline_output": base_output,
                "sft_output": sft_output,
                "baseline_top1": get_top1(base_output),
                "sft_top1": get_top1(sft_output),
            }

    for idx in preferred_indices:
        base_output = str(baseline[idx].get("model_output", "")).strip()
        sft_output = str(candidate[idx].get("model_output", "")).strip()
        if not base_output or not sft_output:
            continue
        if len(sft_output) >= len(base_output) and has_structured_fields(sft_output):
            return {
                "sample_index": idx,
                "instruction": baseline[idx].get("instruction", ""),
                "gold_top1": get_gold_top1(baseline[idx].get("gold_classification", {})),
                "baseline_output": base_output,
                "sft_output": sft_output,
                "baseline_top1": get_top1(base_output),
                "sft_top1": get_top1(sft_output),
            }
    return None


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for selecting comparison inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=PROJECT_ROOT / "results" / "results_baseline.json", help="Baseline inference result JSON file.")
    parser.add_argument("--candidate", type=Path, default=PROJECT_ROOT / "results" / "results_r16_e5.json", help="Fine-tuned model inference result JSON file.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "before_after_samples.json", help="Path to write the selected comparison samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when sampling representative cases.")
    return parser.parse_args()


def load_results(file_path: Path) -> list[dict[str, Any]]:
    """Load a result list from disk with an actionable error if missing."""
    if not file_path.exists():
        raise FileNotFoundError(f"Result file not found: {file_path}. Run inference first or override the path via CLI.")
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)

def get_top1(model_output: str) -> str | None:
    try:
        import json_repair
        parsed = json_repair.loads(model_output)
    except Exception:
        try:
            parsed = json.loads(model_output)
        except Exception:
            return None
    cats = parsed.get("top3_categories", [])
    if not cats:
        return None
    if isinstance(cats[0], dict):
        return cats[0].get("category")
    if isinstance(cats[0], str):
        return cats[0]
    return None


def get_gold_top1(gc: dict[str, Any]) -> str | None:
    cats = gc.get("top3_categories", [])
    if not cats:
        return None
    if isinstance(cats[0], dict):
        return cats[0].get("category")
    return cats[0]


def main() -> None:
    """Select representative baseline-vs-SFT comparison samples."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    baseline = load_results(args.baseline)
    best = load_results(args.candidate)

    if len(baseline) != len(best):
        raise ValueError(
            f"Input length mismatch: baseline has {len(baseline)} rows, candidate has {len(best)} rows. Use result files generated on the same evaluation set."
        )

    baseline_wrong_sft_right = []
    format_improvement = []  # both correct, baseline string list, SFT dict list
    both_correct_deeper = []
    sft_still_wrong = []

    for i in range(len(baseline)):
        gold_top1 = get_gold_top1(baseline[i]["gold_classification"])
        b_top1 = get_top1(baseline[i]["model_output"])
        s_top1 = get_top1(best[i]["model_output"])
        b_correct = b_top1 == gold_top1
        s_correct = s_top1 == gold_top1
        b_has_conf = "confidence_score" in baseline[i]["model_output"]
        s_has_conf = "confidence_score" in best[i]["model_output"]

        if not b_correct and s_correct:
            baseline_wrong_sft_right.append(i)
        elif b_correct and s_correct:
            if not b_has_conf and s_has_conf:
                format_improvement.append(i)
            else:
                both_correct_deeper.append(i)
        elif not s_correct:
            sft_still_wrong.append(i)

    random.seed(args.seed)
    samples = []

    # 2 cases: baseline wrong, SFT correct
    for idx in random.sample(baseline_wrong_sft_right,
                             min(2, len(baseline_wrong_sft_right))):
        samples.append({"index": idx, "type": "baseline_wrong_sft_correct"})

    # 1 case: format improvement
    if format_improvement:
        samples.append({"index": random.choice(format_improvement),
                        "type": "format_improvement"})

    # 1 case: both correct, deeper analysis
    if both_correct_deeper:
        samples.append({"index": random.choice(both_correct_deeper),
                        "type": "both_correct_deeper_analysis"})
    elif format_improvement:
        extra = [x for x in format_improvement
                 if x not in [s["index"] for s in samples]]
        if extra:
            samples.append({"index": random.choice(extra),
                            "type": "both_correct_deeper_analysis"})

    # 1 case: SFT failure
    if sft_still_wrong:
        samples.append({"index": random.choice(sft_still_wrong),
                        "type": "sft_failure_case"})

    output = []
    for s in samples:
        i = s["index"]
        output.append({
            "sample_index": i,
            "comparison_type": s["type"],
            "instruction": baseline[i]["instruction"],
            "gold_top1": get_gold_top1(baseline[i]["gold_classification"]),
            "baseline_top1": get_top1(baseline[i]["model_output"]),
            "sft_top1": get_top1(best[i]["model_output"]),
            "baseline_output": baseline[i]["model_output"],
            "sft_output": best[i]["model_output"],
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    raw_case = _extract_format_case(baseline, best, format_improvement + baseline_wrong_sft_right + both_correct_deeper)
    if raw_case is not None:
        raw_case_path = args.output.with_name(f"{args.output.stem}_raw_output_case.json")
        with open(raw_case_path, "w", encoding="utf-8") as f:
            json.dump(raw_case, f, ensure_ascii=False, indent=2)
        LOGGER.info("Saved raw output comparison case to %s", raw_case_path)

    LOGGER.info("Saved %s before/after samples to %s", len(output), args.output)
    for s in output:
        LOGGER.info("  [%s] idx=%s: gold=%s base=%s sft=%s", s['comparison_type'], s['sample_index'], s['gold_top1'], s['baseline_top1'], s['sft_top1'])


if __name__ == "__main__":
    main()
