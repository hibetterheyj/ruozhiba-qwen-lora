"""Generate before/after comparison samples for lab report Section 3.5."""
import json
import random
from pathlib import Path

# upload/scripts/viz -> 仓库根目录（与推理输出 results/ 一致）
PROJECT_ROOT = Path(__file__).resolve().parents[3]

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


def get_gold_top1(gc: dict) -> str | None:
    cats = gc.get("top3_categories", [])
    if not cats:
        return None
    if isinstance(cats[0], dict):
        return cats[0].get("category")
    return cats[0]


def main():
    with open(PROJECT_ROOT / "results/results_baseline.json") as f:
        baseline = json.load(f)
    with open(PROJECT_ROOT / "results/results_r16_e5.json") as f:
        best = json.load(f)

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

    random.seed(42)
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

    out_path = PROJECT_ROOT / "results/before_after_samples.json"
    with open(out_path, "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} before/after samples to {out_path}")
    for s in output:
        print(f"  [{s['comparison_type']}] idx={s['sample_index']}: "
              f"gold={s['gold_top1']} base={s['baseline_top1']} "
              f"sft={s['sft_top1']}")


if __name__ == "__main__":
    main()
