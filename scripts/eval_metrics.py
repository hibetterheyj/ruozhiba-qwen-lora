#!/usr/bin/env python3
"""Phase 3.2 — 两阶段 JSON 评估协议.

Stage 1: JSON 格式遵循能力 (json_strict / json_tolerant / vsr)
Stage 2: 逻辑准确率 (top1_accuracy / top3_hit_rate / confidence_mae / strict_accuracy / repaired_accuracy)

产出:
  - 单模型评估: results/eval_{tag}.json
  - 对比总表:   results/eval_comparison.json  (--comparison)
  - 混淆矩阵:   results/confusion_matrix_{tag}_*.png
  - 热力图:     results/heatmap_{dataset}_{metric}.png  (--comparison)

用法:
    # 评估单个结果文件
    python scripts/eval_metrics.py \\
        --results results/results_r16_e5.json \\
        --gold data/CQIA/ruozhiba_cqia_classified_v2.json

    # 批量评估 + 生成对比总表 & 热力图
    python scripts/eval_metrics.py \\
        --results_dir results/ \\
        --gold data/CQIA/ruozhiba_cqia_classified_v2.json \\
        --comparison
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from json_repair import repair_json

matplotlib.use("Agg")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVES_DIR = PROJECT_ROOT / "LLaMA-Factory" / "saves" / "qwen3-4b" / "lora"

CATEGORIES = [
    "古典弱智", "奇怪提问", "弱智科学家", "人生态度",
    "文字游戏", "地狱笑话", "谐音梗", "文艺弱智",
]

# 色阶配置: 锁定 vmin/vmax 确保 all 与 last3 同一指标颜色映射一致
COLOR_SCALE = {
    "accuracy": {"vmin": 0.0, "vmax": 1.0, "cmap": "YlOrRd"},
    "loss": {"vmin": 0.5, "vmax": 1.2, "cmap": "YlOrRd_r"},
}

METRICS_FOR_HEATMAP = [
    "strict_accuracy", "repaired_accuracy", "top1_accuracy",
    "top3_hit_rate", "json_strict", "vsr", "eval_loss",
]


# ---------------------------------------------------------------------------
# JSON 解析层
# ---------------------------------------------------------------------------

def parse_json_strict(text: str) -> dict | None:
    """原始输出直接 json.loads，成功返回 dict，失败返回 None。"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_json_tolerant(text: str) -> dict | None:
    """用正则提取最外层 {...} 后 json.loads。"""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except (json.JSONDecodeError, TypeError):
            pass
    return None


def parse_json_repair(text: str) -> dict | None:
    """使用 json-repair 库修复后解析。"""
    try:
        repaired = repair_json(text, return_objects=True)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass
    # fallback: 先提取 {...} 再修复
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            repaired = repair_json(m.group(), return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# 分类提取
# ---------------------------------------------------------------------------

def extract_top_categories(parsed: dict | None) -> list[dict] | None:
    """从解析后的 dict 提取 top3_categories 列表。

    支持两种结构:
      1. {"top3_categories": [...]}  (直接输出)
      2. {"classification": {"top3_categories": [...]}}  (嵌套在 classification 内)
    """
    if parsed is None:
        return None
    cats = parsed.get("top3_categories")
    if cats is None:
        cls = parsed.get("classification")
        if isinstance(cls, dict):
            cats = cls.get("top3_categories")
    if isinstance(cats, list) and len(cats) > 0:
        return cats
    return None


def get_top1_category(cats: list[dict] | None) -> str | None:
    """从 top3_categories 中取 rank=1 的 category。"""
    if cats is None:
        return None
    # 优先找 rank=1
    for c in cats:
        if c.get("rank") == 1:
            return c.get("category")
    # fallback: 取第一个
    return cats[0].get("category") if cats else None


def get_top1_confidence(cats: list[dict] | None) -> float | None:
    """从 top3_categories 中取 rank=1 的 confidence_score。"""
    if cats is None:
        return None
    for c in cats:
        if c.get("rank") == 1:
            return c.get("confidence_score")
    return cats[0].get("confidence_score") if cats else None


def get_top3_category_names(cats: list[dict] | None) -> list[str]:
    """返回 top3 的 category 名称列表。"""
    if cats is None:
        return []
    return [c.get("category", "") for c in cats[:3]]


# ---------------------------------------------------------------------------
# 单模型评估
# ---------------------------------------------------------------------------

def evaluate_single(results: list[dict], gold_data: list[dict]) -> dict:
    """对单个模型的推理结果执行两阶段评估。

    Args:
        results: 推理结果列表 (来自 results_{tag}.json).
        gold_data: 金标准数据 (ruozhiba_cqia_classified_v2.json).

    Returns:
        评估指标 dict.
    """
    n = len(results)
    if n == 0:
        logger.warning("Empty results, returning zeros")
        return {k: 0.0 for k in [
            "json_strict", "json_tolerant", "vsr",
            "top1_accuracy", "top3_hit_rate", "confidence_mae",
            "strict_accuracy", "repaired_accuracy",
            "confidence_when_correct", "confidence_when_wrong",
        ]}

    # 按 index 对齐 gold
    gold_by_idx = {}
    for item in gold_data:
        # gold_data 无 index 字段, 用列表顺序
        pass
    gold_list = gold_data  # 假设顺序一致

    # Stage 1 计数
    strict_ok = 0
    tolerant_ok = 0
    repair_ok = 0

    # Stage 2 累加器
    top1_correct_strict = 0   # strict JSON 通过且 top1 正确
    top1_correct_repair = 0   # repair 后 top1 正确
    top1_correct_any = 0      # 任意方式 top1 正确
    top3_hit = 0
    conf_sum = 0.0
    conf_count_valid = 0
    conf_correct_sum = 0.0
    conf_correct_count = 0
    conf_wrong_sum = 0.0
    conf_wrong_count = 0

    # 混淆矩阵
    cat_to_idx = {c: i for i, c in enumerate(CATEGORIES)}
    confusion = np.zeros((len(CATEGORIES), len(CATEGORIES)), dtype=int)

    per_sample = []

    for i, item in enumerate(results):
        output_text = item.get("model_output", "")
        gold_cls = item.get("gold_classification") or (
            gold_list[i].get("classification") if i < len(gold_list) else None
        )
        gold_cats = gold_cls.get("top3_categories") if isinstance(gold_cls, dict) else None
        gold_top1 = get_top1_category(gold_cats)
        gold_conf = get_top1_confidence(gold_cats)

        # Stage 1: JSON 解析
        parsed_strict = parse_json_strict(output_text)
        parsed_tolerant = parse_json_tolerant(output_text)
        parsed_repair = parse_json_repair(output_text)

        # 取最佳解析结果
        best_parsed = parsed_strict or parsed_tolerant or parsed_repair

        if parsed_strict is not None:
            strict_ok += 1
        if parsed_tolerant is not None:
            tolerant_ok += 1
        if parsed_repair is not None or parsed_strict is not None or parsed_tolerant is not None:
            repair_ok += 1

        # Stage 2: 分类准确率
        pred_cats_strict = extract_top_categories(parsed_strict)
        pred_cats_best = extract_top_categories(best_parsed)

        pred_top1_strict = get_top1_category(pred_cats_strict)
        pred_top1_best = get_top1_category(pred_cats_best)
        pred_conf_best = get_top1_confidence(pred_cats_best)
        pred_top3_names = get_top3_category_names(pred_cats_best)

        # Top-1 accuracy
        is_strict_correct = (pred_top1_strict is not None and pred_top1_strict == gold_top1)
        is_repair_correct = (pred_top1_best is not None and pred_top1_best == gold_top1)

        if is_strict_correct:
            top1_correct_strict += 1
        if is_repair_correct:
            top1_correct_repair += 1
            top1_correct_any += 1
        # Top-3 hit rate
        if gold_top1 and gold_top1 in pred_top3_names:
            top3_hit += 1

        # Confidence MAE (maximum penalty for invalid)
        if pred_conf_best is not None and gold_conf is not None:
            mae_i = abs(pred_conf_best - gold_conf)
            conf_sum += mae_i
            conf_count_valid += 1
        else:
            conf_sum += 1.0  # maximum penalty

        # Confidence calibration
        if pred_conf_best is not None:
            if is_repair_correct:
                conf_correct_sum += pred_conf_best
                conf_correct_count += 1
            else:
                conf_wrong_sum += pred_conf_best
                conf_wrong_count += 1

        # 混淆矩阵
        if gold_top1 in cat_to_idx and pred_top1_best in cat_to_idx:
            confusion[cat_to_idx[gold_top1], cat_to_idx[pred_top1_best]] += 1

        per_sample.append({
            "index": i,
            "gold_top1": gold_top1,
            "pred_top1": pred_top1_best,
            "json_strict_ok": parsed_strict is not None,
            "top1_correct": is_repair_correct,
        })

    metrics = {
        "n_samples": n,
        "json_strict": strict_ok / n,
        "json_tolerant": tolerant_ok / n,
        "vsr": repair_ok / n,
        "top1_accuracy": top1_correct_any / n,
        "top3_hit_rate": top3_hit / n,
        "confidence_mae": conf_sum / n,
        "strict_accuracy": top1_correct_strict / n,
        "repaired_accuracy": top1_correct_repair / n,
        "confidence_when_correct": (
            conf_correct_sum / conf_correct_count if conf_correct_count > 0 else None
        ),
        "confidence_when_wrong": (
            conf_wrong_sum / conf_wrong_count if conf_wrong_count > 0 else None
        ),
    }

    return metrics, confusion, per_sample


# ---------------------------------------------------------------------------
# 混淆矩阵绘制
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    confusion: np.ndarray, tag: str, output_dir: Path
) -> None:
    """绘制归一化和计数两种混淆矩阵。"""
    for mode in ("counts", "normalized"):
        fig, ax = plt.subplots(figsize=(10, 8))
        if mode == "normalized":
            row_sums = confusion.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)  # avoid div by 0
            data = confusion / row_sums
            fmt = ".2f"
        else:
            data = confusion
            fmt = "d"

        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(f"Confusion Matrix ({mode}) — {tag}")
        plt.tight_layout()
        path = output_dir / f"confusion_matrix_{tag}_{mode}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# eval_loss 提取
# ---------------------------------------------------------------------------

def load_eval_losses(train_tag: str) -> list[dict]:
    """从 trainer_log.jsonl 提取 eval_loss 条目列表。

    Args:
        train_tag: 训练标签, e.g. "r8", "r16", "r8_last3", "r16_last3".

    Returns:
        [{"step": int, "epoch": float, "eval_loss": float}, ...] 列表.
    """
    log_path = SAVES_DIR / train_tag / "trainer_log.jsonl"
    if not log_path.exists():
        return []
    result = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            if "eval_loss" in entry:
                result.append({
                    "step": entry["current_steps"],
                    "epoch": entry["epoch"],
                    "eval_loss": entry["eval_loss"],
                })
    return result


def parse_model_tag(tag: str) -> dict:
    """解析 model_tag → {dataset, rank, epoch, train_tag, step}.

    Examples:
        "r8_e3"       → dataset=all, rank=8, epoch=3, train_tag=r8, step=249
        "r16_last3_e5"→ dataset=last3, rank=16, epoch=5, train_tag=r16_last3, step=155
        "baseline"    → dataset=None, rank=None, epoch=None, ...
    """
    if tag == "baseline":
        return {"dataset": None, "rank": None, "epoch": None, "train_tag": None, "step": None}

    # r{rank}_last3_e{epoch} or r{rank}_e{epoch}
    m = re.match(r"^r(\d+)_(last3_)?e(\d+)$", tag)
    if not m:
        return {"dataset": None, "rank": None, "epoch": None, "train_tag": None, "step": None}

    rank = int(m.group(1))
    is_last3 = m.group(2) is not None
    epoch = int(m.group(3))
    dataset = "last3" if is_last3 else "all"
    train_tag = f"r{rank}_last3" if is_last3 else f"r{rank}"

    # Compute step from epoch
    steps_per_epoch = 31 if is_last3 else 83
    step = epoch * steps_per_epoch

    return {
        "dataset": dataset,
        "rank": rank,
        "epoch": epoch,
        "train_tag": train_tag,
        "step": step,
    }


def get_eval_loss_for_tag(tag: str) -> float | None:
    """获取指定 model_tag 对应的 eval_loss (按 epoch 就近匹配)。"""
    info = parse_model_tag(tag)
    if info["train_tag"] is None or info["epoch"] is None:
        return None

    entries = load_eval_losses(info["train_tag"])
    if not entries:
        return None

    target_epoch = float(info["epoch"])
    # 找 epoch 最接近的 eval 条目
    closest = min(entries, key=lambda e: abs(e["epoch"] - target_epoch))
    # 容差: 允许 1.0 epoch 内的匹配 (all 的 eval_steps=100 导致不对齐)
    if abs(closest["epoch"] - target_epoch) <= 1.0:
        return closest["eval_loss"]
    return None


# ---------------------------------------------------------------------------
# 热力图绘制
# ---------------------------------------------------------------------------

def get_color_params(metric_name: str) -> dict:
    """根据指标类型返回色阶参数。"""
    if metric_name in ("eval_loss", "confidence_mae"):
        return COLOR_SCALE["loss"]
    return COLOR_SCALE["accuracy"]


def plot_heatmaps(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 Rank×Epoch 热力图 (7 指标 × 2 数据集 = 14 张)。"""
    for dataset_tag in ["all", "last3"]:
        for metric_name in METRICS_FOR_HEATMAP:
            matrix = np.full((2, 5), np.nan)  # R8, R16 × E3-E7
            for r_idx, rank in enumerate([8, 16]):
                for e_idx, epoch in enumerate([3, 4, 5, 6, 7]):
                    if dataset_tag == "all":
                        tag = f"r{rank}_e{epoch}"
                    else:
                        tag = f"r{rank}_last3_e{epoch}"
                    if tag in all_metrics and metric_name in all_metrics[tag]:
                        val = all_metrics[tag][metric_name]
                        if val is not None:
                            matrix[r_idx, e_idx] = val

            # 跳过全 NaN 的情况
            if np.all(np.isnan(matrix)):
                logger.warning("Skipping heatmap %s_%s: all NaN", dataset_tag, metric_name)
                continue

            color = get_color_params(metric_name)
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.heatmap(
                matrix, annot=True, fmt=".3f",
                xticklabels=["E3", "E4", "E5", "E6", "E7"],
                yticklabels=["R8", "R16"],
                cmap=color["cmap"],
                vmin=color["vmin"], vmax=color["vmax"],
                ax=ax,
            )
            ax.set_title(f"{metric_name} ({dataset_tag}) — Rank × Epoch")
            path = output_dir / f"heatmap_{dataset_tag}_{metric_name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# 对比总表
# ---------------------------------------------------------------------------

def build_comparison(
    all_metrics: dict[str, dict], output_dir: Path
) -> dict:
    """构建对比总表 JSON。"""
    table = []
    for tag, metrics in sorted(all_metrics.items()):
        info = parse_model_tag(tag)
        entry = {
            "model_tag": tag,
            "dataset": info["dataset"],
            "rank": info["rank"],
            "epoch": info["epoch"],
            "eval_loss": metrics.get("eval_loss"),
            "json_strict": metrics["json_strict"],
            "json_tolerant": metrics["json_tolerant"],
            "vsr": metrics["vsr"],
            "top1_accuracy": metrics["top1_accuracy"],
            "top3_hit_rate": metrics["top3_hit_rate"],
            "confidence_mae": metrics["confidence_mae"],
            "strict_accuracy": metrics["strict_accuracy"],
            "repaired_accuracy": metrics["repaired_accuracy"],
            "confidence_when_correct": metrics.get("confidence_when_correct"),
            "confidence_when_wrong": metrics.get("confidence_when_wrong"),
        }
        # VSR < 0.8 标记
        if metrics["vsr"] < 0.8:
            entry["vsr_warning"] = "instruction_following_unusable"
        table.append(entry)

    # Best model selection (排除 baseline)
    non_baseline = [e for e in table if e["model_tag"] != "baseline"]
    best = {}
    for metric_key in ["strict_accuracy", "repaired_accuracy", "top3_hit_rate"]:
        if non_baseline:
            best_entry = max(non_baseline, key=lambda e: e.get(metric_key, 0))
            best[f"by_{metric_key}"] = best_entry["model_tag"]

    # all vs last3 配对对比
    all_vs_last3 = []
    for rank in [8, 16]:
        for epoch in [3, 4, 5, 6, 7]:
            all_tag = f"r{rank}_e{epoch}"
            last3_tag = f"r{rank}_last3_e{epoch}"
            if all_tag in all_metrics and last3_tag in all_metrics:
                all_vs_last3.append({
                    "rank": rank,
                    "epoch": epoch,
                    "all_strict_accuracy": all_metrics[all_tag]["strict_accuracy"],
                    "last3_strict_accuracy": all_metrics[last3_tag]["strict_accuracy"],
                    "delta": round(
                        all_metrics[last3_tag]["strict_accuracy"]
                        - all_metrics[all_tag]["strict_accuracy"], 4
                    ),
                    "all_repaired_accuracy": all_metrics[all_tag]["repaired_accuracy"],
                    "last3_repaired_accuracy": all_metrics[last3_tag]["repaired_accuracy"],
                    "delta_repaired": round(
                        all_metrics[last3_tag]["repaired_accuracy"]
                        - all_metrics[all_tag]["repaired_accuracy"], 4
                    ),
                })

    comparison = {
        "comparison_table": table,
        "best_model": best,
        "all_vs_last3_comparison": all_vs_last3,
    }

    out_path = output_dir / "eval_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    logger.info("Saved comparison table: %s", out_path)
    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.2 — Two-stage JSON evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", type=str, help="单个结果文件路径")
    group.add_argument("--results_dir", type=str, help="结果目录(批量评估)")

    parser.add_argument(
        "--gold", type=str, required=True,
        help="金标准文件路径 (ruozhiba_cqia_classified_v2.json)",
    )
    parser.add_argument(
        "--comparison", action="store_true",
        help="生成对比总表和热力图",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录(默认与结果文件同目录)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 加载金标准
    with open(args.gold, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    logger.info("Loaded %d gold samples", len(gold_data))

    # 收集结果文件
    if args.results:
        result_files = [Path(args.results)]
    else:
        results_dir = Path(args.results_dir)
        result_files = sorted(results_dir.glob("results_*.json"))

    if not result_files:
        logger.error("No result files found")
        return

    output_dir = Path(args.output_dir) if args.output_dir else result_files[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}

    for rf in result_files:
        tag = rf.stem.replace("results_", "")
        logger.info("Evaluating: %s (tag=%s)", rf, tag)

        with open(rf, "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics, confusion, per_sample = evaluate_single(results, gold_data)

        # 获取 eval_loss
        eval_loss = get_eval_loss_for_tag(tag)
        metrics["eval_loss"] = eval_loss

        # 保存单模型评估
        eval_out = output_dir / f"eval_{tag}.json"
        with open(eval_out, "w", encoding="utf-8") as f:
            json.dump(
                {"model_tag": tag, "metrics": metrics, "per_sample": per_sample},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(
            "  → strict_acc=%.3f  repaired_acc=%.3f  vsr=%.3f  json_strict=%.3f  eval_loss=%s",
            metrics["strict_accuracy"], metrics["repaired_accuracy"],
            metrics["vsr"], metrics["json_strict"],
            f"{eval_loss:.4f}" if eval_loss is not None else "N/A",
        )

        # 混淆矩阵
        plot_confusion_matrix(confusion, tag, output_dir)

        all_metrics[tag] = metrics

    # 对比总表 & 热力图
    if args.comparison and len(all_metrics) > 1:
        comparison = build_comparison(all_metrics, output_dir)
        plot_heatmaps(all_metrics, output_dir)

        # 打印摘要
        best = comparison.get("best_model", {})
        logger.info("=== Best Models ===")
        for k, v in best.items():
            logger.info("  %s: %s", k, v)

    logger.info("=== Evaluation complete (%d models) ===", len(all_metrics))


if __name__ == "__main__":
    main()
