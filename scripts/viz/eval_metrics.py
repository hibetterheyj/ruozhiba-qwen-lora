#!/usr/bin/env python3
"""Phase 3.2 — 两阶段 JSON 评估协议.

Stage 1: JSON 格式遵循能力 (json_strict / json_tolerant / vsr)
Stage 2: 逻辑准确率 (top1_accuracy / top3_hit_rate / confidence_mae / strict_accuracy / repaired_accuracy)

产出目录结构:
  results/
    json/           — eval_{tag}.json + eval_comparison.json
    confusion_matrices/ — baseline + best 模型 + 汇总网格
    heatmaps/       — Rank×Epoch 热力图
        charts/         — 折线图、柱状图、雷达图等

训练日志说明:
    若从训练服务器取回 LLaMA-Factory 生成的 trainer_log.jsonl，脚本可额外绘制 train loss
    与 train/eval combined curves；若日志缺失，则只导出 eval loss 相关图表。

用法（仓库根目录）:
    # 评估单个结果文件
    python scripts/viz/eval_metrics.py \\
        --results results/results_r16_e5.json \\
        --gold data/CQIA/ruozhiba_cqia_classified_v2.json

    # 批量评估 + 生成对比总表 & 热力图
    python scripts/viz/eval_metrics.py \\
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
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from json_repair import repair_json
from matplotlib.colors import LinearSegmentedColormap

from font_utils import configure_matplotlib_cjk_fonts

matplotlib.use("Agg")

# CJK 字体支持
_CJK_FONTS = configure_matplotlib_cjk_fonts()

sns.set_theme(
    style="whitegrid",
    context="paper",
    rc={"font.family": _CJK_FONTS[0] if _CJK_FONTS else "sans-serif"},
)

MORANDI = {
    "blue_dark": "#6F8798",
    "blue_mid": "#8FA7B5",
    "blue_light": "#B7C7D1",
    "red_dark": "#B07A7A",
    "red_mid": "#C79B9B",
    "red_light": "#DFC0C0",
    "green_mid": "#98A892",
    "sand": "#C8B79E",
    "plum": "#8F7A8A",
    "gray": "#A7A29A",
    "cream": "#F5F1EB",
}

RB_DIVERGING = LinearSegmentedColormap.from_list(
    "morandi_rb", [MORANDI["blue_light"], MORANDI["cream"], MORANDI["red_mid"]]
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# 全局标题开关: --no_title 时设为 False
SHOW_TITLE = True
EXPORT_PDF = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAVES_DIR = PROJECT_ROOT / "LLaMA-Factory" / "saves" / "qwen3-4b" / "lora"

CATEGORIES = [
    "古典弱智", "奇怪提问", "弱智科学家", "人生态度",
    "文字游戏", "地狱笑话", "谐音梗", "文艺弱智",
]

# 色阶配置: 锁定 vmin/vmax 确保 all 与 last3 同一指标颜色映射一致
COLOR_SCALE = {
    "accuracy": {"vmin": 0.0, "vmax": 1.0, "cmap": RB_DIVERGING},
    "loss": {"vmin": 0.5, "vmax": 1.2, "cmap": RB_DIVERGING.reversed()},
}

METRICS_FOR_HEATMAP = [
    "strict_accuracy", "repaired_accuracy", "top1_accuracy",
    "top3_hit_rate", "json_strict", "vsr", "eval_loss",
]

TRAIN_TAG_LABELS = {
    "r8": "R8 all",
    "r16": "R16 all",
    "r8_last3": "R8 last3",
    "r16_last3": "R16 last3",
}


def save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure to PNG and optionally PDF."""
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", path)
    if EXPORT_PDF:
        pdf_path = path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        logger.info("Saved %s", pdf_path)


def hide_spines(ax_or_axes: plt.Axes | Iterable[plt.Axes]) -> None:
    """Hide top/right spines for cleaner report figures."""
    if isinstance(ax_or_axes, Iterable) and not isinstance(ax_or_axes, plt.Axes):
        for axis in ax_or_axes:
            hide_spines(axis)
        return
    ax_or_axes.spines["top"].set_visible(False)
    ax_or_axes.spines["right"].set_visible(False)


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


def get_top1_category(cats: list | None) -> str | None:
    """从 top3_categories 中取 rank=1 的 category。

    支持两种格式:
      1. [{"rank": 1, "category": "X", ...}, ...]
      2. ["X", "Y", "Z"]
    """
    if cats is None or len(cats) == 0:
        return None
    # 字符串列表格式
    if isinstance(cats[0], str):
        return cats[0]
    # 字典列表格式 — 优先找 rank=1
    for c in cats:
        if isinstance(c, dict) and c.get("rank") == 1:
            return c.get("category")
    # fallback: 取第一个
    if isinstance(cats[0], dict):
        return cats[0].get("category")
    return None


def get_top1_confidence(cats: list | None) -> float | None:
    """从 top3_categories 中取 rank=1 的 confidence_score。"""
    if cats is None or len(cats) == 0:
        return None
    # 字符串列表格式无置信度
    if isinstance(cats[0], str):
        return None
    for c in cats:
        if isinstance(c, dict) and c.get("rank") == 1:
            return c.get("confidence_score")
    if isinstance(cats[0], dict):
        return cats[0].get("confidence_score")
    return None


def get_top3_category_names(cats: list | None) -> list[str]:
    """返回 top3 的 category 名称列表。"""
    if cats is None:
        return []
    result = []
    for c in cats[:3]:
        if isinstance(c, str):
            result.append(c)
        elif isinstance(c, dict):
            result.append(c.get("category", ""))
    return result


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
            data, annot=True, fmt=fmt, cmap=RB_DIVERGING,
            xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        if SHOW_TITLE:
            ax.set_title(f"Confusion Matrix ({mode}) — {tag}")
        plt.tight_layout()
        path = output_dir / f"confusion_matrix_{tag}_{mode}.png"
        save_figure(fig, path)
        plt.close(fig)


def plot_confusion_grid(
    confusion_dict: dict[str, np.ndarray],
    tags: list[str],
    output_dir: Path,
    filename: str = "confusion_grid.png",
) -> None:
    """将多个模型的归一化混淆矩阵绘制在一张网格图中。"""
    n = len(tags)
    if n == 0:
        return
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(9 * cols, 7 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, tag in enumerate(tags):
        ax = axes[idx]
        confusion = confusion_dict[tag]
        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        data = confusion / row_sums
        sns.heatmap(
            data, annot=True, fmt=".2f", cmap=RB_DIVERGING,
            xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax,
            cbar=idx == len(tags) - 1,
        )
        if SHOW_TITLE:
            ax.set_title(tag, fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    path = output_dir / filename
    save_figure(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 新增图表: 折线图、柱状图、雷达图
# ---------------------------------------------------------------------------

def plot_accuracy_lines(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 Strict Accuracy & Top-3 Hit Rate 随 Epoch 变化的折线图 (4 组实验)。"""
    epochs = [3, 4, 5, 6, 7]
    groups = [
        ("R8 all", "r8", "#1f77b4", "-", "o"),
        ("R16 all", "r16", "#ff7f0e", "-", "s"),
        ("R8 last3", "r8_last3", "#1f77b4", "--", "^"),
        ("R16 last3", "r16_last3", "#ff7f0e", "--", "D"),
    ]

    for metric_key, ylabel, title_suffix in [
        ("strict_accuracy", "Strict Accuracy", "Strict Accuracy vs Epoch"),
        ("top3_hit_rate", "Top-3 Hit Rate", "Top-3 Hit Rate vs Epoch"),
        ("top1_accuracy", "Top-1 Accuracy", "Top-1 Accuracy vs Epoch"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, prefix, color, ls, marker in groups:
            vals = []
            for e in epochs:
                tag = f"{prefix}_e{e}"
                v = all_metrics.get(tag, {}).get(metric_key)
                vals.append(v)

            ax.plot(epochs, vals, color=color, linestyle=ls, marker=marker,
                    label=label, linewidth=2, markersize=7)

        # Baseline horizontal line
        bl_val = all_metrics.get("baseline", {}).get(metric_key)
        if bl_val is not None:
            ax.axhline(y=bl_val, color="gray", linestyle=":", linewidth=1.5,
                       label=f"Baseline ({bl_val:.3f})")

        hide_spines(ax)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if SHOW_TITLE:
            ax.set_title(title_suffix, fontsize=14)
        ax.set_xticks(epochs)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)

        fname = f"line_{metric_key}.png"
        path = output_dir / fname
        save_figure(fig, path)
        plt.close(fig)


def plot_eval_loss_lines(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 eval_loss 随 epoch 变化的折线图。"""
    epochs = [3, 4, 5, 6, 7]
    fig, ax = plt.subplots(figsize=(8, 5))

    line_cfgs = [
        ("R8 all", "r8", MORANDI["red_mid"], "-", "o"),
        ("R16 all", "r16", MORANDI["blue_dark"], "-", "s"),
        ("R8 last3", "r8_last3", MORANDI["red_light"], "--", "^"),
        ("R16 last3", "r16_last3", MORANDI["blue_light"], "--", "D"),
    ]

    for label, prefix, color, ls, marker in line_cfgs:
        vals = []
        for e in epochs:
            tag = f"{prefix}_e{e}"
            v = all_metrics.get(tag, {}).get("eval_loss")
            vals.append(v)
        ax.plot(epochs, vals, color=color, linestyle=ls, marker=marker,
                label=label, linewidth=2, markersize=7)

    hide_spines(ax)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Eval Loss", fontsize=12)
    if SHOW_TITLE:
        ax.set_title("Eval Loss vs Epoch", fontsize=14)
    ax.set_xticks(epochs)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    path = output_dir / "line_eval_loss.png"
    save_figure(fig, path)
    plt.close(fig)


def plot_baseline_vs_best_bar(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 baseline vs top-3 模型的多指标分组柱状图。"""
    # 选 top-3 by strict_accuracy (排除 baseline)
    non_bl = [(t, m) for t, m in all_metrics.items() if t != "baseline"]
    non_bl.sort(key=lambda x: x[1].get("strict_accuracy", 0), reverse=True)
    top_tags = [t for t, _ in non_bl[:3]]
    tags_to_show = ["baseline"] + top_tags

    metrics_to_show = [
        ("strict_accuracy", "Strict Acc"),
        ("top1_accuracy", "Top-1 Acc"),
        ("top3_hit_rate", "Top-3 Hit"),
        ("json_strict", "JSON Strict"),
    ]

    x = np.arange(len(metrics_to_show))
    width = 0.16
    colors = [MORANDI["gray"], MORANDI["blue_dark"], MORANDI["red_mid"], MORANDI["green_mid"]]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, tag in enumerate(tags_to_show):
        vals = [all_metrics.get(tag, {}).get(m, 0) for m, _ in metrics_to_show]
        ax.bar(x + i * width, vals, width * 0.88, label=tag, color=colors[i], alpha=0.92)

    hide_spines(ax)
    ax.set_xticks(x + width * (len(tags_to_show) - 1) / 2)
    ax.set_xticklabels([label for _, label in metrics_to_show], fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    if SHOW_TITLE:
        ax.set_title("Baseline vs Top-3 Models", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "bar_baseline_vs_top3.png"
    save_figure(fig, path)
    plt.close(fig)


def plot_all_vs_last3_delta(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 all vs last3 的 strict_accuracy 差值柱状图。"""
    labels = []
    deltas = []
    for rank in [8, 16]:
        for epoch in [3, 4, 5, 6, 7]:
            all_tag = f"r{rank}_e{epoch}"
            last3_tag = f"r{rank}_last3_e{epoch}"
            if all_tag in all_metrics and last3_tag in all_metrics:
                labels.append(f"R{rank} E{epoch}")
                deltas.append(
                    all_metrics[all_tag]["strict_accuracy"]
                    - all_metrics[last3_tag]["strict_accuracy"]
                )

    if not deltas:
        return

    red_series = [MORANDI["red_light"], MORANDI["red_mid"], MORANDI["red_dark"], "#9D6F6F", "#865D5D"]
    blue_series = [MORANDI["blue_light"], MORANDI["blue_mid"], MORANDI["blue_dark"], "#61798A", "#536878"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_colors = []
    for label in labels:
        rank = 8 if label.startswith("R8") else 16
        epoch = int(label.split("E")[1])
        idx = min(max(epoch - 3, 0), 4)
        bar_colors.append(red_series[idx] if rank == 8 else blue_series[idx])
    ax.bar(range(len(labels)), deltas, color=bar_colors, alpha=0.9, width=0.72)
    hide_spines(ax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Δ Strict Accuracy (all − last3)", fontsize=12)
    if SHOW_TITLE:
        ax.set_title("Full Dataset Advantage over Last-3-Year Subset", fontsize=14)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)

    path = output_dir / "bar_all_vs_last3_delta.png"
    save_figure(fig, path)
    plt.close(fig)


def plot_per_category_accuracy(
    confusion: np.ndarray, tag: str, output_dir: Path
) -> None:
    """绘制最优模型的各类别 recall 柱状图。"""
    row_sums = confusion.sum(axis=1)
    diag = confusion.diagonal()
    recall = np.where(row_sums > 0, diag / row_sums, 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        range(len(CATEGORIES)),
        recall,
        color=sns.color_palette([
            MORANDI["blue_light"], MORANDI["blue_mid"], MORANDI["blue_dark"], MORANDI["green_mid"],
            MORANDI["sand"], MORANDI["red_light"], MORANDI["red_mid"], MORANDI["plum"]
        ])
    )
    hide_spines(ax)
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, fontsize=11)
    ax.set_ylabel("Recall", fontsize=12)
    if SHOW_TITLE:
        ax.set_title(f"Per-Category Recall — {tag}", fontsize=14)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # 标注数值和样本数
    for i, (r, n) in enumerate(zip(recall, row_sums)):
        ax.text(i, r + 0.02, f"{r:.2f}\n(n={int(n)})", ha="center", fontsize=9)

    path = output_dir / f"bar_per_category_recall_{tag}.png"
    save_figure(fig, path)
    plt.close(fig)


def plot_radar_top_models(
    all_metrics: dict[str, dict], output_dir: Path
) -> None:
    """绘制 top-3 模型 + baseline 的雷达图。"""
    non_bl = [(t, m) for t, m in all_metrics.items() if t != "baseline"]
    non_bl.sort(key=lambda x: x[1].get("strict_accuracy", 0), reverse=True)
    tags_to_show = ["baseline"] + [t for t, _ in non_bl[:3]]

    radar_metrics = [
        ("strict_accuracy", "Strict Acc"),
        ("top1_accuracy", "Top-1 Acc"),
        ("top3_hit_rate", "Top-3 Hit"),
        ("json_strict", "JSON Strict"),
        ("vsr", "VSR"),
    ]

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = [MORANDI["gray"], MORANDI["blue_dark"], MORANDI["red_mid"], MORANDI["green_mid"]]

    for i, tag in enumerate(tags_to_show):
        vals = [all_metrics.get(tag, {}).get(m, 0) for m, _ in radar_metrics]
        vals += vals[:1]
        ax.plot(angles, vals, color=colors[i], linewidth=2, label=tag)
        ax.fill(angles, vals, color=colors[i], alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([label for _, label in radar_metrics], fontsize=11)
    ax.set_ylim(0, 1.05)
    if SHOW_TITLE:
        ax.set_title("Radar: Baseline vs Top-3 Models", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

    path = output_dir / "radar_top_models.png"
    save_figure(fig, path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 训练 loss 曲线 (从 trainer_log.jsonl)
# ---------------------------------------------------------------------------

def plot_training_loss_curves(output_dir: Path) -> None:
    """从 trainer_log.jsonl 绘制 4 组实验的训练 loss 曲线。"""
    line_cfgs = [
        ("R8 all", "r8", "#1f77b4", "-"),
        ("R16 all", "r16", "#ff7f0e", "-"),
        ("R8 last3", "r8_last3", "#1f77b4", "--"),
        ("R16 last3", "r16_last3", "#ff7f0e", "--"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, train_tag, color, ls in line_cfgs:
        log_path = SAVES_DIR / train_tag / "trainer_log.jsonl"
        if not log_path.exists():
            continue
        steps, losses = [], []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "loss" in entry and "eval_loss" not in entry:
                    steps.append(entry["current_steps"])
                    losses.append(entry["loss"])
        if steps:
            ax.plot(steps, losses, color=color, linestyle=ls,
                    label=label, linewidth=1.5, alpha=0.85)

    hide_spines(ax)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    if SHOW_TITLE:
        ax.set_title("Training Loss vs Step", fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    path = output_dir / "line_training_loss.png"
    save_figure(fig, path)
    plt.close(fig)

    summary_rows = []
    for label, train_tag, _, _ in line_cfgs:
        log_path = SAVES_DIR / train_tag / "trainer_log.jsonl"
        if not log_path.exists():
            continue
        best_eval = None
        best_step = None
        final_train = None
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "loss" in entry and "eval_loss" not in entry:
                    final_train = entry["loss"]
                elif "eval_loss" in entry:
                    eval_loss = entry["eval_loss"]
                    if best_eval is None or eval_loss < best_eval:
                        best_eval = eval_loss
                        best_step = entry["current_steps"]
        summary_rows.append({
            "train_tag": train_tag,
            "label": label,
            "source": str(log_path.relative_to(PROJECT_ROOT)),
            "best_eval_loss": best_eval,
            "best_eval_step": best_step,
            "final_train_loss": final_train,
        })

    if summary_rows:
        summary_path = output_dir.parent / "training" / "training_loss_summary_from_eval.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_rows, f, ensure_ascii=False, indent=2)
        logger.info("Saved %s", summary_path)


def plot_training_eval_loss_combined(output_dir: Path) -> None:
    """绘制每组实验的 train loss + eval loss 合并图 (2x2 子图)。"""
    train_tags = ["r8", "r16", "r8_last3", "r16_last3"]
    labels = ["R8 all", "R16 all", "R8 last3", "R16 last3"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (train_tag, label) in enumerate(zip(train_tags, labels)):
        ax = axes[idx]
        log_path = SAVES_DIR / train_tag / "trainer_log.jsonl"
        if not log_path.exists():
            continue

        train_steps, train_losses = [], []
        eval_steps, eval_losses = [], []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                if "eval_loss" in entry:
                    eval_steps.append(entry["current_steps"])
                    eval_losses.append(entry["eval_loss"])
                elif "loss" in entry:
                    train_steps.append(entry["current_steps"])
                    train_losses.append(entry["loss"])

        ax.plot(train_steps, train_losses, color="#1f77b4", linewidth=1.2,
                alpha=0.7, label="Train Loss")
        ax.plot(eval_steps, eval_losses, color="#ff7f0e", linewidth=2,
                marker="o", markersize=5, label="Eval Loss")
        hide_spines(ax)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel("Loss", fontsize=10)
        if SHOW_TITLE:
            ax.set_title(label, fontsize=12, fontweight="bold")
        else:
            # 无标题模式下用子图标签区分
            ax.text(0.02, 0.98, label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="top")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / "grid_train_eval_loss.png"
    save_figure(fig, path)
    plt.close(fig)


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
    output_dir.mkdir(parents=True, exist_ok=True)
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
            if SHOW_TITLE:
                ax.set_title(f"{metric_name} ({dataset_tag}) — Rank × Epoch")
            path = output_dir / f"heatmap_{dataset_tag}_{metric_name}.png"
            save_figure(fig, path)
            plt.close(fig)


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
    parser.add_argument(
        "--no_title", action="store_true",
        help="生成不带标题的图片 (用于报告嵌入)",
    )
    parser.add_argument(
        "--export_pdf", action="store_true",
        help="除 PNG 外额外导出 PDF 版本图片",
    )
    return parser.parse_args()


def main() -> None:
    global SHOW_TITLE, EXPORT_PDF
    args = parse_args()

    if args.no_title:
        SHOW_TITLE = False
    if args.export_pdf:
        EXPORT_PDF = True

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

    # 子目录
    json_dir = output_dir / "json"
    confusion_dir = output_dir / "confusion_matrices"
    heatmap_dir = output_dir / "heatmaps"
    chart_dir = output_dir / "charts"
    for d in [json_dir, confusion_dir, heatmap_dir, chart_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    all_confusions: dict[str, np.ndarray] = {}

    for rf in result_files:
        tag = rf.stem.replace("results_", "")
        logger.info("Evaluating: %s (tag=%s)", rf, tag)

        with open(rf, "r", encoding="utf-8") as f:
            results = json.load(f)

        metrics, confusion, per_sample = evaluate_single(results, gold_data)

        # 获取 eval_loss
        eval_loss = get_eval_loss_for_tag(tag)
        metrics["eval_loss"] = eval_loss

        # 保存单模型评估 JSON
        eval_out = json_dir / f"eval_{tag}.json"
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

        all_metrics[tag] = metrics
        all_confusions[tag] = confusion

    # --- 混淆矩阵 (精选模型) ---
    # baseline + top-3 by strict_accuracy
    non_bl = [(t, m) for t, m in all_metrics.items() if t != "baseline"]
    non_bl.sort(key=lambda x: x[1].get("strict_accuracy", 0), reverse=True)
    top3_tags = [t for t, _ in non_bl[:3]]
    cm_tags = []
    if "baseline" in all_confusions:
        cm_tags.append("baseline")
    cm_tags.extend(top3_tags)

    for tag in cm_tags:
        plot_confusion_matrix(all_confusions[tag], tag, confusion_dir)

    # 混淆矩阵网格 (baseline + top-3 in one figure)
    if len(cm_tags) > 1:
        plot_confusion_grid(all_confusions, cm_tags, confusion_dir, "confusion_grid_top_models.png")

    # 最优模型 per-category recall
    if top3_tags:
        best_tag = top3_tags[0]
        plot_per_category_accuracy(all_confusions[best_tag], best_tag, chart_dir)

    # --- 对比总表 & 热力图 & 新图表 ---
    if args.comparison and len(all_metrics) > 1:
        comparison = build_comparison(all_metrics, json_dir)
        plot_heatmaps(all_metrics, heatmap_dir)

        # 新增图表
        plot_accuracy_lines(all_metrics, chart_dir)
        plot_eval_loss_lines(all_metrics, chart_dir)
        plot_baseline_vs_best_bar(all_metrics, chart_dir)
        plot_all_vs_last3_delta(all_metrics, chart_dir)
        plot_radar_top_models(all_metrics, chart_dir)

        # 训练 loss 曲线 (从 trainer_log.jsonl)
        plot_training_loss_curves(chart_dir)
        plot_training_eval_loss_combined(chart_dir)

        # 打印摘要
        best = comparison.get("best_model", {})
        logger.info("=== Best Models ===")
        for k, v in best.items():
            logger.info("  %s: %s", k, v)

    logger.info("=== Evaluation complete (%d models) ===", len(all_metrics))


if __name__ == "__main__":
    main()
