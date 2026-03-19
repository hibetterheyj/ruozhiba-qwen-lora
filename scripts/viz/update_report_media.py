import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path('/Users/heyujie/Documents/code/ruozhiba-qwen-lora')
RESULTS_CHARTS = ROOT / 'results' / 'charts'
RESULTS_HEATMAPS = ROOT / 'results' / 'heatmaps'
LATEX_MEDIA = ROOT / 'archive' / 'lab3_report_latex' / 'media'

MORANDI = {
    'blue': '#7C93A6',
    'green': '#9AA88F',
    'rose': '#B28C8C',
    'mustard': '#C2A46F',
    'plum': '#8F7A8A',
    'gray': '#A7A29A',
    'teal': '#7E9B9A',
    'cream': '#D8CBB8',
}

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    'report_heatmap_contrast',
    ['#234A6B', '#F4EFE8', '#8E2F3F'],
)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def make_dataset_donut() -> None:
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect='equal'))

    inner_sizes = [2785, 240]
    inner_labels = ['Training\n2785', 'Test\n240']
    inner_colors = [MORANDI['blue'], MORANDI['rose']]

    outer_sizes = [1424, 1361, 240]
    outer_labels = ['Tieba\n1424', 'GitHub\n1361', 'CQIA\n240']
    outer_colors = [MORANDI['blue'], '#9EB3C1', '#E2C4C4']

    outer_wedges, _ = ax.pie(
        outer_sizes,
        radius=1.0,
        colors=outer_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor='white'),
    )
    inner_wedges, _ = ax.pie(
        inner_sizes,
        radius=0.72,
        colors=inner_colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(width=0.28, edgecolor='white'),
    )

    def annotate_ring(wedges, labels, radius):
        for wedge, label in zip(wedges, labels):
            angle = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            ax.text(x, y, label, ha='center', va='center', fontsize=12, fontweight='bold', color='#3F3A36')

    annotate_ring(inner_wedges, inner_labels, 0.58)
    annotate_ring(outer_wedges, outer_labels, 0.88)
    # ax.set_title('Dataset Usage and Source Composition', fontsize=14, color='#3F3A36', pad=18)
    out = RESULTS_CHARTS / 'dataset_source_donut.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def remake_strict_accuracy_heatmap() -> None:
    src = ROOT / 'results' / 'json' / 'eval_comparison.json'
    if not src.exists():
        return

    import json

    with open(src, 'r', encoding='utf-8') as f:
        comparison = json.load(f)

    lookup = {
        row['model_tag']: row.get('strict_accuracy')
        for row in comparison.get('comparison_table', [])
    }

    matrix = np.full((2, 5), np.nan)
    for r_idx, rank in enumerate([8, 16]):
        for e_idx, epoch in enumerate([3, 4, 5, 6, 7]):
            tag = f'r{rank}_e{epoch}'
            value = lookup.get(tag)
            if value is not None:
                matrix[r_idx, e_idx] = value

    if np.all(np.isnan(matrix)):
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(matrix, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect='auto')
    ax.set_xticks(range(5), ['E3', 'E4', 'E5', 'E6', 'E7'])
    ax.set_yticks(range(2), ['R8', 'R16'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rank')
    # ax.set_title('Strict Accuracy (all)')

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if not np.isnan(matrix[i, j]):
                ax.text(j, i, f'{matrix[i, j]:.3f}', ha='center', va='center', color='#1F1F1F', fontsize=10, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Strict accuracy')
    fig.tight_layout()

    out = RESULTS_HEATMAPS / 'heatmap_all_strict_accuracy.pdf'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def remake_all_vs_last3_delta_bar() -> None:
    src = ROOT / 'results' / 'json' / 'eval_comparison.json'
    if not src.exists():
        return

    import json

    with open(src, 'r', encoding='utf-8') as f:
        comparison = json.load(f)

    rows = comparison.get('all_vs_last3_comparison', [])
    if not rows:
        return

    labels = [f"R{row['rank']} E{row['epoch']}" for row in rows]
    deltas = [row['all_strict_accuracy'] - row['last3_strict_accuracy'] for row in rows]

    red_series = ['#D7A6A6', '#C88484', '#B56565', '#9E4D4D', '#843838']
    blue_series = ['#B8CAD8', '#96B1C4', '#7396AF', '#527B98', '#365F80']

    colors = []
    for row in rows:
        idx = min(max(int(row['epoch']) - 3, 0), 4)
        colors.append(red_series[idx] if int(row['rank']) == 8 else blue_series[idx])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(labels)), deltas, color=colors, alpha=0.95, width=0.72)

    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
    ax.set_ylabel('Δ Strict Accuracy (all − last3)')
    # ax.set_title('Full Dataset Advantage over Last-3-Year Subset')
    ax.axhline(y=0, color='#4A4A4A', linewidth=0.9)
    ax.grid(True, axis='y', alpha=0.25, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    upper = max(deltas) if deltas else 0.0
    ax.set_ylim(0, max(0.02, upper + 0.04))

    for bar, delta in zip(bars, deltas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.004,
            f"+{delta:.3f}",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color='#2F2A26',
        )

    fig.tight_layout()
    out = RESULTS_CHARTS / 'bar_all_vs_last3_delta.pdf'
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def copy_files() -> None:
    LATEX_MEDIA.mkdir(parents=True, exist_ok=True)
    chart_needed = [
        'dataset_source_donut.pdf',
        'line_eval_loss.pdf',
        'line_strict_accuracy.pdf',
        'bar_all_vs_last3_delta.pdf',
        'bar_baseline_vs_top3.pdf',
    ]
    heatmap_needed = ['heatmap_all_strict_accuracy.pdf']

    for name in chart_needed:
        src = RESULTS_CHARTS / name
        if src.exists():
            shutil.copy2(src, LATEX_MEDIA / name)

    for name in heatmap_needed:
        src = RESULTS_HEATMAPS / name
        if src.exists():
            shutil.copy2(src, LATEX_MEDIA / name)


if __name__ == '__main__':
    make_dataset_donut()
    remake_strict_accuracy_heatmap()
    remake_all_vs_last3_delta_bar()
    copy_files()
