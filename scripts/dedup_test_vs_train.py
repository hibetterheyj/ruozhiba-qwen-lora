"""去重防污染：确保 CQIA 测试集条目不出现在贴吧训练集中，防止数据泄露。

去重方向：从训练集中剔除与测试集重复的条目，测试集保持不变。

输入:
  - 测试集: data/CQIA/ruozhiba_cqia_classified_v2.json (或 _classified.json)
  - 训练集: data/tieba/best*_classified.json (9 个文件)
输出:
  - 去重报告: data/dedup_report.json
  - 去重后训练集: data/tieba/best*_classified_dedup.json
"""

import hashlib
import json
import glob
import logging
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_json(file_path: Path) -> List[Dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, file_path: Path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def text_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Exact dedup (MD5)
# ---------------------------------------------------------------------------

def find_exact_matches(
    test_hashes: Set[str],
    test_hash_to_text: Dict[str, str],
    train_files: List[Tuple[Path, List[Dict]]],
) -> Tuple[Dict[str, List[Dict]], List[Dict]]:
    """返回 {file_name: [matched_records]} 和全局 matched_records 列表。"""
    per_file: Dict[str, List[Dict]] = {}
    all_records: List[Dict] = []

    for file_path, data in train_files:
        fname = file_path.name
        per_file[fname] = []
        for item in data:
            h = text_hash(item["text"])
            if h in test_hashes:
                record = {
                    "train_file": fname,
                    "train_text": item["text"],
                    "test_text": test_hash_to_text[h],
                    "similarity": 1.0,
                    "match_type": "exact",
                }
                per_file[fname].append(record)
                all_records.append(record)

    return per_file, all_records


# ---------------------------------------------------------------------------
# Fuzzy dedup (SequenceMatcher) — parallelised per train file
# ---------------------------------------------------------------------------

def _fuzzy_worker(args: Tuple[str, List[str], List[str], float]) -> List[Dict]:
    """Worker: compare all texts in one train file against test texts."""
    fname, train_texts, test_texts, threshold = args
    matches: List[Dict] = []
    for train_text in train_texts:
        for test_text in test_texts:
            sim = SequenceMatcher(None, train_text.strip(), test_text.strip()).ratio()
            if threshold <= sim < 1.0:
                matches.append({
                    "train_file": fname,
                    "train_text": train_text,
                    "test_text": test_text,
                    "similarity": round(sim, 4),
                    "match_type": "fuzzy",
                })
    return matches


def find_fuzzy_matches(
    test_texts: List[str],
    train_files: List[Tuple[Path, List[Dict]]],
    exact_matched_train_texts: Set[str],
    threshold: float = 0.9,
) -> List[Dict]:
    """模糊匹配：跳过已精确命中的训练集条目。"""
    tasks = []
    for file_path, data in train_files:
        texts = [
            item["text"] for item in data
            if item["text"] not in exact_matched_train_texts
        ]
        tasks.append((file_path.name, texts, test_texts, threshold))

    all_matches: List[Dict] = []
    with ProcessPoolExecutor() as pool:
        for result in pool.map(_fuzzy_worker, tasks):
            all_matches.extend(result)

    return all_matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    # ---- 定位测试集 (优先 v2，回退 v1) ----
    test_path_v2 = base_dir / "data" / "CQIA" / "ruozhiba_cqia_classified_v2.json"
    test_path_v1 = base_dir / "data" / "CQIA" / "ruozhiba_cqia_classified.json"
    test_path = test_path_v2 if test_path_v2.exists() else test_path_v1
    log.info("测试集: %s", test_path.relative_to(base_dir))

    test_data = load_json(test_path)
    log.info("测试集条数: %d", len(test_data))

    test_texts = [item["instruction"] for item in test_data]
    test_hashes: Set[str] = {text_hash(t) for t in test_texts}
    test_hash_to_text: Dict[str, str] = {text_hash(t): t for t in test_texts}

    # ---- 加载全部贴吧训练集 ----
    tieba_dir = base_dir / "data" / "tieba"
    train_paths = sorted(
        Path(p) for p in glob.glob(str(tieba_dir / "best*_classified.json"))
        if "_dedup" not in Path(p).stem
    )
    log.info("训练集文件数: %d", len(train_paths))

    train_files: List[Tuple[Path, List[Dict]]] = []
    total_before = 0
    per_file_before: Dict[str, int] = {}
    for p in train_paths:
        data = load_json(p)
        train_files.append((p, data))
        per_file_before[p.name] = len(data)
        total_before += len(data)
        log.info("  %s: %d 条", p.name, len(data))

    log.info("训练集总量 (去重前): %d", total_before)

    # ---- Phase 1: 精确去重 ----
    log.info("Phase 1: 精确去重 (MD5 哈希)...")
    exact_per_file, exact_records = find_exact_matches(
        test_hashes, test_hash_to_text, train_files
    )
    exact_matched_train_texts: Set[str] = {r["train_text"] for r in exact_records}
    log.info("精确命中: %d 条 (跨 %d 个文件)",
             len(exact_records),
             sum(1 for v in exact_per_file.values() if v))

    # ---- Phase 2: 模糊去重 ----
    log.info("Phase 2: 模糊去重 (SequenceMatcher, 阈值=0.9)...")
    fuzzy_records = find_fuzzy_matches(
        test_texts, train_files, exact_matched_train_texts, threshold=0.9
    )
    fuzzy_matched_train_texts: Set[str] = {r["train_text"] for r in fuzzy_records}
    log.info("模糊命中: %d 条", len(fuzzy_records))

    # ---- 合并待移除集合 & 输出去重后训练集 ----
    all_remove_texts = exact_matched_train_texts | fuzzy_matched_train_texts

    total_after = 0
    per_file_after: Dict[str, int] = {}
    per_file_removed: Dict[str, int] = {}

    for file_path, data in train_files:
        deduped = [item for item in data if item["text"] not in all_remove_texts]
        out_path = file_path.with_name(
            file_path.stem.replace("_classified", "_classified_dedup") + ".json"
        )
        save_json(deduped, out_path)

        removed = len(data) - len(deduped)
        per_file_after[file_path.name] = len(deduped)
        per_file_removed[file_path.name] = removed
        total_after += len(deduped)
        log.info("  %s: %d → %d (移除 %d)", file_path.name, len(data), len(deduped), removed)

    log.info("训练集总量 (去重后): %d (移除 %d 条)", total_after, total_before - total_after)

    # ---- 近三年 / 全量分组统计 ----
    recent_keywords = ["2023", "2024", "2025"]

    def _sum_group(names: List[str], store: Dict[str, int]) -> int:
        return sum(store.get(n, 0) for n in names)

    recent_files = [n for n in per_file_before if any(k in n for k in recent_keywords)]
    all_files = list(per_file_before.keys())

    # ---- 构建去重报告 ----
    report = {
        "test_set": {
            "file": str(test_path.relative_to(base_dir)),
            "count": len(test_data),
        },
        "before": {
            "total": total_before,
            "recent_3y": _sum_group(recent_files, per_file_before),
            "per_file": per_file_before,
        },
        "after": {
            "total": total_after,
            "recent_3y": _sum_group(recent_files, per_file_after),
            "per_file": per_file_after,
        },
        "removed": {
            "total": total_before - total_after,
            "exact": len(exact_records),
            "fuzzy": len(fuzzy_records),
            "per_file": per_file_removed,
        },
        "exact_matches": exact_records,
        "fuzzy_matches": fuzzy_records,
    }

    report_path = base_dir / "data" / "dedup_report.json"
    save_json(report, report_path)
    log.info("去重报告已保存: %s", report_path.relative_to(base_dir))

    # ---- 打印摘要 ----
    print("\n" + "=" * 60)
    print("去重统计摘要")
    print("=" * 60)
    print(f"去重前:")
    print(f"  贴吧训练集总量: {total_before} 条")
    print(f"    近三年 (2023-2025): {_sum_group(recent_files, per_file_before)} 条")
    print(f"    全量 (2018-2025):   {total_before} 条")
    print(f"  CQIA 测试集: {len(test_data)} 条")
    print(f"\n去重后:")
    print(f"  贴吧训练集总量: {total_after} 条 (移除 {total_before - total_after} 条重复)")
    print(f"    近三年 (2023-2025): {_sum_group(recent_files, per_file_after)} 条")
    print(f"    全量 (2018-2025):   {total_after} 条")
    print(f"\n命中明细:")
    print(f"  精确命中: {len(exact_records)} 条")
    print(f"  模糊命中 (≥0.9): {len(fuzzy_records)} 条")
    for fname in sorted(per_file_removed):
        if per_file_removed[fname] > 0:
            print(f"    {fname}: 移除 {per_file_removed[fname]} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
