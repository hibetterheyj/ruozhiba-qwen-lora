#!/usr/bin/env python3
"""Phase 3.1 — sglang 离线批量推理脚本.

对指定模型(或模型目录)在 CQIA 测试集上执行 greedy-decoding 推理，
逐模型加载→推理→释放显存，输出 results/results_{tag}.json。

用法:
    # 单模型
    python scripts/inference_eval.py \
        --model_path models/Qwen3-4B-Instruct-2507 --tag baseline --gpu 0

    # 批量(扫描 models/merged/ 下所有子目录)
    python scripts/inference_eval.py --model_dir models/merged --gpu 0

    # 指定多个模型
    python scripts/inference_eval.py \
        --model_paths models/merged/r8_e3 models/merged/r16_e5 --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import time
from pathlib import Path

import torch
import yaml

# 抑制 sglang 底层日志，保持终端清爽
logging.getLogger("sglang").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_DATA = PROJECT_ROOT / "data" / "CQIA" / "ruozhiba_cqia_classified_v2.json"
DEFAULT_PROMPTS = PROJECT_ROOT / "configs" / "prompts.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results"


def load_test_data(path: Path) -> list[dict]:
    """加载 CQIA 测试集 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info("Loaded %d test samples from %s", len(data), path)
    return data


def load_system_prompt(path: Path) -> str:
    """从 prompts.yaml 加载 system_prompt。"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["system_prompt"]


def run_inference(
    model_path: str,
    test_data: list[dict],
    system_prompt: str,
    tag: str,
    output_dir: Path,
    gpu_id: int = 0,
) -> Path:
    """对单个模型执行批量推理并保存结果。

    Args:
        model_path: 模型路径(HF 格式目录).
        test_data: 测试样本列表.
        system_prompt: 系统提示词.
        tag: 模型标签(用于输出命名).
        output_dir: 输出目录.
        gpu_id: 使用的 GPU 编号(已通过 CUDA_VISIBLE_DEVICES 映射).

    Returns:
        输出文件路径.
    """
    import sglang as sgl

    output_path = output_dir / f"results_{tag}.json"

    # 跳过已存在的结果
    if output_path.exists():
        logger.info("⏭️  Results already exist, skipping: %s", output_path)
        return output_path

    logger.info("Loading model: %s (tag=%s)", model_path, tag)
    t0 = time.time()

    # 1. 启动引擎
    engine = sgl.Engine(
        model_path=model_path,
        tp_size=1,
        mem_fraction_static=0.85,
    )

    # 2. 构造 prompt (apply_chat_template)
    tokenizer = engine.get_tokenizer()
    prompts = []
    for item in test_data:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["instruction"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)

    # 3. 批量推理
    sampling_params = {"temperature": 0.0, "max_new_tokens": 1500}
    logger.info("Running inference on %d samples...", len(prompts))
    outputs = engine.generate(prompts, sampling_params)

    # 4. 保存结果
    results = []
    for i, out in enumerate(outputs):
        results.append(
            {
                "index": i,
                "instruction": test_data[i]["instruction"],
                "gold_classification": test_data[i].get("classification"),
                "model_output": out["text"],
                "model_tag": tag,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    logger.info("✅ Saved %d results to %s (%.1fs)", len(results), output_path, elapsed)

    # 5. 三重清理 —— 彻底释放 CUDA 显存
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(3)

    return output_path


def derive_tag_from_path(model_path: Path) -> str:
    """从模型目录名推导 tag，例如 models/merged/r8_e3 → r8_e3。"""
    return model_path.name


def collect_models_from_dir(model_dir: Path) -> list[tuple[Path, str]]:
    """扫描模型目录，返回 (路径, tag) 列表(按名称排序)。"""
    models = []
    for d in sorted(model_dir.iterdir()):
        if d.is_dir() and (d / "config.json").exists():
            models.append((d, derive_tag_from_path(d)))
    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3.1 — sglang batch inference")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model_path", type=str, help="单个模型路径")
    group.add_argument("--model_dir", type=str, help="模型目录(扫描所有子目录)")
    group.add_argument("--model_paths", type=str, nargs="+", help="多个模型路径")

    parser.add_argument("--tag", type=str, default=None, help="模型标签(仅 --model_path 时生效)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU 编号")
    parser.add_argument(
        "--test_data",
        type=str,
        default=str(DEFAULT_TEST_DATA),
        help="测试集路径",
    )
    parser.add_argument(
        "--prompts",
        type=str,
        default=str(DEFAULT_PROMPTS),
        help="prompts.yaml 路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载测试数据和 system_prompt
    test_data = load_test_data(Path(args.test_data))
    system_prompt = load_system_prompt(Path(args.prompts))

    # 构造模型列表: [(path, tag), ...]
    jobs: list[tuple[str, str]] = []
    if args.model_path:
        tag = args.tag or derive_tag_from_path(Path(args.model_path))
        jobs.append((args.model_path, tag))
    elif args.model_dir:
        for path, tag in collect_models_from_dir(Path(args.model_dir)):
            jobs.append((str(path), tag))
    else:
        for p in args.model_paths:
            jobs.append((p, derive_tag_from_path(Path(p))))

    if not jobs:
        logger.error("No models found. Check your arguments.")
        return

    logger.info("=== Batch inference: %d model(s), %d samples each ===", len(jobs), len(test_data))

    for idx, (model_path, tag) in enumerate(jobs, 1):
        logger.info("[%d/%d] Model: %s (tag=%s)", idx, len(jobs), model_path, tag)
        run_inference(
            model_path=model_path,
            test_data=test_data,
            system_prompt=system_prompt,
            tag=tag,
            output_dir=output_dir,
            gpu_id=args.gpu,
        )

    logger.info("=== All inference complete (%d models) ===", len(jobs))


if __name__ == "__main__":
    main()
