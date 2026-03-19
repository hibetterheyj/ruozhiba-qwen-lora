# 弱智吧幽默分类 SFT 微调 — 开发计划 v0.3

> 基于 `dev_plan_0_2.md` 更新。Phase 1-2 全部完成，Phase 2.7 全量合并完成。
>
> **v0.2 → v0.3 变更**: Phase 3.1 推理后端从 **sglang** 迁移至 **vLLM**。
> 原因: sgl_kernel 与系统 torch (2.8.0+cu129) ABI 不兼容，`common_ops.abi3.so` 加载失败。
> 详见 `doc/course/changelog.md`（本仓库变更日志）中 "Phase 3.1 sglang 推理失败" 条目。
>
> **Phase 3.2 评估脚本 (`eval_metrics.py`) 无需修改** — 仅消费 JSON 结果文件。

---

## Phase 3.1 (v0.3): vLLM 批量推理

### 环境变更

| 项目 | v0.2 (sglang) | v0.3 (vLLM) |
|------|--------------|-------------|
| 推理后端 | sglang 0.5.3 | vLLM 0.9+ |
| 虚拟环境 | env_sft | **env_vllm** (新建) |
| 引擎 API | `sgl.Engine` | `vllm.LLM` |
| 显存管理 | `mem_fraction_static=0.85` | `gpu_memory_utilization=0.85` |
| 清理方式 | engine.shutdown() + 三重清理 | del llm + gc + empty_cache |

### 环境搭建: `env_vllm`

```bash
cd /root/code/llm_ruozhiba
python3 -m venv env_vllm
source env_vllm/bin/activate
pip install vllm pyyaml json-repair
```

> **不修改** env_sft 和系统环境。env_vllm 仅用于推理，与训练环境完全隔离。

### 更新脚本: `scripts/inference/inference_eval.py`

**核心变更**: `sgl.Engine` → `vllm.LLM` + `SamplingParams`

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

def run_inference(model_path, test_data, system_prompt, tag, output_dir, gpu_id=0):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 构造 prompt
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
    
    # vLLM 推理
    llm = LLM(model=model_path, trust_remote_code=True, gpu_memory_utilization=0.85)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=1500)
    outputs = llm.generate(prompts, sampling_params)
    
    # 提取文本
    results = []
    for i, out in enumerate(outputs):
        results.append({
            "index": i,
            "instruction": test_data[i]["instruction"],
            "gold_classification": test_data[i].get("classification"),
            "model_output": out.outputs[0].text,
            "model_tag": tag,
        })
    
    # 保存 + 清理
    # ...
    del llm
    gc.collect()
    torch.cuda.empty_cache()
```

**关键差异 vs sglang**:

| 对比项 | sglang | vLLM |
|--------|--------|------|
| 引擎创建 | `sgl.Engine(model_path=..., tp_size=1, mem_fraction_static=0.85)` | `LLM(model=..., trust_remote_code=True, gpu_memory_utilization=0.85)` |
| Tokenizer | `engine.get_tokenizer()` | `AutoTokenizer.from_pretrained(model_path)` |
| 推理调用 | `engine.generate(prompts, sampling_params)` → `out["text"]` | `llm.generate(prompts, sampling_params)` → `out.outputs[0].text` |
| 参数命名 | `max_new_tokens` | `max_tokens` |
| 清理 | `engine.shutdown()` 必须显式调用 | `del llm` 即可 |

### 更新脚本: `scripts/inference/batch_inference.sh`

仅需变更虚拟环境激活路径:

```bash
# v0.2: source "${PROJECT_ROOT}/env_sft/bin/activate"
# v0.3:
source "${PROJECT_ROOT}/env_vllm/bin/activate"
```

其余逻辑 (模型遍历顺序、CUDA_VISIBLE_DEVICES、幂等跳过) 保持不变。

### 产出

与 v0.2 完全一致:
- `results/results_{tag}.json` × 21 个文件
- 每个文件包含 240 条推理结果
- JSON 结构不变: `{index, instruction, gold_classification, model_output, model_tag}`

---

## Phase 3.2: 定量评估 (不变)

**无需修改** — `scripts/viz/eval_metrics.py` 仅消费 `results/results_*.json`，与推理后端无关。

执行命令不变:

```bash
source env_sft/bin/activate  # eval_metrics.py 使用 env_sft (matplotlib/seaborn/json-repair)
python scripts/viz/eval_metrics.py \
    --results_dir results/ \
    --gold data/CQIA/ruozhiba_cqia_classified_v2.json \
    --comparison
```

---

## Phase 3.3 / 3.4 / Phase 4: 不变

与 `dev_plan_0_2.md` 完全一致，不受推理后端迁移影响。
