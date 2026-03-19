# 环境与依赖

## 硬件建议

- **训练**：NVIDIA GPU，本项目使用 2× L20Z（各 80GB VRAM）
- **推理 / 评估**：≥ 24GB VRAM 较稳妥（视 batch 与模型而定）
- **CUDA**：12.x（与 PyTorch / vLLM 版本匹配即可）

## 技术栈概览

| 组件 | 说明 |
|------|------|
| Python | 3.12 |
| 环境管理 | [uv](https://docs.astral.sh/uv/)，虚拟环境目录 `env_sft` |
| 微调 | [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) + LoRA + [accelerate](https://github.com/huggingface/accelerate) |
| 基座模型 | Qwen3-4B-Instruct-2507 |
| 爬虫（可选） | [aiotieba](https://github.com/Starry-OvO/aiotieba) |
| 批量推理 | vLLM |
| 分类标注（数据构建） | 兼容 OpenAI API 的客户端（如 Claude 等） |

## 安装命令

```bash
uv venv env_sft --python 3.12
source env_sft/bin/activate

# 训练
uv pip install 'llamafactory[metrics]' accelerate

# 数据构建 / 分类脚本
uv pip install openai tenacity tqdm python-dotenv pyyaml

# 推理与评估、绘图
uv pip install vllm json-repair seaborn matplotlib
```

## 数据处理与分类常用依赖

| 包 | 用途 |
|----|------|
| `openai` | LLM API |
| `tenacity` | 重试 |
| `tqdm` | 进度条 |
| `python-dotenv` | `.env` |
| `pyyaml` | YAML 配置 |

标准库：`json`, `re`, `os`, `glob`, `time`, `pathlib`, `typing`, `datetime`, `difflib`, `concurrent.futures`, `multiprocessing` 等。

## 相关文档

- 端到端命令顺序见 [`reproduction.md`](reproduction.md)
- 双卡训练 tmux 步骤见 [`../analysis/training_execution.md`](../analysis/training_execution.md)
