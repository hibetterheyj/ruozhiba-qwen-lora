# Phase 2.5 训练执行手册

> **目标**: 双卡并行训练 Qwen3-4B LoRA SFT，Rank=8 (GPU 0) + Rank=16 (GPU 1)，各 7 epochs

---

## 前置检查

```bash
# 1. 确认两张 GPU 空闲
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader

# 2. 确认虚拟环境可用
source /root/code/llm_ruozhiba/env_sft/bin/activate
python -c "import llamafactory; print('OK')"

# 3. 确认数据集已注册
cat /root/code/llm_ruozhiba/LLaMA-Factory/data/dataset_info.json | python -m json.tool | grep ruozhiba

# 4. wandb 已配置 (report_to: wandb), 确认已登录:
#    wandb login
```

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `configs/qwen3_4b_base.yaml` | 共享基础配置 (BS=16×2, 7 epochs, seed=42) |
| `scripts/run_training.sh` | 启动脚本，通过 CLI 参数注入 rank/alpha/output_dir |

---

## 实验矩阵

| Run | GPU | Rank | Alpha | Epochs | Output Dir |
|-----|-----|------|-------|--------|------------|
| A | 0 | 8 | 16 | 7 | `saves/qwen3-4b/lora/r8` |
| B | 1 | 16 | 32 | 7 | `saves/qwen3-4b/lora/r16` |

**产出 Checkpoint** (共 14 个，评估时取 epoch ≥ 3):

```
saves/qwen3-4b/lora/r8/checkpoint-{82,164,246,328,410,492,574}
saves/qwen3-4b/lora/r16/checkpoint-{82,164,246,328,410,492,574}
```

---

## tmux 执行步骤

```bash
# 1. 创建 tmux session
tmux new-session -s train

# 2. 在第一个 pane 启动 Run A (GPU 0, rank=8)
bash /root/code/llm_ruozhiba/scripts/run_training.sh 0 8 2>&1 | tee /root/code/llm_ruozhiba/logs/run_a_r8.log

# 3. 分割窗口，启动 Run B (GPU 1, rank=16)
#    按 Ctrl+B 然后 % (垂直分割) 或 " (水平分割)
bash /root/code/llm_ruozhiba/scripts/run_training.sh 1 16 2>&1 | tee /root/code/llm_ruozhiba/logs/run_b_r16.log

# 4. 断开 tmux (训练继续在后台运行)
#    按 Ctrl+B 然后 d

# 5. 重新连接查看进度
tmux attach -t train
```

### 快速一键启动 (适合直接复制粘贴)

```bash
# 创建日志目录
mkdir -p /root/code/llm_ruozhiba/logs

# 启动 tmux 双窗口训练
tmux new-session -d -s train \
  "bash /root/code/llm_ruozhiba/scripts/run_training.sh 0 8 2>&1 | tee /root/code/llm_ruozhiba/logs/run_a_r8.log" \; \
  split-window -h \
  "bash /root/code/llm_ruozhiba/scripts/run_training.sh 1 16 2>&1 | tee /root/code/llm_ruozhiba/logs/run_b_r16.log" \; \
  attach
```

---

## 监控

```bash
# 实时 GPU 监控 (另开终端)
watch -n 1 nvidia-smi

# 查看训练日志
tail -f /root/code/llm_ruozhiba/logs/run_a_r8.log
tail -f /root/code/llm_ruozhiba/logs/run_b_r16.log

# 检查 checkpoint 是否生成
ls -la /root/code/llm_ruozhiba/LLaMA-Factory/saves/qwen3-4b/lora/r8/
ls -la /root/code/llm_ruozhiba/LLaMA-Factory/saves/qwen3-4b/lora/r16/
```

---

## 训练参数摘要

| 参数 | 值 | 来源 |
|------|-----|------|
| 模型 | Qwen3-4B-Instruct-2507 | 基座 |
| 方法 | LoRA (target=all) | base.yaml |
| 数据集 | ruozhiba_all (2645 train / 140 eval) | base.yaml |
| 模板 | qwen3_nothink | base.yaml |
| Batch Size | 16 (×2 grad_accum = 有效 32) | Phase 2.4 压测 + OOM 修正 |
| 学习率 | 1e-4, cosine, warmup=0.1 | base.yaml |
| Epochs | 7 | base.yaml |
| Seed | 42 | base.yaml |
| 精度 | bf16 | base.yaml |
| Eval | 每 100 steps, val_size=0.05 | base.yaml |
| Save | 每 epoch, 不限数量 | base.yaml |

---

## 训练完成后

1. 检查 Loss 曲线图: `saves/qwen3-4b/lora/r{8,16}/training_loss.png`
2. 对比 eval_loss 趋势，选择最优 checkpoint (epoch ≥ 3)
3. 进入 Phase 3 评估流程
