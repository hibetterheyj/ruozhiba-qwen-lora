# data/LLaMA-Factory — 备份副本

此文件夹是 `LLaMA-Factory/data/` 中 SFT 数据集的备份，**非**脚本直接输出目录。

`scripts/data/build_sft_data.py` 会将去重后的贴吧数据转换为 ShareGPT 格式，输出到 `LLaMA-Factory/data/`，同时在 `LLaMA-Factory/data/dataset_info.json` 中注册数据集。此处的文件由手动复制而来，仅作存档用途。

## 文件说明

| 文件 | 说明 |
|------|------|
| `data/ruozhiba_all.json` | 全量训练集 (2018-2025) |
| `data/ruozhiba_last3.json` | 近三年训练集 (2023-2025) |
| `data/dataset_info.json` | LLaMA-Factory 数据集注册配置 |
