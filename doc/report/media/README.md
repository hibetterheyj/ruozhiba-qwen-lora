# 报告配图目录

将实验报告 [`../lab3_report.md`](../lab3_report.md) 引用的图片放在本目录，文件名与报告一致：

- `fig1_train_eval_loss.png` … `fig8_all_vs_last3.png`

生成脚本通常为 `scripts/viz/eval_metrics.py` 等；`scripts/viz/update_report_media.py` 会按既定映射从 `results/charts/`、`results/heatmaps/`、`results/confusion_matrices/` 复制对应 PNG 到本目录。

当前仓库根目录 `readme.md` 已改为直接引用 `results/.../*.png`，本目录主要保留给报告静态配图使用。
