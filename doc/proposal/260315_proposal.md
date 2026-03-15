针对你两张显卡的硬件配置和 Qwen3 系列模型的微调需求，我为你制定了这份详细的四阶段开发计划。该计划旨在通过严谨的数据预处理防止污染，并利用并行实验高效验证模型尺寸与数据规模对“幽默逻辑解构”能力的影响。

---

## Phase 1：数据工程与深度清洗 (Day 1-2)

本阶段的核心是确保测试集（CQIA）的纯净性，并构建标准化的指令微调数据集。

### 1.1 全局去重与防止污染 (Data Sanitization)

* **精确去重**：使用 `hashlib.md5` 对 `ruozhiba_cqia_classified.json` 中的所有 `instruction` 生成指纹存入 `set`。
* **模糊去重**：针对贴吧数据，调用 `difflib.SequenceMatcher(None, a, b).ratio()`。
* **阈值设定**：相似度 $> 0.9$ 的文本直接剔除，确保测试集在训练过程中是“未见”的。


* **脚本实现**：利用 `concurrent.futures.ProcessPoolExecutor` 多进程加速对数千条数据的循环比对。

### 1.2 数据格式化 (ShareGPT)

* **字段合并**：将 `classification` 对象中的 `thought_process` 和 `top3_categories` 序列化为 JSON 字符串，作为 `assistant` 角色的输出内容。
* **元数据注入**：在 `system` 提示词中加入时间戳标签（如：`"【年份：2025】你是一个弱智吧解析专家..."`），帮助模型学习梗的时效性逻辑。
* **数据集划分**：
* `train_last3_years.jsonl`：包含 2023-2025 年去重后的数据。
* `train_all_years.jsonl`：包含 2018-2025 年全量去重数据。



---

## Phase 2：双卡并行训练实验 (Day 3-5)

利用变量控制法，同步推进模型尺寸（1.5B vs 8B）和数据集范围（近 3 年 vs 全量）的实验。

### 2.1 实验环境配置

* **注册数据集**：在 `LLaMA-Factory/data/dataset_info.json` 中定义 `ruozhiba_last3` 和 `ruozhiba_all`。
* **并行启动方案**：
* **GPU 0**: 运行 **Run A** (Qwen3-1.5B + Last 3y) -> 结束后启动 **Run C** (Qwen3-8B + All)。
* **GPU 1**: 运行 **Run B** (Qwen3-8B + Last 3y)。



### 2.2 核心训练参数设计

| 参数 | 设定值 | 理由 |
| --- | --- | --- |
| **Finetuning Type** | `lora` | 兼顾效率与性能。 |
| **LoRA Target** | `all` | 作用于所有线性层，提升复杂 JSON 格式的学习效果。 |
| **LoRA Rank** | `16` | 较默认值 8 更高，适合结构化输出任务。 |
| **Template** | `qwen` | 必须与基座模型提示词模板一致。 |
| **Learning Rate** | `1e-4` | LoRA SFT 的标准推荐值。 |
| **BF16** | `true` | 利用显卡精度提升加速训练。 |

---

## Phase 3：定量自动化评测 (Day 6-7)

对训练产出的三个模型（Run A, B, C）进行标准化能力测试。

### 3.1 推理脚本编写

* 使用 `transformers` 加载微调后的 Adapter 权重。
* 针对 CQIA 测试集生成 240 条预测结果，保存为 `results_model_x.json`。

### 3.2 自动化指标计算 (Metrics)

1. **JSON 遵循率**：利用 `re.search(r'\{.*\}', output, re.DOTALL)` 提取 JSON 部分，计算 `json.loads()` 的成功率。
2. **Top-1 准确率**：比较模型给出的第一个 `category` 与金标准是否一致。
3. **Top-3 Hit Rate**：计算金标准类别是否落在模型预测的前三个候选中。

---

## Phase 4：定性分析与裁判评估 (Day 8-10)

深入分析模型的“解构能力”并完成最终报告。

### 4.1 LLM-as-a-Judge (高级评估)

* **逻辑对比**：抽取 20 条典型错误案例，输入 DeepSeek/Claude。
* **评分维度**：
* **逻辑准确度**（1-10）：是否识别出段子中的逻辑谬误？
* **深度解构**（1-10）：是否解释了梗的社会背景或语言学机制？


* **自动化**：编写脚本循环调用 API，并将结果保存至 CSV 进行可视化对比。

### 4.2 复盘与总结

* **结论验证**：对比 Run B 与 Run C，验证“老数据（2018-2020）是否对现代语境下的分类产生负面噪音”。
* **Before vs After 展示**：选取 3 个经典段子，直观展示微调前后的输出差异（由胡言乱语变为结构化解析）。
