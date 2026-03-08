# source data

- from paper: COIG-CQIA
  - https://huggingface.co/datasets/m-a-p/COIG-CQIA/tree/main/ruozhiba
- from github:
  - https://github.com/Leymore/ruozhiba
- extracted from original tieba
  - https://tieba.baidu.com/p/10354221105
  - https://tieba.baidu.com/p/10130417881
- other refs
  - https://github.com/nick7nlp/evol-ruozhiba
  - https://zhuanlan.zhihu.com/p/698564550

* [m-a-p/COIG-CQIA · Datasets at Hugging Face](https://link.zhihu.com/?target=https%3A//huggingface.co/datasets/m-a-p/COIG-CQIA)

```python
from datasets import load_dataset

dataset = load_dataset("m-a-p/COIG-CQIA", 'ruozhiba') ## 弱智吧数据只是其中一个子集

str(dataset)

# DatasetDict({
#    train: Dataset({
#        features: ['instruction', 'input', 'output', 'task_type', 'domain', 'metadata',
#                     'answer_from', 'human_verified', 'copyright'],
#        num_rows: 240 ## 一共才240行
#    })
#})
```

数据里面有很多列，我们只需要查看代表问答的两列。

```python
# 将 train 数据集转换为 DataFrame
import pandas as pd
train_df = pd.DataFrame(dataset['train'])

# 查看 DataFrame
df = train_df[['instruction', 'output']]
df.head()
```
