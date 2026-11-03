# week04 数据准备说明

对 `data/corpus_general.txt` 完成三件事：加终止符、切分训练/测试集、构建词表。
流水线脚本可一键复现：

```bash
python util/prepare_data.py     # 生成 train.txt / test.txt / vocab.txt / stats.json
python code/verify_data.py      # 校验产物（行数、终止符、词表、OOV、编解码往返）
```

## 1. 产物

| 文件 | 说明 |
| --- | --- |
| `data/corpus_general.txt` | 原始语料，395,286 行，UTF-8（带 BOM） |
| `data/train.txt` | 训练集，每行末尾一个 `<eos>` |
| `data/test.txt` | 测试集，每行末尾一个 `<eos>` |
| `data/vocab.txt` | 字符级词表，每行一个 token，**行号即 token id** |
| `data/stats.json` | 行数、字数、token 数、OOV 率等统计 |
| `code/tokenizer.py` | 字符级分词器 `CharTokenizer`（训练/加载/编码/解码） |
| `code/util` | 数据准备脚本（可配置比例、种子、最小频次、去重） |
| `code/verify_data.py` | 数据校验脚本 |

## 2. 处理决策

**终止符加在哪。** 语料每行是一条独立文本（一句提问、一次发言或一条新闻），
因此终止符加在**每行末尾**：`问题？<eos>`。这样一行就是一个完整样本，
训练时模型学到“`<eos>` = 文本结束”，推理时以它作为停止条件。
语料没有句子级空行，无需额外处理；原始行内不会出现 `<eos>` 字面量（脚本会检查并拒绝）。
未加 `<bos>`：因果语言模型以 `<eos>` 分界即可，`<bos>` 仅作为保留 token。

**怎么切分。** 先按行随机打乱（`seed=42`），再按 9:1 切分，训练集 355,757 行、
测试集 39,529 行。切分作用在**加 `<eos>` 之前**的原始行上，因此两边都只多一个结尾符号。
换用其它比例/种子：`python util/prepare_data.py --train-ratio 0.95 --seed 123`。

**词表怎么建。** 字符级（中文按字切分最自然，词表小、无 OOV 词问题）。
只统计**训练集**字符频次，避免测试集信息泄漏；`min_freq=1` 保留训练集中出现的全部字符。
特殊 token 固定占前 4 位，其余按频次降序排列（同频按字符编码升序，保证可复现）：

| id | token | 用途 |
| --- | --- | --- |
| 0 | `<pad>` | 批次对齐填充 |
| 1 | `<unk>` | 未登录字（测试集出现、训练集未出现的字） |
| 2 | `<bos>` | 保留未用 |
| 3 | `<eos>` | 终止符，每行末尾 |

编码时，文本中的 `<eos>` 等特殊 token 字面量会被识别为一个整体 id，而不是拆成字符；
解码时默认跳过特殊 token。`train.txt` 中的行可直接 `encode` 得到含 `<eos>` 的 id 序列。

## 3. 结果统计

| 项目 | 数值 |
| --- | --- |
| 原始语料 | 395,286 行 / 5,626,360 字 |
| 训练集 | 355,757 行 / 5,063,705 字 / **5,419,462 token**（含 `<eos>`） |
| 测试集 | 39,529 行 / 562,655 字 / **602,184 token**（含 `<eos>`） |
| 词表大小 | 5,262（4 个特殊 token + 5,258 个字符） |
| 测试集 OOV | 45 token / 602,184 = **0.0075%** |
| 平均行长 | 14.23 字 |

## 4. 注意：语料重复率 37.7%

原始语料有 149,057 行是完全重复的（占 37.7%），例如 `欢迎发表评论我要评论`。
按行随机切分时，同一句话可能同时出现在训练集和测试集，测试指标会因此偏高。
如需更严格的泛化评估，先去重再切分：

```bash
python util/prepare_data.py --dedup   # 246,229 行，词表 5,241，OOV 0.0166%
```

## 5. 在模型代码中使用

```python
import sys
sys.path.insert(0, "code")
from tokenizer import CharTokenizer

tok = CharTokenizer.from_file("data/vocab.txt")
ids = tok.encode("今天天气怎么样？", add_bos=True, add_eos=True)
print(ids)                      # [2, ...字符 id..., 3]
print(tok.decode(ids))          # 今天天气怎么样？
```

按行读取并编码（`train.txt` 自带 `<eos>`，不要再 `add_eos`）：

```python
with open("data/train.txt", encoding="utf-8") as f:
    sequences = [tok.encode(line.rstrip("\n")) for line in f]
```

按固定长度打包成训练样本（跨行连续，`<eos>` 作为分隔）：

```python
import torch
from torch.utils.data import Dataset

class LMDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.ids = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                self.ids.extend(tokenizer.encode(line.rstrip("\n")))

    def __len__(self):
        return max(0, (len(self.ids) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        x = self.ids[start:start + self.seq_len]
        y = self.ids[start + 1:start + 1 + self.seq_len]
        return torch.tensor(x), torch.tensor(y)
```
