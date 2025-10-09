---
type: "code-note"
tags: [nlp, transformer, pytorch, text-classification]
status: "done"
model: "Custom Transformer Encoder"
dataset: "IMDB"
task: "Sentiment Analysis"
---
ps: 自己的电脑用的是GPU，1分钟左右一轮，显卡4060，显存8GB。

------
我们来详细分析一下这段使用 PyTorch 实现的基于 Transformer 的 IMDB 电影评论情感分类代码。我们将逐步分析每个部分，重点关注数据处理和模型结构。

**1. 导入库和基本设置**

```python
import torch                     # PyTorch 核心库
import torch.nn as nn            # 神经网络模块 (层, 激活函数, 损失函数等)
import torch.optim as optim      # 优化器模块 (Adam, SGD 等)
from torch.utils.data import Dataset, DataLoader # 数据集和数据加载工具
import matplotlib.pyplot as plt  # 用于绘图 (损失曲线, 混淆矩阵等)
from sklearn.metrics import accuracy_score, confusion_matrix # 评估指标 (准确率, 混淆矩阵)
import seaborn as sns            # 用于绘制更美观的图表 (混淆矩阵热力图)
import pandas as pd              # 数据处理库 (虽然这里没直接用，但常用于数据分析)
import time                      # 用于计时 (计算训练耗时)
import re                        # 正则表达式库 (用于文本清洗)
import os                        # 操作系统接口 (文件路径操作, 创建目录等)
import tarfile                   # 用于处理 .tar.gz 压缩文件
import urllib.request            # 用于从 URL 下载文件
import random                    # 用于生成随机数 (打乱数据, 设置种子)
import numpy as np               # NumPy 库 (用于数值计算, 设置种子)
from collections import Counter   # 用于计数 (统计词频, 预测分布)
from torch.nn.utils.rnn import pad_sequence # 用于将不同长度的序列填充到相同长度
from tqdm import tqdm             # 用于显示进度条 (美化循环过程)

# 设置随机种子以便结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# 解释: 为了让代码每次运行时产生相同的结果 (例如，模型初始化权重、数据打乱顺序等)，
# 我们固定了 Python 内置 random, NumPy 和 PyTorch 的随机数生成器的种子。
# 如果 CUDA 可用，也设置 CUDA 的随机种子。这对于调试和复现实验结果非常重要。

# 设置数据集下载路径为当前文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)
# 解释: 这段代码确定了存储下载数据集的目录。
# 首先获取当前脚本所在的目录 (如果作为脚本运行) 或当前工作目录 (如果在交互式环境运行)。
# 然后在当前目录下创建一个名为 "data" 的子目录用于存放数据。
# `os.makedirs(data_dir, exist_ok=True)` 会创建目录，如果目录已存在则不会报错。

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"数据集将下载到: {data_dir}")
# 解释: 检查系统中是否有可用的 NVIDIA GPU (通过 `torch.cuda.is_available()`)。
# 如果有，则将 `device` 设置为 'cuda'，模型和数据将转移到 GPU 上进行计算，速度更快。
# 如果没有，则使用 'cpu'。打印出所使用的设备和数据目录。

if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
# 解释: 如果使用了 CUDA，打印出 PyTorch 编译时使用的 CUDA 版本。
```

**2. 数据下载与解压 (`download_and_extract_imdb`)**

```python
# 手动下载并处理IMDB数据集
def download_and_extract_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz" # 数据集下载地址
    dataset_path = os.path.join(data_dir, "aclImdb_v1.tar.gz") # 本地保存压缩文件的路径
    extracted_path = os.path.join(data_dir, "aclImdb") # 解压后数据的根目录

    # 检查解压后的目录是否存在
    if not os.path.exists(extracted_path):
        # 如果解压目录不存在，再检查压缩文件是否存在
        if not os.path.exists(dataset_path):
            print(f"下载IMDB数据集到 {dataset_path}...")
            # 使用 urllib.request 下载文件
            urllib.request.urlretrieve(url, dataset_path)
            print("下载完成！")

        print("解压数据集...")
        # 使用 tarfile 打开 .tar.gz 文件 ('r:gz' 模式)
        with tarfile.open(dataset_path, 'r:gz') as tar:
            # 将压缩包内容解压到指定的 data_dir 目录下
            tar.extractall(path=data_dir)
        print("解压完成！")
    else:
        # 如果解压目录已存在，说明数据已准备好
        print("IMDB数据集已存在，跳过下载和解压步骤。")

    # 返回解压后数据的路径
    return extracted_path

# 解释: 这个函数负责准备 IMDB 数据集。
# 1. 定义了数据集的 URL、本地压缩包路径和解压后的路径。
# 2. 首先检查解压后的文件夹 (`aclImdb`) 是否存在。如果存在，就跳过下载和解压。
# 3. 如果解压文件夹不存在，接着检查压缩包 (`aclImdb_v1.tar.gz`) 是否存在。
# 4. 如果压缩包也不存在，则使用 `urllib.request.urlretrieve` 从指定的 URL 下载文件到 `dataset_path`。
# 5. 下载完成后（或压缩包原本就存在），使用 `tarfile` 库打开并解压 `.tar.gz` 文件到 `data_dir`。解压后会生成 `aclImdb` 文件夹。
# 6. 最后返回解压后数据的路径 `extracted_path`。
```

**3. 数据预处理 (`clean_text`, `basic_english_tokenizer`)**

```python
# 简单的分词器
def basic_english_tokenizer(text):
    # 简易版的basic_english tokenizer
    text = text.lower() # 转换为小写
    # 将非字母、非数字、非空白符的字符替换为空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 将连续的多个空格合并为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾空格，并按空格分割成单词列表
    return text.strip().split()

def clean_text(text):
    text = text.lower() # 转换为小写
    text = re.sub(r'<br />', ' ', text) # 移除 HTML 换行符 <br />
    # 将非字母、非数字、非空白符的字符替换为空格 (移除非单词字符)
    text = re.sub(r'[^\w\s]', ' ', text)
    # 将连续的多个空格合并为一个空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首尾多余空格
    return text.strip()

# 解释: 这两个函数用于文本的初步清洗和分词。
# `clean_text(text)`:
#   - 将文本全部转为小写，以保证大小写不敏感（例如 "Good" 和 "good" 被视为同一个词）。
#   - 使用正则表达式 `re.sub` 移除 IMDB 数据中常见的 HTML 换行标签 `<br />`，替换为空格。
#   - 使用 `[^\w\s]` 匹配所有不是字母、数字或空白字符的字符（即标点符号等），并将它们替换为空格。
#   - 使用 `\s+` 匹配一个或多个连续的空白字符，并将它们合并成一个空格。
#   - `strip()` 移除文本开头和结尾可能存在的空格。
# `basic_english_tokenizer(text)`:
#   - 首先进行与 `clean_text` 类似的清洗操作（小写、去标点、合并空格）。
#   - 最后使用 `text.strip().split()` 将清洗后的字符串按空格分割，得到一个单词（token）列表。这是一个非常基础的分词器。
```

**4. 数据集类 (`IMDBDataset`)**

```python
# 数据预处理和加载
class IMDBDataset(Dataset): # 继承自 torch.utils.data.Dataset
    def __init__(self, texts, labels, vocab, tokenizer, max_len=512):
        self.texts = texts         # 存储所有文本评论的列表
        self.labels = labels       # 存储所有对应标签的列表 (0或1)
        self.vocab = vocab         # 存储词汇表 (一个字典，word -> index)
        self.tokenizer = tokenizer # 存储使用的分词器函数
        self.max_len = max_len     # 限制序列的最大长度

    def __len__(self):
        # 返回数据集中样本的总数
        return len(self.texts)

    def __getitem__(self, idx):
        # 根据索引 idx 获取单个样本
        text = self.texts[idx]    # 获取第 idx 条文本
        label = self.labels[idx]  # 获取第 idx 个标签

        # 使用分词器将文本转换为 token 列表
        tokens = self.tokenizer(text)

        # 如果 token 数量超过最大长度，则截断
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        # 将 token 转换为词汇表中的索引
        # 使用 vocab.get(token, self.vocab['<unk>'])
        # 如果 token 在词汇表中，则返回其索引
        # 如果不在 (称为 OOV, Out-Of-Vocabulary)，则返回特殊标记 '<unk>' (unknown) 的索引
        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        # 返回一个字典，包含处理后的数据
        return {
            # 将 token 索引列表转换为 PyTorch 张量 (LongTensor)
            'text': torch.tensor(token_ids, dtype=torch.long),
            # 存储原始 token 数量 (截断后的长度)
            'length': len(token_ids),
            # 将标签转换为 PyTorch 张量 (LongTensor)
            'label': torch.tensor(label, dtype=torch.long)
        }

# 解释: 这个类是 PyTorch 中用于处理数据集的标准方式。
# - 它继承自 `torch.utils.data.Dataset`，必须实现 `__len__` 和 `__getitem__` 两个方法。
# - `__init__`: 构造函数，接收文本列表、标签列表、构建好的词汇表、分词器函数和最大序列长度作为参数，并将它们存储为类的属性。
# - `__len__`: 返回数据集中样本的总数，DataLoader 需要这个信息。
# - `__getitem__(idx)`: 定义了如何获取和处理索引为 `idx` 的单个数据样本。
#   1. 获取原始文本和标签。
#   2. 调用 `tokenizer` 对文本进行分词。
#   3. 对分词后的 `tokens` 进行长度截断，确保不超过 `max_len`。
#   4. **核心步骤**: 遍历 `tokens`，在 `vocab` 字典中查找每个 token 对应的索引。如果 token 不在词汇表中（即训练时未见过或频率过低被过滤掉的词），则使用预定义的 `<unk>`（未知词）标记的索引。这步将文本序列转换成了模型可以理解的数字序列（索引）。
#   5. 将处理后的 `token_ids`（整数列表）和 `label`（整数）转换为 PyTorch 张量 `torch.tensor`。`dtype=torch.long` 是因为 Embedding 层的输入和损失函数的标签输入通常要求是长整型。
#   6. 返回一个包含 `text`（token 索引张量）、`length`（序列实际长度）和 `label`（标签张量）的字典。DataLoader 会将这些字典收集起来组成一个批次。
```

**5. 手动加载 IMDB 数据 (`load_imdb_data_manually`)**

```python
# 加载IMDB数据集（不使用torchtext）
def load_imdb_data_manually():
    # 1. 下载并解压数据集
    dataset_dir = download_and_extract_imdb() # 获取解压后的数据路径

    tokenizer = basic_english_tokenizer # 指定使用的分词器

    # 2. 加载训练数据
    print("加载训练数据...")
    train_texts, train_labels = [], [] # 初始化列表存储训练文本和标签

    # 加载正面评价 (positive, label=1)
    pos_dir = os.path.join(dataset_dir, "train", "pos") # 正面训练评论目录
    # 使用 tqdm 显示加载进度
    for filename in tqdm(os.listdir(pos_dir), desc="加载正面训练评价"):
        if filename.endswith('.txt'): # 确保只读取 .txt 文件
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read() # 读取文件内容
            train_texts.append(clean_text(text)) # 清洗文本后添加到列表
            train_labels.append(1) # 添加标签 1

    # 加载负面评价 (negative, label=0)
    neg_dir = os.path.join(dataset_dir, "train", "neg") # 负面训练评论目录
    for filename in tqdm(os.listdir(neg_dir), desc="加载负面训练评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            train_texts.append(clean_text(text)) # 清洗文本
            train_labels.append(0) # 添加标签 0

    # 3. 加载测试数据 (过程与加载训练数据类似)
    print("加载测试数据...")
    test_texts, test_labels = [], []

    # 加载正面测试评价
    pos_dir = os.path.join(dataset_dir, "test", "pos")
    for filename in tqdm(os.listdir(pos_dir), desc="加载正面测试评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(clean_text(text))
            test_labels.append(1)

    # 加载负面测试评价
    neg_dir = os.path.join(dataset_dir, "test", "neg")
    for filename in tqdm(os.listdir(neg_dir), desc="加载负面测试评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(clean_text(text))
            test_labels.append(0)

    # 4. 创建词汇表 (Vocabulary)
    print("构建词汇表...")
    counter = Counter() # 使用 Counter 对象统计词频
    # 遍历所有训练文本
    for text in tqdm(train_texts, desc="统计词频"):
        # 使用分词器分词，并更新 Counter 中的词频
        counter.update(tokenizer(text))

    # 初始化词汇表，包含两个特殊 token:
    # '<pad>': 用于填充 (padding) 较短序列，使其与批次中最长序列等长。通常索引为 0。
    # '<unk>': 用于表示未在词汇表中出现的词 (Unknown)。通常索引为 1。
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2 # 从索引 2 开始分配给其他词
    # 遍历统计好的词频 (counter.items() 返回 (word, count) 对)
    for word, count in counter.items():
        # 只将词频大于等于 5 的词加入词汇表
        # 这是一种常见的过滤低频词的方法，可以减小词汇表大小，去除噪音。
        if count >= 5:
            vocab[word] = idx
            idx += 1

    print(f"词汇表大小: {len(vocab)}") # 打印最终词汇表包含的词数量

    # 5. 打乱数据集顺序
    # 将文本和标签打包成 (text, label) 对
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined) # 使用 random.shuffle 原地打乱列表顺序
    # 解包，得到打乱后的文本和标签列表
    train_texts, train_labels = zip(*combined)

    combined = list(zip(test_texts, test_labels))
    random.shuffle(combined)
    test_texts, test_labels = zip(*combined)
    # 解释: 打乱数据顺序对于训练很重要，可以防止模型学到数据原始顺序带来的偏差，
    # 使每个批次的数据更具代表性。训练集和测试集都需要打乱。

    # 6. 创建数据集对象
    # 使用前面定义的 IMDBDataset 类创建训练集和测试集实例
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, tokenizer)

    # 返回创建好的数据集对象和词汇表
    return train_dataset, test_dataset, vocab

# 解释: 这个函数是数据准备的核心环节，它不依赖像 `torchtext` 这样的库，而是手动完成了所有步骤：
# - 下载解压数据。
# - 逐个读取训练和测试文件，进行文本清洗 (`clean_text`)，并存储文本和对应标签。
# - **构建词汇表**: 这是关键一步。它统计了训练集中所有单词的出现频率，并只保留了出现次数达到一定阈值（这里是 5 次）的单词。同时，加入了 `<pad>` 和 `<unk>` 两个特殊符号。这个词汇表 `vocab` 是一个字典，将单词映射到唯一的整数索引。
# - 打乱数据，确保训练的随机性。
# - 使用 `IMDBDataset` 类将处理好的文本列表、标签列表、词汇表和分词器封装成 PyTorch 的 Dataset 对象，方便后续 DataLoader 使用。
```

**6. Transformer 编码器层 (`TransformerEncoderLayer`)**

```python
# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        # embed_dim: 输入/输出的特征维度 (词嵌入维度)
        # num_heads: 多头注意力机制中的头数
        # ff_dim: 前馈网络中间层的维度
        # dropout: Dropout 比率
        super(TransformerEncoderLayer, self).__init__()

        # 1. 多头自注意力 (Multi-Head Self-Attention)
        self.self_attn = nn.MultiheadAttention(
            embed_dim,      # 嵌入维度
            num_heads,      # 注意力头数
            dropout=dropout # 注意力权重上的 dropout
            # batch_first=False (默认) - 输入形状预期为 (SeqLen, Batch, EmbedDim)
        )

        # 2. 前馈网络 (Feed Forward Network)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), # 线性层 1: 从 embed_dim 扩展到 ff_dim
            nn.ReLU(),                    # ReLU 激活函数
            nn.Dropout(dropout),          # Dropout
            nn.Linear(ff_dim, embed_dim)  # 线性层 2: 从 ff_dim 压缩回 embed_dim
        )

        # 3. 层归一化 (Layer Normalization)
        self.norm1 = nn.LayerNorm(embed_dim) # 应用在自注意力之后
        self.norm2 = nn.LayerNorm(embed_dim) # 应用在前馈网络之后

        # 4. Dropout
        self.dropout = nn.Dropout(dropout) # 应用在残差连接中

    def forward(self, x, mask=None):
        # x: 输入序列, 形状预期为 (SeqLen, Batch, EmbedDim)
        # mask: Padding mask, 形状预期为 (Batch, SeqLen), True 表示是 padding 位置

        # --- 子层 1: 多头自注意力 + 残差连接 + 层归一化 ---
        # 计算多头自注意力
        # query, key, value 都来自输入 x
        # key_padding_mask=mask 告诉注意力机制忽略 mask 中为 True 的位置 (padding)
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        # 残差连接 (Add): 将注意力输出加到原始输入 x 上
        # Dropout 应用于注意力输出
        x = x + self.dropout(attn_output)
        # 层归一化 (Norm)
        x = self.norm1(x)

        # --- 子层 2: 前馈网络 + 残差连接 + 层归一化 ---
        # 通过前馈网络
        ff_output = self.feed_forward(x)
        # 残差连接 (Add): 将前馈网络输出加到上一层的结果 x 上
        # Dropout 应用于前馈网络输出
        x = x + self.dropout(ff_output)
        # 层归一化 (Norm)
        x = self.norm2(x)

        return x # 返回编码器层的输出, 形状仍为 (SeqLen, Batch, EmbedDim)

# 解释: 这是构成 Transformer Encoder 的基本单元。一个 Encoder Layer 包含两个主要的子层：
# 1.  **多头自注意力 (Multi-Head Self-Attention)**:
#     -   允许模型在处理序列中的每个词时，同时关注序列中的其他所有词，并计算它们之间的相关性（注意力权重）。
#     -   "多头"表示这种注意力计算过程并行地进行多次（`num_heads`次），每个头学习不同的关注模式，然后将结果合并，增强了模型的表达能力。
#     -   `self.self_attn(x, x, x, key_padding_mask=mask)`: 输入 `x` 同时作为查询 (Query)、键 (Key) 和值 (Value) 进行自注意力计算。`key_padding_mask` 用于确保模型不会关注到输入序列中的填充部分（`<pad>` token）。
# 2.  **前馈网络 (Feed Forward Network)**:
#     -   这是一个简单的全连接神经网络，独立地应用于序列中的每个位置（每个词的表示向量）。
#     -   通常由两个线性层和一个非线性激活函数（如 ReLU）组成。`nn.Sequential` 将这些层按顺序组合起来。
#     -   它对自注意力层的输出进行进一步的非线性变换，增加模型的拟合能力。

# 每个子层的输出都经过了 **残差连接 (Residual Connection)** 和 **层归一化 (Layer Normalization)**：
# -   **残差连接**: 将子层的输入 `x` 直接加到子层的输出上 (`x = x + sublayer(x)`)。这有助于缓解深度网络中的梯度消失问题，使得模型更容易训练。
# -   **层归一化**: 对每个样本的特征进行归一化，稳定训练过程，加速收敛。它与批归一化 (Batch Normalization) 不同，是在特征维度上进行归一化，而不是在批次维度上。
# -   **Dropout**: 在自注意力和前馈网络的输出以及残差连接中加入 Dropout，随机将一部分神经元的输出置零，是一种有效的正则化手段，防止模型过拟合。

# `forward` 方法清晰地展示了数据的流动过程：输入 `x` -> 自注意力 -> Add & Norm -> 前馈网络 -> Add & Norm -> 输出。
# 注意 PyTorch 的 `nn.MultiheadAttention` 默认期望输入形状是 `(SeqLen, Batch, EmbedDim)`，这与常见的 `(Batch, SeqLen, EmbedDim)` 不同，需要在模型主体中进行转换。
```

## 理解多头注意力机制 (Multi-Head Attention)

### 核心思想：在不同子空间中学习注意力

多头注意力机制的核心思想不是让每个头直接学习输入词向量内部的某几个特定特征维度，而是让**每一个“头”（head）学习在输入序列的不同“表示子空间”（representation subspace）中计算注意力**。

过程如下：

1.  **原始词向量（Input Embedding）**:
    输入是词向量序列，每个词向量维度为 $d_{model}$ (例如 256)。这些向量蕴含丰富的语义信息。

2.  **投影到子空间 (Projection)**:
    对于 $h$ 个注意力头中的每一个头 $i$：
    *   存在三组**独立**的权重矩阵：$W_i^Q$ (查询), $W_i^K$ (键), $W_i^V$ (值)。
    *   原始的输入词向量 $x$ (维度 $d_{model}$) 会被这些独立的矩阵分别线性变换（投影）到**更低维度** ($d_k = d_v = d_{model} / h$) 的子空间中，得到该头对应的查询 ($Q_i$), 键 ($K_i$), 值 ($V_i$) 向量：
        *   $Q_i = x W_i^Q$
        *   $K_i = x W_i^K$
        *   $V_i = x W_i^V$
    *   例如，如果 $d_{model}=256$ 且 $h=8$，则每个 $Q_i, K_i, V_i$ 的维度是 32。

3.  **独立计算注意力**:
    **每个头都在它自己的子空间内独立地**执行标准的缩放点积注意力计算：
    *   头 $i$ 使用它的 $Q_i, K_i$ 计算注意力得分，然后用这些得分去加权它的 $V_i$。
    *   $Attention_i = softmax(\frac{Q_i K_i^T}{\sqrt{d_k}}) V_i$

4.  **学习不同的关系**:
    由于每个头 $i$ 拥有自己独立的投影矩阵 ($W_i^Q, W_i^K, W_i^V$)，它能够学习关注输入序列中不同类型的关系或模式。不同的头可以看作是从原始词向量的不同“视角”或“方面”（通过不同的投影）来学习这些关系。例如：
    *   一个头可能关注句法依赖。
    *   另一个头可能关注语义相似性。
    *   还有一个头可能关注相对位置。

5.  **拼接与最终投影**:
    所有 $h$ 个头的输出结果 $Attention_i$ (每个维度是 $d_v$) 会被拼接（concatenate）在一起，恢复到原始的 $d_{model}$ 维度。然后通常会再通过一个最终的线性投影层（乘以权重矩阵 $W^O$），将拼接后的结果融合，得到多头注意力的最终输出：
    *   $MultiHead(Q, K, V) = Concat(Attention_1, ..., Attention_h) W^O$

**总结单个头的作用：** 一个注意力头是将**整个**词向量投影到一个**特定的、较低维度的子空间**，并在这个子空间内学习**词与词之间**应该如何根据这个子空间的表示来相互关注（计算注意力权重）。

### 为何不同的头能学习不同模式？

虽然所有头接收相同的初始输入 $x$，但它们之所以能学习到不同的注意力模式，关键在于以下几点：

1.  **独立的、随机初始化的权重矩阵**:
    *   这是**最重要**的原因。如上所述，每个头 $i$ 都有**自己独立**的一组线性投影权重矩阵 $W_i^Q, W_i^K, W_i^V$ (以及最终的 $W^O$)。
    *   这些权重矩阵在训练开始时是**随机初始化**的，并且**彼此不同**。
    *   因此，即使输入 $x$ 相同，由于初始权重不同，每个头计算出的 $Q_i, K_i, V_i$ 就是**不同**的。

2.  **不同的梯度路径**:
    *   因为每个头的初始 $Q_i, K_i, V_i$ 不同，它们计算出的注意力得分和加权后的 $Attention_i$ 也不同。
    *   在反向传播计算梯度时，损失函数对每个头内部的独特权重矩阵 ($W_i^Q, W_i^K, W_i^V$) 的梯度也会是**不同**的。
    *   优化器（如 Adam）会根据这些不同的梯度来更新每个头的权重。

3.  **不同的学习“方向”**:
    *   由于初始状态不同，梯度不同，每个头在训练过程中会探索不同的参数空间区域。
    *   它们会各自找到能够降低整体损失的不同方式，从而学习到关注输入序列中不同方面的信息或不同类型的依赖关系。

4.  **Dropout 等正则化技术 (若使用)**:
    *   正则化技术（如 Dropout）会进一步增加随机性，使得不同的头更难学习到完全一样的模式。

**核心机制:** 正是**为每个头使用独立且随机初始化的权重矩阵**打破了对称性，使得即使输入相同，不同的头也能在训练过程中分化，学习捕捉输入数据中不同子空间的特征和关系，使得模型能够同时从多个角度理解输入，获得更强大的表示能力。

**7. Transformer 分类器模型 (`TransformerClassifier`)**

```python
# 定义Transformer分类器
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, max_len=512, dropout=0.1):
        # vocab_size: 词汇表大小 (用于 Embedding 层)
        # embed_dim: 词嵌入和模型内部的特征维度
        # num_heads: 每个 Transformer 层中的注意力头数
        # num_layers: Transformer 编码器层的数量 (深度)
        # ff_dim: Transformer 层中前馈网络的中间维度
        # num_classes: 输出类别的数量 (情感分类是 2 类)
        # max_len: 输入序列的最大长度 (用于位置编码)
        # dropout: Dropout 比率
        super(TransformerClassifier, self).__init__()

        # 1. 词嵌入层 (Token Embedding)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # - vocab_size: 词汇表的大小，决定了嵌入矩阵的第一维。
        # - embed_dim: 每个词向量的维度。
        # - padding_idx=0: 指定索引为 0 的 token (<pad>) 的嵌入向量在训练中不会被更新，并且其梯度始终为零。

        # 2. 位置编码 (Positional Embedding)
        # 使用可学习的位置编码 (Learned Positional Embedding)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))
        # - 创建一个形状为 (max_len, embed_dim) 的参数张量，初始化为全零。
        # - nn.Parameter 表明这是一个模型的可学习参数，会在训练过程中被优化器更新。
        # - 模型将学习到每个位置的最佳表示。

        # 3. Transformer 编码器层堆叠
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers) # 创建 num_layers 个独立的 TransformerEncoderLayer 实例
        ])
        # - nn.ModuleList 是一个列表，可以像普通 Python 列表一样索引，但它能正确地注册其中包含的模块，
        #   确保这些子模块的参数能被 PyTorch 的优化器识别和更新。

        # 4. 分类器头部 (Classifier Head)
        self.classifier = nn.Linear(embed_dim, num_classes)
        # - 一个简单的线性层，将 Transformer 最后一层的输出特征 (经过池化后) 映射到最终的类别分数 (logits)。
        # - 输入维度是 embed_dim (与 Transformer 输出维度一致)。
        # - 输出维度是 num_classes (类别数)。

        # 5. Dropout 层 (用于分类器之前)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths): # lengths 参数在这里没有直接使用，但可以用于更复杂的池化或处理
        # x: 输入的 token 索引序列, 形状为 (Batch, SeqLen)

        # 创建 Padding Mask
        # mask 的形状为 (Batch, SeqLen)，在 padding 位置 (索引为 0) 为 True
        mask = create_padding_mask(x)

        # 1. 获取词嵌入
        # x 形状: (Batch, SeqLen) -> (Batch, SeqLen, EmbedDim)
        x = self.token_embedding(x)

        # 2. 添加位置编码
        batch_size, seq_len = x.size(0), x.size(1)
        # position_embedding 形状: (max_len, embed_dim)
        # 取前 seq_len 行: (seq_len, embed_dim)
        # 使用广播机制加到 x 上:
        # x (Batch, SeqLen, EmbedDim) + pos_embed (SeqLen, EmbedDim) -> (Batch, SeqLen, EmbedDim)
        x = x + self.position_embedding[:seq_len, :]

        # 3. 调整维度以适配 PyTorch Transformer 层
        # (Batch, SeqLen, EmbedDim) -> (SeqLen, Batch, EmbedDim)
        x = x.transpose(0, 1)

        # 4. 通过 Transformer 编码器层
        # 逐层传递数据，每一层的输出作为下一层的输入
        # mask (Batch, SeqLen) 会被传递给每个 MultiheadAttention 层的 key_padding_mask 参数
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # 5. 调整维度回原始格式
        # (SeqLen, Batch, EmbedDim) -> (Batch, SeqLen, EmbedDim)
        x = x.transpose(0, 1)

        # 6. 平均池化 (Mean Pooling over non-padding tokens)
        # mask 形状: (Batch, SeqLen)
        # mask.float() -> (Batch, SeqLen), True 变 1.0, False 变 0.0
        # mask.unsqueeze(-1) -> (Batch, SeqLen, 1)
        # 1 - mask -> 反转 mask，有效位置为 1，padding 位置为 0
        pool_mask = (1.0 - mask.float()).unsqueeze(-1)
        # (x * pool_mask): 将 padding 位置的向量置零
        # .sum(dim=1): 沿着序列长度维度求和, 得到 (Batch, EmbedDim)
        # pool_mask.sum(dim=1): 计算每个序列的有效长度 (排除 padding), 得到 (Batch, 1)
        # .clamp(min=1e-9): 防止除以零
        # 最终得到每个序列的平均向量表示 (Batch, EmbedDim)
        x = (x * pool_mask).sum(dim=1) / pool_mask.sum(dim=1).clamp(min=1e-9)
        # 解释: 这里没有使用像 BERT 那样取 [CLS] token 的输出，而是计算了序列中所有非 <pad> token 输出向量的平均值。
        # 这是一种常用的将变长序列的 Transformer 输出转换为固定大小向量的方法，用于下游分类任务。

        # 7. 分类
        # 应用 Dropout
        x = self.dropout(x)
        # 通过最后的线性分类层得到 logits
        # x 形状: (Batch, EmbedDim) -> (Batch, num_classes)
        logits = self.classifier(x)

        return logits # 返回每个样本属于各个类别的原始分数 (logits)

# 解释: 这是模型的主体结构。
# - `__init__`: 初始化了模型的所有组件：词嵌入层、位置编码参数、一个包含多个 `TransformerEncoderLayer` 的列表、最终的线性分类层和 Dropout 层。
# - `forward(x, lengths)`: 定义了数据在模型中的流动路径：
#   1.  **输入处理**: 获取词嵌入，并加上可学习的位置编码。位置编码给模型提供了单词在序列中的位置信息，这对于 Transformer 至关重要，因为它本身不像 RNN 那样具有内在的顺序处理机制。
#   2.  **维度转换**: 将数据维度从 `(Batch, SeqLen, EmbedDim)` 转换成 `(SeqLen, Batch, EmbedDim)` 以符合 PyTorch `nn.MultiheadAttention` 的要求。
#   3.  **Transformer 编码**: 数据依次通过 `ModuleList` 中的每一个 `TransformerEncoderLayer`。每一层都会对输入序列进行自注意力和前馈网络的处理，逐步提炼序列的表示。Padding Mask 会在每一层的自注意力计算中使用。
#   4.  **维度转换**: 将输出维度转换回 `(Batch, SeqLen, EmbedDim)`。
#   5.  **池化**: 使用 Mean Pooling 将 Transformer 最后一层输出的序列表示（每个 token 一个向量）聚合成一个单一的向量，代表整个输入序列的语义信息。这里巧妙地利用了 padding mask 来确保只对有效的 token 进行平均。
#   6.  **分类**: 将池化得到的序列表示向量通过一个 Dropout 层和最后的线性层，输出每个类别的预测分数（logits）。
```

## Transformer 中的位置编码 (Positional Encoding)

Transformer 模型本身不包含处理序列顺序的循环或卷积结构，因此需要一种方式将单词在序列中的位置信息注入模型。这就是位置编码的作用。主要有以下几种方法：

### 1. 固定的、基于函数计算的位置编码 (Fixed Positional Encoding)

*   **代表作**: 原始 Transformer 论文 ("Attention Is All You Need")
*   **原理**: 使用不同频率的正弦 (sine) 和余弦 (cosine) 函数为每个位置 $pos$ 和每个维度 $i$ 生成一个独特的编码向量 $PE$。公式如下 ($d_{model}$ 是嵌入维度)：
    *   $PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})$
    *   $PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})$
*   **特点**:
    *   **无需学习**: 编码值直接计算得出，不是模型参数。
    *   **可扩展性**: 理论上可以处理任意长度的序列。
    *   **相对位置**: 结构上易于模型学习相对位置关系。
*   **使用**: 将计算出的 $PE$ 向量**加到**对应的词嵌入向量上。

### 2. 可学习的位置嵌入 (Learned Positional Embeddings)

*   **代表作**: BERT, GPT 系列, ==以及你代码中使用的方法 (`nn.Parameter(...)`)==。
*   **原理**: 创建一个嵌入查找表，大小通常为 `(最大序列长度, 嵌入维度)`。表中的每一行是一个可学习的向量，代表一个绝对位置（如位置0, 位置1, ...）。
*   **特点**:
    *   **作为参数学习**: 位置向量是模型参数，在训练中更新。
    *   **灵活性**: 模型可以根据数据自行学习最优的位置表示。
    *   **长度限制**: 通常需要预设最大序列长度 `max_len`，对超过此长度的序列泛化能力可能受限。
*   **使用**: 根据 token 的位置索引从查找表中获取相应的位置嵌入向量，然后**加到**对应的词嵌入向量上。

### 3. 相对位置编码 (Relative Positional Encoding)

*   **代表作**: Transformer-XL, T5, DeBERTa
*   **原理**: 不直接编码绝对位置，而是修改自注意力 (Self-Attention) 机制，使其在计算注意力得分时考虑查询 (Query) 和键 (Key) 之间的相对距离或关系。
*   **特点**: 通常在处理长序列和需要精确相对位置信息的任务上表现更好。

### 4. 旋转位置编码 (Rotary Position Embedding, RoPE)

*   **代表作**: Llama, PaLM
*   **原理**: 通过在 Attention 计算中对 Query 和 Key 向量应用与位置相关的旋转操作来融入相对位置信息。
*   **特点**: 在现代大型语言模型中非常流行，结合了固定编码的可扩展性和相对编码的优势。

**总结:**

位置编码是 Transformer 理解序列顺序的关键。**直接学习位置嵌入** (方法2) 和**固定的三角函数编码** (方法1) 是两种基础且常用的方法。你的代码采用了方法2。更先进的模型则可能采用相对位置编码 (方法3) 或旋转位置编码 (方法4) 等变体。选择哪种方法取决于具体的模型设计和任务需求。

**8. Padding Mask 函数 (`create_padding_mask`)**

```python
def create_padding_mask(batch):
    """创建用于attention的padding mask"""
    # batch: 输入的 token 索引序列, 形状为 (Batch, SeqLen)
    # 假设 padding token 的索引为 0
    # (batch == 0) 会生成一个布尔张量
    # 在 token 索引等于 0 的位置为 True, 其他位置为 False
    # 形状与 batch 相同: (Batch, SeqLen)
    return (batch == 0)

# 解释: 这个辅助函数非常简单但重要。它接收一个批次的 token 索引张量，并返回一个相同形状的布尔张量。
# 在 Transformer 的自注意力计算中，我们需要告诉模型哪些是真实的 token，哪些是为了让序列等长而填充的 `<pad>` token（其索引通常设为 0）。
# 这个函数通过比较输入张量中的每个元素是否等于 0 来生成这个掩码 (mask)。
# 返回的掩码中，`True` 值对应的位置表示是 padding，在 `nn.MultiheadAttention` 的 `key_padding_mask` 参数中使用时，这些位置的注意力权重会被忽略。
```

**9. 批处理函数 (`collate_batch`)**

```python
# 批处理函数：将不同长度的序列填充到相同长度
def collate_batch(batch):
    # batch: 一个列表，包含 N 个由 IMDBDataset.__getitem__ 返回的字典
    # 每个字典形如 {'text': tensor, 'length': int, 'label': tensor}

    # 提取批次中所有样本的 'text' (token 索引张量)
    texts = [item['text'] for item in batch]
    # 提取批次中所有样本的 'label' (标签张量)
    labels = [item['label'] for item in batch]
    # 提取批次中所有样本的 'length' (原始长度)
    lengths = [item['length'] for item in batch]

    # 核心步骤：填充序列 (Padding)
    # texts 是一个包含不同长度张量的列表
    # pad_sequence 会将这些张量填充到该批次中最长序列的长度
    padded_texts = pad_sequence(
        texts,           # 输入的张量列表
        batch_first=True,# 输出张量的形状为 (Batch, MaxSeqLen)
        padding_value=0  # 使用 0 (即 <pad> 的索引) 进行填充
    )

    # 将标签列表堆叠成一个张量 (Batch,)
    labels = torch.stack(labels)
    # 将长度列表转换为张量 (Batch,)
    lengths = torch.tensor(lengths)

    # 返回一个包含批处理后数据的字典
    return {
        'texts': padded_texts, # 填充后的文本序列张量
        'labels': labels,      # 标签张量
        'lengths': lengths     # 原始长度张量 (虽然模型里没直接用，但保留了信息)
    }

# 解释: DataLoader 在组合单个样本形成一个批次 (batch) 时，会调用 `collate_fn` 指定的函数。
# 由于文本序列通常长度不同，而神经网络（特别是需要矩阵运算的 Transformer）通常要求输入是形状规整的张量，因此需要进行填充 (Padding)。
# `collate_batch` 函数的作用就是：
# 1. 从输入的 `batch`（一个包含多个样本字典的列表）中，分别提取出 `texts`（token 索引张量列表）、`labels`（标签张量列表）和 `lengths`（长度列表）。
# 2. 使用 `torch.nn.utils.rnn.pad_sequence` 函数对 `texts` 列表中的张量进行填充。它会自动找到当前批次中最长的序列长度，并将所有其他序列用指定的 `padding_value`（这里是 0，代表 `<pad>`）填充到这个长度。`batch_first=True` 确保输出的 `padded_texts` 张量形状是 `(BatchSize, MaxSequenceLengthInBatch)`。
# 3. 使用 `torch.stack` 将 `labels` 列表中的单个标签张量堆叠成一个形状为 `(BatchSize,)` 的张量。
# 4. 将 `lengths` 列表转换为张量。
# 5. 返回一个字典，其中包含了处理好的、形状规整的批次数据，可以直接输入到模型中。
```

**10. 训练函数 (`train`)**

```python
# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    # model: 要训练的模型
    # dataloader: 提供训练数据的 DataLoader
    # optimizer: 优化器 (如 Adam)
    # criterion: 损失函数 (如 CrossEntropyLoss)
    # device: 计算设备 (CPU 或 CUDA)

    model.train() # 将模型设置为训练模式
    # 这会启用 Dropout、Batch Normalization 等在训练时需要开启的层

    total_loss = 0 # 累积一个 epoch 内的总损失
    all_preds = [] # 存储一个 epoch 内所有的预测结果
    all_labels = [] # 存储一个 epoch 内所有的真实标签

    # 使用 tqdm 包装 dataloader 以显示进度条
    progress_bar = tqdm(dataloader, desc='训练')

    # 迭代 DataLoader 提供的每个批次
    for batch in progress_bar:
        # 从批次字典中获取数据，并移动到指定设备
        texts = batch['texts'].to(device)   # (Batch, SeqLen)
        labels = batch['labels'].to(device) # (Batch,)
        lengths = batch['lengths'].to(device) # (Batch,) - 在这个模型中没直接用

        # 1. 清空梯度
        # 在计算新的梯度之前，必须清除上一步累积的梯度
        optimizer.zero_grad()

        # 2. 前向传播
        # 将输入数据传入模型，得到模型的输出 (logits)
        outputs = model(texts, lengths) # outputs 形状: (Batch, num_classes)

        # 3. 计算损失
        # 使用损失函数计算模型输出和真实标签之间的差距
        loss = criterion(outputs, labels)

        # 4. 反向传播
        # 计算损失相对于模型所有可训练参数的梯度
        loss.backward()

        # 5. 梯度裁剪 (可选但推荐)
        # 防止梯度爆炸问题，将梯度的范数限制在最大值 1.0 以内
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 6. 更新参数
        # 优化器根据计算出的梯度更新模型的参数
        optimizer.step()

        # --- 记录损失和准确率计算所需数据 ---
        # 从模型输出 (logits) 中获取预测类别
        # torch.max 返回 (values, indices)，我们取 indices (预测的类别索引)
        _, preds = torch.max(outputs, 1)
        # 将当前批次的预测和标签从 GPU 移到 CPU (.cpu()) 并转为 NumPy 数组 (.numpy())
        # 然后添加到列表中
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 累加当前批次的损失值
        total_loss += loss.item() # .item() 获取纯数值，避免累积计算图

        # 更新 tqdm 进度条的后缀信息，显示当前批次的损失
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    # --- Epoch 结束后的处理 ---
    # 计算整个 epoch 的训练准确率
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"训练准确率: {train_acc:.4f}")
    # 打印预测类别的分布，有助于发现模型是否倾向于预测某个特定类别
    print(f"训练预测分布: {Counter(all_preds)}")

    # 返回该 epoch 的平均训练损失
    return total_loss / len(dataloader)

# 解释: 这个函数封装了标准的 PyTorch 训练循环的一个完整 epoch。
# 1.  `model.train()`: 将模型切换到训练模式。
# 2.  初始化损失累加器和用于计算准确率的列表。
# 3.  迭代 `dataloader` 获取每个批次的数据。
# 4.  **核心训练步骤 (对于每个批次)**:
#     *   数据移动到 `device`。
#     *   `optimizer.zero_grad()`: 清除旧梯度。
#     *   `outputs = model(...)`: 前向传播，获得模型预测。
#     *   `loss = criterion(...)`: 计算损失。
#     *   `loss.backward()`: 反向传播，计算梯度。
#     *   `clip_grad_norm_`: (可选) 梯度裁剪，稳定训练。
#     *   `optimizer.step()`: 根据梯度更新模型权重。
# 5.  记录当前批次的损失和预测/标签，用于后续计算平均损失和准确率。
# 6.  Epoch 结束后，计算并打印整个 epoch 的训练准确率和预测分布。
# 7.  返回平均训练损失。
```

**11. 评估函数 (`evaluate`)**

```python
# 评估函数
def evaluate(model, dataloader, criterion, device):
    # model: 要评估的模型
    # dataloader: 提供评估数据的 DataLoader (通常是验证集或测试集)
    # criterion: 损失函数 (用于计算评估损失)
    # device: 计算设备

    model.eval() # 将模型设置为评估模式
    # 这会关闭 Dropout、Batch Normalization 的更新等，确保评估结果的一致性。

    total_loss = 0 # 累积评估过程中的总损失
    all_preds = [] # 存储所有的预测结果
    all_labels = [] # 存储所有的真实标签
    all_raw_outputs = [] # 存储模型原始输出 (logits 或概率)，用于更深入的分析

    # 使用 tqdm 显示评估进度
    progress_bar = tqdm(dataloader, desc='评估')

    # 关闭梯度计算上下文
    # 在评估阶段，我们不需要计算梯度，这样做可以节省内存并加速计算
    with torch.no_grad():
        # 迭代评估数据加载器
        for batch in progress_bar:
            # 获取数据并移动到设备
            texts = batch['texts'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            # 前向传播，获取模型输出
            outputs = model(texts, lengths) # outputs 形状: (Batch, num_classes)
            # 计算损失
            loss = criterion(outputs, labels)

            # 累加损失
            total_loss += loss.item()
            # 更新进度条信息
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 保存原始输出 (logits) 到列表中，先移动到 CPU 并分离计算图
            all_raw_outputs.append(outputs.cpu().detach())

            # 获取预测类别
            _, preds = torch.max(outputs, 1)
            # 收集预测和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # --- 调试信息：打印前几个批次的预测、标签和概率 ---
            # 这有助于在训练/评估早期快速判断模型是否在学习
            # if len(all_preds) <= 160:  # 假设 batch_size=32, 打印约前5个批次
            #     print(f"批次 {len(all_preds) // 32}:")
            #     print(f"  预测: {preds[:5].cpu().numpy()}")
            #     print(f"  标签: {labels[:5].cpu().numpy()}")
            #     # 使用 softmax 将 logits 转换为概率
            #     print(f"  输出 (概率): {torch.softmax(outputs[:5], dim=1).cpu().numpy()}")

    # --- 评估结束后的分析 ---
    # 检查预测和标签的分布
    print("评估预测分布:", Counter(all_preds))
    print("评估标签分布:", Counter(all_labels)) # 理想情况下应与预测分布相似

    # 计算随机预测的准确率作为基准
    # 如果模型的准确率不显著高于随机猜测，说明模型可能没有学到有效信息
    random_preds = np.random.randint(0, num_classes, size=len(all_labels)) # num_classes 在这里是 2
    random_acc = accuracy_score(all_labels, random_preds)
    print(f"随机预测准确率 (基准): {random_acc:.4f}")

    # 计算模型在评估集上的真实准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # --- 对原始输出进行更详细的分析 ---
    # 将所有批次的原始输出张量连接起来
    all_outputs = torch.cat(all_raw_outputs, dim=0) # 形状: (TotalSamples, num_classes)
    # 计算每个样本属于每个类别的概率
    probs = torch.softmax(all_outputs, dim=1).numpy() # 形状: (TotalSamples, num_classes)

    # 计算每个类别的平均预测概率
    # 这可以帮助了解模型预测的“信心”程度
    avg_prob_class0 = np.mean(probs[:, 0]) # 类别 0 (负面) 的平均概率
    avg_prob_class1 = np.mean(probs[:, 1]) # 类别 1 (正面) 的平均概率
    print(f"类别0平均概率: {avg_prob_class0:.4f}, 类别1平均概率: {avg_prob_class1:.4f}")

    # 返回平均评估损失、准确率、所有预测标签和所有真实标签
    return total_loss / len(dataloader), accuracy, all_preds, all_labels

# 解释: 这个函数用于在模型训练完成后（或每个 epoch 后）评估其在未见过的数据（验证集或测试集）上的性能。
# 1.  `model.eval()`: 将模型切换到评估模式。这很重要，因为它会禁用 Dropout 等只在训练时使用的层，确保评估结果的确定性。
# 2.  `with torch.no_grad()`: 创建一个上下文管理器，在此代码块内禁用梯度计算。因为评估时不需要更新模型参数，这样做可以减少内存消耗并加快计算速度。
# 3.  迭代 `dataloader` 获取评估批次数据。
# 4.  **核心评估步骤 (对于每个批次)**:
#     *   数据移动到 `device`。
#     *   `outputs = model(...)`: 前向传播，获得模型预测。
#     *   `loss = criterion(...)`: 计算损失（可选，但有助于监控验证损失）。
#     *   记录损失、原始输出 (`logits`)、预测类别 (`preds`) 和真实标签 (`labels`)。
# 5.  **评估后分析**:
#     *   打印预测类别和真实标签的分布（使用 `Counter`），可以快速检查是否有类别不平衡或模型预测偏差问题。
#     *   计算并打印随机猜测的准确率作为基准，方便比较模型性能。
#     *   使用 `sklearn.metrics.accuracy_score` 计算模型在整个评估集上的准确率。
#     *   对所有批次的原始输出 (`logits`) 进行汇总，计算 Softmax 概率，并打印每个类别的平均预测概率，提供模型预测置信度的信息。
# 6.  返回平均损失、准确率以及用于后续分析（如绘制混淆矩阵）的预测标签列表和真实标签列表。
```

**12. 绘制混淆矩阵 (`plot_confusion_matrix`)**

```python
def plot_confusion_matrix(true_labels, pred_labels):
    # true_labels: 真实的标签列表
    # pred_labels: 模型预测的标签列表

    # 1. 计算混淆矩阵
    # 使用 sklearn.metrics.confusion_matrix 计算
    # cm 是一个 NumPy 数组，例如 [[TN, FP], [FN, TP]]
    # TN: True Negative, FP: False Positive
    # FN: False Negative, TP: True Positive
    cm = confusion_matrix(true_labels, pred_labels)

    # 2. 使用 Matplotlib 和 Seaborn 绘制热力图
    plt.figure(figsize=(8, 6)) # 设置图像大小
    sns.heatmap(cm,              # 要绘制的数据 (混淆矩阵)
                annot=True,      # 在单元格中显示数值
                fmt='d',         # 数值格式为整数 ('d')
                cmap='Blues',    # 使用蓝色系调色板
                xticklabels=['Negative', 'Positive'], # x 轴刻度标签
                yticklabels=['Negative', 'Positive']  # y 轴刻度标签
               )
    plt.xlabel('Predicted') # x 轴标签
    plt.ylabel('True')      # y 轴标签
    plt.title('Confusion Matrix') # 图像标题
    plt.savefig('confusion_matrix.png') # 将图像保存到文件
    plt.show() # 显示图像

# 解释: 这个函数用于可视化模型的分类结果，展示模型在每个类别上的表现以及容易混淆的类别。
# 1.  它接收真实的标签列表和模型预测的标签列表作为输入。
# 2.  调用 `sklearn.metrics.confusion_matrix` 来计算混淆矩阵。
# 3.  使用 `seaborn.heatmap` 将混淆矩阵绘制成热力图，颜色深浅表示数量多少。
#     -   `annot=True`: 在每个格子上显示具体的数值。
#     -   `fmt='d'`: 设置显示格式为十进制整数。
#     -   `cmap='Blues'`: 设置颜色映射为蓝色系。
#     -   `xticklabels`, `yticklabels`: 设置坐标轴刻度的含义（负面/正面）。
# 4.  添加坐标轴标签和标题。
# 5.  `plt.savefig`: 将生成的图像保存为 'confusion_matrix.png' 文件。
# 6.  `plt.show()`: 在屏幕上显示图像。
# 混淆矩阵是评估分类模型性能的重要工具，可以清晰地看到真阳性、真阴性、假阳性、假阴性的数量。
```

**13. 绘制训练历史 (`plot_training_history`)**

```python
def plot_training_history(train_losses, val_losses, accuracies):
    # train_losses: 包含每个 epoch 训练损失的列表
    # val_losses: 包含每个 epoch 验证损失的列表
    # accuracies: 包含每个 epoch 验证准确率的列表

    plt.figure(figsize=(12, 5)) # 设置整个图的大小

    # 绘制第一个子图：损失曲线
    plt.subplot(1, 2, 1) # 创建一个 1 行 2 列的子图网格，当前激活第 1 个
    plt.plot(train_losses, label='Training Loss') # 绘制训练损失曲线
    plt.plot(val_losses, label='Validation Loss') # 绘制验证损失曲线
    plt.xlabel('Epoch') # x 轴标签
    plt.ylabel('Loss') # y 轴标签
    plt.legend() # 显示图例 (区分训练和验证损失)
    plt.title('Loss Curves') # 子图标题

    # 绘制第二个子图：准确率曲线
    plt.subplot(1, 2, 2) # 激活第 2 个子图
    plt.plot(accuracies) # 绘制验证准确率曲线
    plt.xlabel('Epoch') # x 轴标签
    plt.ylabel('Accuracy') # y 轴标签
    plt.title('Validation Accuracy') # 子图标题

    plt.tight_layout() # 自动调整子图参数，使其填充整个图像区域，防止标签重叠
    plt.savefig('training_history.png') # 保存图像
    plt.show() # 显示图像

# 解释: 这个函数用于可视化训练过程中的关键指标随 epoch 的变化情况。
# 1.  接收训练损失列表、验证损失列表和验证准确率列表。
# 2.  创建一个包含两个子图的图形窗口。
# 3.  **第一个子图**: 绘制训练损失和验证损失随 epoch 变化的曲线。这有助于：
#     *   观察模型是否在学习（损失是否下降）。
#     *   判断是否过拟合（训练损失持续下降，但验证损失开始上升）。
#     *   判断是否欠拟合（训练和验证损失都很高或下降缓慢）。
# 4.  **第二个子图**: 绘制验证准确率随 epoch 变化的曲线。这直接反映了模型在未见过数据上的性能变化。
# 5.  添加标签、标题和图例，使图像易于理解。
# 6.  `plt.tight_layout()` 优化布局。
# 7.  保存并显示图像。
# 这些曲线对于监控训练过程、调整超参数（如学习率、正则化强度）以及选择最佳模型（通常是验证性能最好时的模型）至关重要。
```

**14. 主函数 (`main`)**

```python
# 主函数
def main():
    print("加载数据集...")
    try:
        # 1. 加载数据
        # 使用我们自己实现的加载函数替代torchtext
        train_dataset, test_dataset, vocab = load_imdb_data_manually()
    except Exception as e:
        print(f"无法加载IMDB数据集: {e}")
        return # 如果加载失败，则退出程序

    # 2. 创建数据集子集 (可选，用于快速实验)
    # 为了快速训练和测试，可以只使用原始数据集的一部分
    train_subset_size = min(10000, len(train_dataset)) # 最多取 10000 个训练样本
    test_subset_size = min(2000, len(test_dataset))   # 最多取 2000 个测试样本

    # 使用 random.sample 随机抽取指定数量的索引，而不是简单地取前 N 个
    # 这样可以保证子集的数据分布与原始数据集更接近
    train_indices = random.sample(range(len(train_dataset)), train_subset_size)
    test_indices = random.sample(range(len(test_dataset)), test_subset_size)

    # 使用 torch.utils.data.Subset 创建子集对象
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"训练数据集大小: {len(train_subset)}")
    print(f"测试数据集大小: {len(test_subset)}")
    print(f"词汇表大小: {len(vocab)}") # 打印最终使用的词汇表大小

    # 3. 创建数据加载器 (DataLoaders)
    train_dataloader = DataLoader(
        train_subset,           # 使用训练子集
        batch_size=32,          # 每个批次包含 32 个样本
        shuffle=True,           # 在每个 epoch 开始时打乱数据顺序 (对训练很重要)
        collate_fn=collate_batch # 指定使用我们定义的 collate_batch 函数来处理批次数据 (处理变长序列和填充)
    )

    test_dataloader = DataLoader(
        test_subset,            # 使用测试子集
        batch_size=32,          # 批次大小与训练时一致
        shuffle=False,          # 测试时不需要打乱顺序
        collate_fn=collate_batch # 同样使用 collate_batch 处理填充
    )

    # 4. 定义模型超参数
    vocab_size = len(vocab)    # 词汇表大小，来自加载的数据
    embed_dim = 256            # 词嵌入向量的维度
    num_heads = 8              # Transformer 注意力头数
    num_layers = 4             # Transformer 编码器层数
    ff_dim = 512               # Transformer 前馈网络中间层维度
    num_classes = 2            # 类别数量 (正面/负面)
    dropout = 0.3              # Dropout 比率 (设置稍高以增强正则化)

    # 5. 初始化模型
    model = TransformerClassifier(
        vocab_size, embed_dim, num_heads, num_layers,
        ff_dim, num_classes, dropout=dropout
    ).to(device) # 将模型移动到之前确定的设备 (CPU 或 GPU)

    # 6. 设置优化器和损失函数
    # 使用 Adam 优化器，学习率设为 0.0001
    # 添加 weight_decay (L2 正则化)，有助于防止过拟合
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    # 使用交叉熵损失函数，适用于多分类任务 (这里是二分类)
    # 它内部包含了 Softmax 操作，所以模型输出原始 logits 即可
    criterion = nn.CrossEntropyLoss()

    # 7. 训练循环
    epochs = 5 # 训练的总轮数
    train_losses = [] # 记录每个 epoch 的训练损失
    val_losses = []   # 记录每个 epoch 的验证损失
    accuracies = []   # 记录每个 epoch 的验证准确率

    for epoch in range(epochs):
        start_time = time.time() # 记录 epoch 开始时间

        # 调用训练函数进行一轮训练
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        # 调用评估函数在测试集上进行评估 (这里用测试集作为验证集)
        val_loss, accuracy, _, _ = evaluate(model, test_dataloader, criterion, device)

        # 记录当前 epoch 的结果
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        elapsed_time = time.time() - start_time # 计算 epoch 耗时

        # 打印当前 epoch 的结果
        print(f"Epoch {epoch + 1}/{epochs} - 耗时: {elapsed_time:.2f}s")
        print(f"  训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 验证准确率: {accuracy:.4f}")

    # 8. 最终评估
    # 训练结束后，在测试集上进行最后一次评估，获取最终性能指标和预测结果
    _, final_accuracy, pred_labels, true_labels = evaluate(
        model, test_dataloader, criterion, device
    )
    print(f"最终测试准确率: {final_accuracy:.4f}")

    # 9. 可视化结果
    print("正在生成混淆矩阵...")
    # 使用最终评估得到的真实标签和预测标签绘制混淆矩阵
    plot_confusion_matrix(true_labels, pred_labels)

    print("正在生成训练历史图...")
    # 使用记录的训练/验证损失和准确率绘制训练历史曲线
    plot_training_history(train_losses, val_losses, accuracies)

    # 10. 保存模型
    # 保存训练好的模型的状态字典 (包含所有可学习参数)
    torch.save(model.state_dict(), "transformer_classifier.pth")
    print("模型已保存到 transformer_classifier.pth")


# Python 入口点检查
if __name__ == "__main__":
    # 只有当脚本作为主程序直接运行时，才执行 main() 函数
    # 如果此脚本被其他脚本导入，则 main() 不会自动执行
    main()

# 解释: `main` 函数是整个程序的入口和协调者，它按照顺序执行了模型训练和评估的所有步骤。
# 1.  **加载数据**: 调用 `load_imdb_data_manually` 获取训练/测试数据集和词汇表。包含错误处理。
# 2.  **创建子集**: (可选) 为了加速实验，使用 `random.sample` 和 `torch.utils.data.Subset` 创建了较小规模的训练和测试子集。
# 3.  **创建 DataLoader**: 使用 `DataLoader` 将数据集包装起来，实现批处理、数据打乱和自动填充（通过 `collate_fn`）。
# 4.  **定义超参数**: 设置了模型结构（嵌入维度、头数、层数等）、类别数和 Dropout 比率。
# 5.  **初始化模型**: 根据超参数创建 `TransformerClassifier` 实例，并将其移动到 `device`。
# 6.  **设置优化器和损失函数**: 选择 Adam 优化器和交叉熵损失函数。加入了 `weight_decay` 进行 L2 正则化。
# 7.  **训练循环**:
#     *   设置训练的总 `epochs`。
#     *   初始化列表来存储每个 epoch 的损失和准确率。
#     *   外层循环遍历每个 epoch。
#     *   在每个 epoch 内，调用 `train` 函数执行一次训练。
#     *   调用 `evaluate` 函数在测试集（在此代码中充当验证集）上评估模型性能。
#     *   记录该 epoch 的训练损失、验证损失和验证准确率。
#     *   打印该 epoch 的结果和耗时。
# 8.  **最终评估**: 训练循环结束后，再次调用 `evaluate` 函数获取模型在测试集上的最终性能指标以及预测结果和真实标签，用于后续分析。
# 9.  **可视化**: 调用 `plot_confusion_matrix` 和 `plot_training_history` 函数，生成并保存/显示混淆矩阵和训练曲线图。
# 10. **保存模型**: 使用 `torch.save` 保存模型的 `state_dict`（包含所有学习到的权重和偏差）。这允许之后加载模型进行推理或继续训练，而无需重新训练。
# 11. `if __name__ == "__main__":`: 这是 Python 脚本的标准入口点，确保 `main()` 函数只在直接运行此脚本时执行。
```

至此，整个代码的分析就完成了。这段代码实现了一个完整的情感分类流程：从手动下载和预处理 IMDB 数据集，构建词汇表，定义 Transformer 模型，到训练、评估模型，最后可视化结果并保存模型。
