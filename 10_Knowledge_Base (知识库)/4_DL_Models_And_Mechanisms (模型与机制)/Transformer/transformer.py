import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import pandas as pd
import time
import re
import os
import tarfile
import urllib.request
import random
import numpy as np
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# 设置随机种子以便结果可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置数据集下载路径为当前文件夹
current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
data_dir = os.path.join(current_dir, "data")
os.makedirs(data_dir, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"数据集将下载到: {data_dir}")

if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)


# 手动下载并处理IMDB数据集
def download_and_extract_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    extracted_path = os.path.join(data_dir, "aclImdb")

    if not os.path.exists(extracted_path):
        if not os.path.exists(dataset_path):
            print(f"下载IMDB数据集到 {dataset_path}...")
            urllib.request.urlretrieve(url, dataset_path)
            print("下载完成！")

        print("解压数据集...")
        with tarfile.open(dataset_path, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        print("解压完成！")
    else:
        print("IMDB数据集已存在，跳过下载和解压步骤。")

    return extracted_path


# 数据预处理和加载
class IMDBDataset(Dataset):
    def __init__(self, texts, labels, vocab, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为token索引
        tokens = self.tokenizer(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        token_ids = [self.vocab.get(token, self.vocab['<unk>']) for token in tokens]

        return {
            'text': torch.tensor(token_ids, dtype=torch.long),
            'length': len(token_ids),
            'label': torch.tensor(label, dtype=torch.long)
        }


# 简单的分词器
def basic_english_tokenizer(text):
    # 简易版的basic_english tokenizer
    text = text.lower()
    # 将标点符号替换为空格
    text = re.sub(r'[^\w\s]', ' ', text)
    # 将多个空格合并为一个
    text = re.sub(r'\s+', ' ', text)
    return text.strip().split()


def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# 加载IMDB数据集（不使用torchtext）
def load_imdb_data_manually():
    # 下载并解压数据集
    dataset_dir = download_and_extract_imdb()

    tokenizer = basic_english_tokenizer

    # 加载训练数据
    print("加载训练数据...")
    train_texts, train_labels = [], []

    # 加载正面评价
    pos_dir = os.path.join(dataset_dir, "train", "pos")
    for filename in tqdm(os.listdir(pos_dir), desc="加载正面训练评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            train_texts.append(clean_text(text))
            train_labels.append(1)  # 正面为1

    # 加载负面评价
    neg_dir = os.path.join(dataset_dir, "train", "neg")
    for filename in tqdm(os.listdir(neg_dir), desc="加载负面训练评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            train_texts.append(clean_text(text))
            train_labels.append(0)  # 负面为0

    # 加载测试数据
    print("加载测试数据...")
    test_texts, test_labels = [], []

    # 加载正面评价
    pos_dir = os.path.join(dataset_dir, "test", "pos")
    for filename in tqdm(os.listdir(pos_dir), desc="加载正面测试评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(clean_text(text))
            test_labels.append(1)

    # 加载负面评价
    neg_dir = os.path.join(dataset_dir, "test", "neg")
    for filename in tqdm(os.listdir(neg_dir), desc="加载负面测试评价"):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_dir, filename), 'r', encoding='utf-8') as f:
                text = f.read()
            test_texts.append(clean_text(text))
            test_labels.append(0)

    # 创建词汇表
    print("构建词汇表...")
    counter = Counter()
    for text in tqdm(train_texts, desc="统计词频"):
        counter.update(tokenizer(text))

    # 选择频率大于5的词
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, count in counter.items():
        if count >= 5:
            vocab[word] = idx
            idx += 1

    print(f"词汇表大小: {len(vocab)}")

    # 打乱数据集顺序
    combined = list(zip(train_texts, train_labels))
    random.shuffle(combined)
    train_texts, train_labels = zip(*combined)

    combined = list(zip(test_texts, test_labels))
    random.shuffle(combined)
    test_texts, test_labels = zip(*combined)

    # 创建数据集
    train_dataset = IMDBDataset(train_texts, train_labels, vocab, tokenizer)
    test_dataset = IMDBDataset(test_texts, test_labels, vocab, tokenizer)

    return train_dataset, test_dataset, vocab


# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


# 定义Transformer分类器
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_dim, num_classes, max_len=512, dropout=0.1):
        super(TransformerClassifier, self).__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(max_len, embed_dim))

        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, lengths):
        # 创建掩码，处理变长序列
        mask = create_padding_mask(x)

        # 添加位置编码
        batch_size, seq_len = x.size()
        x = self.token_embedding(x)  # [batch_size, seq_len, embed_dim]
        x = x + self.position_embedding[:seq_len, :]

        # 将序列维度放到第一位，以适配PyTorch的Transformer接口
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]

        # 应用Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # 转回 [batch_size, seq_len, embed_dim]
        x = x.transpose(0, 1)

        # 使用mean pooling而不是last token (更稳定)
        mask = mask.float().unsqueeze(-1)
        mask = 1 - mask  # 反转mask，1表示有效位置，0表示padding
        # 计算每个序列的平均值，忽略padding
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        # 分类
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits


def create_padding_mask(batch):
    """创建用于attention的padding mask"""
    return (batch == 0)


# 批处理函数：将不同长度的序列填充到相同长度
def collate_batch(batch):
    texts = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    lengths = [item['length'] for item in batch]

    # 填充到同一长度
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)

    return {
        'texts': padded_texts,
        'labels': torch.stack(labels),
        'lengths': torch.tensor(lengths)
    }


# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc='训练')

    for batch in progress_bar:
        texts = batch['texts'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()

        outputs = model(texts, lengths)
        loss = criterion(outputs, labels)

        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

    train_acc = accuracy_score(all_labels, all_preds)
    print(f"训练准确率: {train_acc:.4f}")
    print(f"训练预测分布: {Counter(all_preds)}")

    return total_loss / len(dataloader)


# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_raw_outputs = []

    progress_bar = tqdm(dataloader, desc='评估')

    with torch.no_grad():
        for batch in progress_bar:
            texts = batch['texts'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            outputs = model(texts, lengths)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            # 保存原始输出用于分析
            all_raw_outputs.append(outputs.cpu().detach())

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 打印前5个批次的调试信息
            if len(all_preds) <= 160:  # 约5个批次
                print(f"批次 {len(all_preds) // 32}:")
                print(f"  预测: {preds[:5].cpu().numpy()}")
                print(f"  标签: {labels[:5].cpu().numpy()}")
                print(f"  输出: {torch.softmax(outputs[:5], dim=1).cpu().numpy()}")

    # 检查预测分布
    print("预测分布:", Counter(all_preds))
    print("标签分布:", Counter(all_labels))

    # 随机预测作为基准测试
    random_preds = np.random.randint(0, 2, size=len(all_labels))
    random_acc = accuracy_score(all_labels, random_preds)
    print(f"随机预测准确率: {random_acc:.4f}")

    # 计算真实准确率
    accuracy = accuracy_score(all_labels, all_preds)

    # 连接所有原始输出进行分析
    all_outputs = torch.cat(all_raw_outputs, dim=0)
    probs = torch.softmax(all_outputs, dim=1).numpy()

    # 输出概率分布统计
    avg_prob_class0 = np.mean(probs[:, 0])
    avg_prob_class1 = np.mean(probs[:, 1])
    print(f"类别0平均概率: {avg_prob_class0:.4f}, 类别1平均概率: {avg_prob_class1:.4f}")

    return total_loss / len(dataloader), accuracy, all_preds, all_labels


def plot_confusion_matrix(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()


def plot_training_history(train_losses, val_losses, accuracies):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# 主函数
def main():
    print("加载数据集...")
    try:
        # 使用我们自己实现的加载函数替代torchtext
        train_dataset, test_dataset, vocab = load_imdb_data_manually()
    except Exception as e:
        print(f"无法加载IMDB数据集: {e}")
        return

    # 为了快速训练，使用较小的子集
    train_subset_size = min(10000, len(train_dataset))
    test_subset_size = min(2000, len(test_dataset))

    # 随机采样而不是取前N个
    train_indices = random.sample(range(len(train_dataset)), train_subset_size)
    test_indices = random.sample(range(len(test_dataset)), test_subset_size)

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    print(f"训练数据集大小: {len(train_subset)}")
    print(f"测试数据集大小: {len(test_subset)}")
    print(f"词汇表大小: {len(vocab)}")

    # 创建数据加载器
    train_dataloader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_batch
    )

    test_dataloader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_batch
    )

    # 模型参数
    vocab_size = len(vocab)
    embed_dim = 256  # 嵌入维度
    num_heads = 8  # 注意力头数量
    num_layers = 4  # Transformer层数量
    ff_dim = 512  # 前馈神经网络维度
    num_classes = 2  # 二分类任务
    dropout = 0.3  # 增加dropout值以增强正则化

    # 初始化模型
    model = TransformerClassifier(
        vocab_size, embed_dim, num_heads, num_layers,
        ff_dim, num_classes, dropout=dropout
    ).to(device)

    # 设置优化器和损失函数，增加权重衰减
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 训练模型
    epochs = 5
    train_losses = []
    val_losses = []
    accuracies = []

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, accuracy, _, _ = evaluate(model, test_dataloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        accuracies.append(accuracy)

        elapsed_time = time.time() - start_time

        print(f"Epoch {epoch + 1}/{epochs} - 耗时: {elapsed_time:.2f}s")
        print(f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, 准确率: {accuracy:.4f}")

    # 评估模型
    _, final_accuracy, pred_labels, true_labels = evaluate(
        model, test_dataloader, criterion, device
    )

    print(f"最终测试准确率: {final_accuracy:.4f}")

    # 可视化结果
    print("正在生成混淆矩阵...")
    plot_confusion_matrix(true_labels, pred_labels)

    print("正在生成训练历史图...")
    plot_training_history(train_losses, val_losses, accuracies)

    # 保存模型
    torch.save(model.state_dict(), "transformer_classifier.pth")
    print("模型已保存到 transformer_classifier.pth")


if __name__ == "__main__":
    main()