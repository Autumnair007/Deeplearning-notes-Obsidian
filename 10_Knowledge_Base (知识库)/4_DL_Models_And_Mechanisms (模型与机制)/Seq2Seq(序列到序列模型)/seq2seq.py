import collections
import os
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import requests
import zipfile
import re
import time

# 下载并处理数据
# 这一部分可能存在网络请求和文件操作的错误，可能需要手动下载数据集
def download_and_extract_data():
    """下载并解压翻译数据集"""
    url = 'http://www.manythings.org/anki/fra-eng.zip'
    zip_path = 'fra-eng.zip'
    extracted_dir = 'fra-eng'
    txt_path = 'fra.txt'

    # 如果数据集还不存在，则下载
    if not os.path.exists(extracted_dir) and not os.path.exists(extracted_dir + txt_path):
        print(f"下载数据集从 {url}...")
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"解压数据集...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall('.')

        # 删除zip文件
        os.remove(zip_path)
        print("数据集下载和解压完成!")
    else:
        print("数据集已经存在，无需下载")

    return os.path.join(extracted_dir, 'fra.txt')

# 预处理文本数据
def preprocess_nmt_data(data_file, max_examples=10000):
    """预处理翻译数据，返回英语-法语句子对"""
    text_pairs = []
    with open(data_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[:max_examples]:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                # 英文句子与法语句子对, 添加开始和结束标记
                en = parts[0].lower()
                fr = parts[1].lower()
                # 只保留简单的标点和英文字符
                en = re.sub(r'[^a-zA-Z0-9,.!?\'\" ]+', ' ', en)
                fr = re.sub(r'[^a-zA-Z0-9,.!?\'\" ]+', ' ', fr)
                text_pairs.append((en, fr))
    return text_pairs

class Vocab:
    """词汇表"""

    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # 未知标记的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

def build_vocab(text_pairs, min_freq=2):
    """构建词汇表"""
    source_tokens = []
    target_tokens = []
    for source, target in text_pairs:
        source_tokens.extend(source.split())
        target_tokens.extend(target.split())

    src_vocab = Vocab(source_tokens, min_freq,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocab(target_tokens, min_freq,
                      reserved_tokens=['<pad>', '<bos>', '<eos>'])
    return src_vocab, tgt_vocab

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列"""
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充

def build_array(text_pairs, src_vocab, tgt_vocab, num_steps):
    """将文本序列转换为小批量"""
    src_seqs = []
    tgt_seqs = []
    for (src_text, tgt_text) in text_pairs:
        src_tokens = src_text.split()
        tgt_tokens = tgt_text.split()
        # 添加<eos>标记
        src_tokens.append('<eos>')
        tgt_tokens.append('<eos>')
        # 转换为索引
        src_indices = src_vocab[src_tokens]
        tgt_indices = tgt_vocab[tgt_tokens]
        # 添加到列表中
        src_seqs.append(src_indices)
        tgt_seqs.append(tgt_indices)

    # 截断或填充序列
    src_array = torch.tensor([truncate_pad(
        seq, num_steps, src_vocab['<pad>']) for seq in src_seqs])
    tgt_array = torch.tensor([truncate_pad(
        seq, num_steps, tgt_vocab['<pad>']) for seq in tgt_seqs])

    return src_array, tgt_array

def compute_valid_len(X, pad_val):
    """计算每个序列的有效长度"""
    if X.dim() == 1:  # Single sequence
        return (X != pad_val).sum()
    else:  # Batch of sequences
        return (X != pad_val).sum(dim=1)

class NMTDataset(data.Dataset):
    """机器翻译数据集"""

    def __init__(self, src_array, tgt_array, src_vocab, tgt_vocab):
        self.src_array = src_array
        self.tgt_array = tgt_array
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __getitem__(self, index):
        src = self.src_array[index]
        tgt = self.tgt_array[index]
        src_valid_len = compute_valid_len(src, self.src_vocab['<pad>'])
        tgt_valid_len = compute_valid_len(tgt, self.tgt_vocab['<pad>'])
        return src, src_valid_len, tgt, tgt_valid_len

    def __len__(self):
        return len(self.src_array)

def load_data_nmt(batch_size, num_steps, max_examples=10000):
    """加载NMT数据集"""
    data_file = download_and_extract_data()
    text_pairs = preprocess_nmt_data(data_file, max_examples)
    print(f"加载了 {len(text_pairs)} 个翻译句子对")

    src_vocab, tgt_vocab = build_vocab(text_pairs)
    print(f"源语言词汇量: {len(src_vocab)}, 目标语言词汇量: {len(tgt_vocab)}")

    # 划分训练集和验证集
    n = len(text_pairs)
    train_pairs = text_pairs[:int(0.9 * n)]
    valid_pairs = text_pairs[int(0.9 * n):]

    # 构建数据集
    train_src_array, train_tgt_array = build_array(
        train_pairs, src_vocab, tgt_vocab, num_steps)
    valid_src_array, valid_tgt_array = build_array(
        valid_pairs, src_vocab, tgt_vocab, num_steps)

    train_dataset = NMTDataset(train_src_array, train_tgt_array, src_vocab, tgt_vocab)
    valid_dataset = NMTDataset(valid_src_array, valid_tgt_array, src_vocab, tgt_vocab)

    # 构建数据迭代器
    train_iter = data.DataLoader(train_dataset, batch_size, shuffle=True)
    valid_iter = data.DataLoader(valid_dataset, batch_size)

    return train_iter, valid_iter, src_vocab, tgt_vocab

# Encoder基类
class Encoder(nn.Module):
    """编码器的基类"""

    def __init__(self):
        super().__init__()

    def forward(self, X, *args):
        raise NotImplementedError


# Decoder基类
class Decoder(nn.Module):
    """解码器的基类"""

    def __init__(self):
        super().__init__()

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, enc_valid_len):
        enc_outputs = self.encoder(enc_X, enc_valid_len)
        dec_state = self.decoder.init_state(enc_outputs, enc_valid_len)
        return self.decoder(dec_X, dec_state)


class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super(Seq2SeqEncoder, self).__init__()
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                          dropout=dropout)

    def forward(self, X, valid_len):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        X = X.permute(1, 0, 2)
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                          dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)
        self.attention_weights = None

    def init_state(self, enc_outputs, enc_valid_len):
        return enc_outputs[1]

    def forward(self, X, state):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播context，使其具有与X相同的num_steps
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)
        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output的形状:(batch_size,num_steps,vocab_size)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state


def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""

    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = torch.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction = 'none'
        unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
            pred.permute(0, 2, 1), label)
        weighted_loss = (unweighted_loss * weights).sum(dim=1) / valid_len
        return weighted_loss


def grad_clipping(net, theta):
    """梯度裁剪"""
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            if param.grad is not None:
                param.grad[:] *= theta / norm


class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并记录时间"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()


class Accumulator:
    """累加器"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_seq2seq(net, train_iter, valid_iter, lr, num_epochs, tgt_vocab, device):
    """训练序列到序列模型"""

    def xavier_init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
        if type(m) == nn.GRU:
            for param in m._flat_weights_names:
                if "weight" in param:
                    nn.init.xavier_uniform_(m._parameters[param])

    net.apply(xavier_init_weights)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = MaskedSoftmaxCELoss()
    net.train()

    # 用于可视化的损失历史记录
    train_losses = []
    valid_losses = []

    for epoch in range(num_epochs):
        timer = Timer()
        metric = Accumulator(2)  # 训练损失总和，词元数量
        for batch in train_iter:
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            l.sum().backward()  # 损失函数的标量进行"反向传播"
            grad_clipping(net, 1)
            num_tokens = Y_valid_len.sum()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), num_tokens)

        train_loss = metric[0] / metric[1]
        train_losses.append(train_loss)

        # 计算验证损失
        if valid_iter is not None:
            valid_loss = evaluate_loss(net, valid_iter, loss, device)
            valid_losses.append(valid_loss)
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {train_loss:.4f}, '
                  f'Valid loss: {valid_loss:.4f}, '
                  f'{metric[1] / timer.stop():.1f} tokens/sec')
        else:
            print(f'Epoch {epoch + 1}/{num_epochs}, Train loss: {train_loss:.4f}, '
                  f'{metric[1] / timer.stop():.1f} tokens/sec')

    # 绘制训练过程的损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    if valid_iter is not None:
        plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('training_loss.png')
    plt.close()

    return train_losses, valid_losses


def evaluate_loss(net, data_iter, loss_fn, device):
    """评估模型在数据集上的损失"""
    net.eval()
    metric = Accumulator(2)  # 损失总和，词元数量
    with torch.no_grad():
        for batch in data_iter:
            X, X_valid_len, Y, Y_valid_len = [x.to(device) for x in batch]
            bos = torch.tensor([data_iter.dataset.tgt_vocab['<bos>']] * Y.shape[0],
                               device=device).reshape(-1, 1)
            dec_input = torch.cat([bos, Y[:, :-1]], 1)  # 强制教学
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss_fn(Y_hat, Y, Y_valid_len)
            metric.add(l.sum(), Y_valid_len.sum())
    net.train()
    return metric[0] / metric[1]


def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """序列到序列模型的预测"""
    # 在预测时将net设置为评估模式
    net.eval()
    src_tokens = src_sentence.lower().split(' ')
    src_tokens = ['<bos>'] + src_tokens + ['<eos>']
    enc_valid_len = torch.tensor([len(src_tokens)], device=device)
    src_indices = src_vocab[src_tokens]
    src_indices = torch.tensor(truncate_pad(src_indices, num_steps, src_vocab['<pad>']),
                               dtype=torch.long, device=device).unsqueeze(0)
    # 添加批量轴
    enc_outputs = net.encoder(src_indices, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # 添加批量轴
    dec_X = torch.tensor([[tgt_vocab['<bos>']]], dtype=torch.long, device=device)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # 我们使用具有预测最高可能性的词元，作为解码器在下一时间步的输入
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # 保存注意力权重
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # 一旦序列结束词元被预测，输出序列的生成就完成了
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def translate_sentences(net, sentences, src_vocab, tgt_vocab, num_steps, device):
    """翻译多个句子并打印结果"""
    for i, sentence in enumerate(sentences):
        translation, _ = predict_seq2seq(
            net, sentence, src_vocab, tgt_vocab, num_steps, device)
        print(f"{i + 1}. 源语言: {sentence}")
        print(f"   翻译结果: {translation}\n")


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def main():
    # 参数设置
    embed_size = 32
    num_hiddens = 32
    num_layers = 2
    dropout = 0.1
    batch_size = 64
    num_steps = 10  # 序列长度上限
    lr = 0.005
    num_epochs = 30  # 减少训练轮数，方便快速运行
    max_examples = 10000  # 限制数据集大小

    # 使用GPU或者CPU
    device = try_gpu()
    print(f"使用设备: {device}")

    # 加载数据
    train_iter, valid_iter, src_vocab, tgt_vocab = load_data_nmt(
        batch_size, num_steps, max_examples)

    # 创建模型
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
    net = EncoderDecoder(encoder, decoder)

    # 训练模型
    print("开始训练模型...")
    train_losses, valid_losses = train_seq2seq(
        net, train_iter, valid_iter, lr, num_epochs, tgt_vocab, device)

    # 保存模型
    torch.save(net.state_dict(), 'seq2seq_model.pth')
    print("模型已保存到 'seq2seq_model.pth'")

    # 翻译一些示例句子
    test_sentences = [
        "hello world",
        "how are you",
        "i love machine translation"
    ]

    print("\n翻译示例句子:")
    translate_sentences(net, test_sentences, src_vocab, tgt_vocab, num_steps, device)


if __name__ == "__main__":
    main()