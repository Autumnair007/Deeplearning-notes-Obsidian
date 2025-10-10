---
type: code-note
tags:
  - cv
  - nlp
  - multimodal
  - contrastive-learning
  - transformer
  - resnet
  - pytorch
  - clip
status: done
model: CLIP (Simplified)
key_concept: Contrastive Learning between Image and Text in a Shared Embedding Space.
---
## 1. 附有逐行注释的完整代码

这是一个简化版的 CLIP (Contrastive Language-Image Pre-training) 模型的实现，针对中低端 GPU (如 RTX 4060) 进行了优化。代码包含了模型定义、数据处理、训练、保存/加载和测试的全过程。

```python
# 导入所有必需的库
import torch # PyTorch 核心库
import torch.nn as nn # PyTorch 神经网络模块，包含各种层和损失函数
from torch.utils.data import Dataset, DataLoader # 用于创建和管理数据集的工具
from torchvision import transforms, models # 用于图像处理和预训练模型
import re # 正则表达式库，用于文本处理
from collections import Counter # 计数器，用于构建词汇表
import os # 操作系统接口，用于文件路径操作
import json # 用于读写 JSON 文件
import numpy as np # NumPy 库，用于数值计算
from PIL import Image # Python Imaging Library，用于图像文件的读写和处理
import matplotlib.pyplot as plt # 用于绘制图表，如训练曲线
from tqdm import tqdm # 一个快速、可扩展的 Python 进度条库
import argparse # 用于解析命令行参数

# =============================================================================
# 模型定义部分 (Model Definition)
# =============================================================================

class ImageEncoder(nn.Module): # 定义图像编码器类，继承自 nn.Module
    """
    图像编码器：使用 torchvision 的预训练视觉模型
    针对 4060 显卡优化，使用较小的模型
    """
    def __init__(self, model_name='resnet18', pretrained=True): # 初始化函数
        super().__init__() # 调用父类 nn.Module 的初始化方法
        # 使用 torchvision 的轻量级模型
        if model_name == 'resnet18': # 如果模型名称是 'resnet18'
            self.model = models.resnet18(pretrained=pretrained) # 加载预训练的 ResNet-18 模型
            # 移除最后的分类层 (全连接层)，我们只需要特征提取部分
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 512 # ResNet-18 去掉分类层后的输出特征维度是 512
        elif model_name == 'resnet34': # 如果模型名称是 'resnet34'
            self.model = models.resnet34(pretrained=pretrained) # 加载预训练的 ResNet-34 模型
            self.model = nn.Sequential(*list(self.model.children())[:-1]) # 同样移除分类层
            self.output_dim = 512 # ResNet-34 去掉分类层后的输出特征维度也是 512
        else: # 如果是不支持的模型名称
            raise ValueError(f"Unsupported model: {model_name}") # 抛出错误
        
        print(f"Image encoder output dim: {self.output_dim}") # 打印图像编码器的输出维度

    def forward(self, images): # 定义前向传播函数
        # 输入: [batch_size, 3, 224, 224]
        # 输出: [batch_size, output_dim]
        features = self.model(images) # 将图像传入模型，提取特征
        return features.flatten(1)  # 展平除 batch 维度外的所有维度，得到 [batch_size, output_dim] 的向量

class TextEncoder(nn.Module): # 定义文本编码器类
    """
    文本编码器：使用简单的嵌入层和 Transformer
    """
    def __init__(self, vocab_size=5000, hidden_size=256, num_layers=2): # 初始化函数
        super().__init__() # 调用父类初始化
        self.vocab_size = vocab_size # 词汇表大小
        self.hidden_size = hidden_size # 隐藏层大小，也是词嵌入维度
        self.output_dim = hidden_size # 文本编码器的输出维度等于隐藏层大小
        
        # 嵌入层：将词索引映射为密集向量
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0) # padding_idx=0 表示索引0是填充符，其向量为0且不参与梯度更新
        self.positional_encoding = nn.Embedding(77, hidden_size)  # 位置编码层，最大长度为77 (同CLIP原文)
        
        # 定义一个 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, # 模型的特征维度
            nhead=8, # 多头注意力机制的头数
            dim_feedforward=hidden_size * 2, # 前馈神经网络的隐藏层维度
            dropout=0.1, # dropout 比率
            batch_first=True # 输入和输出张量的第一个维度是 batch_size
        )
        # 将多个编码器层堆叠成一个 Transformer 编码器
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        print(f"Text encoder output dim: {self.output_dim}") # 打印文本编码器的输出维度

    def forward(self, input_ids, attention_mask): # 定义前向传播函数
        batch_size, seq_len = input_ids.shape # 获取输入的 batch 大小和序列长度
        
        # 词嵌入
        token_embeddings = self.embedding(input_ids) # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1) # 创建位置索引
        pos_embeddings = self.positional_encoding(positions) # [batch_size, seq_len] -> [batch_size, seq_len, hidden_size]
        
        # 组合词嵌入和位置嵌入
        embeddings = token_embeddings + pos_embeddings # 得到最终的输入嵌入
        
        # 创建注意力掩码 (Transformer期望的格式)
        # True 表示该位置应该被忽略 (mask掉)
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer 编码
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask) # 将嵌入和掩码传入 Transformer
        
        # 返回 [CLS] token 的表示 (序列的第一个 token)，作为整个句子的语义表示
        return encoded[:, 0, :] # [batch_size, hidden_size]

class CLIP(nn.Module): # 定义 CLIP 主模型
    """
    CLIP主模型：针对 4060 显卡优化的版本
    """
    def __init__(self, image_encoder, text_encoder, shared_embedding_dim=256): # 初始化函数
        super().__init__() # 调用父类初始化
        self.image_encoder = image_encoder # 实例化的图像编码器
        self.text_encoder = text_encoder # 实例化的文本编码器

        # 投射层：将图像和文本特征投射到同一个共享的嵌入空间
        self.image_projection = nn.Sequential( # 图像投射头
            nn.Linear(self.image_encoder.output_dim, shared_embedding_dim, bias=False), # 线性层
            nn.LayerNorm(shared_embedding_dim) # 层归一化
        )
        self.text_projection = nn.Sequential( # 文本投射头
            nn.Linear(self.text_encoder.output_dim, shared_embedding_dim, bias=False), # 线性层
            nn.LayerNorm(shared_embedding_dim) # 层归一化
        )

        # 可学习的温度系数，用于缩放 logits，控制对比损失的锐度
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07))) # 初始化为 log(1/0.07)

    def forward(self, images, input_ids, attention_mask): # 定义前向传播
        # 编码
        image_features_raw = self.image_encoder(images) # 获取原始图像特征
        text_features_raw = self.text_encoder(input_ids, attention_mask) # 获取原始文本特征

        # 投射到共享空间
        image_embedding = self.image_projection(image_features_raw) # 将图像特征投射到共享空间
        text_embedding = self.text_projection(text_features_raw) # 将文本特征投射到共享空间

        # L2 归一化，使得特征向量在单位超球面上
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return image_embedding, text_embedding # 返回归一化后的图文嵌入

    def get_image_features(self, images): # 辅助函数：单独获取图像特征 (用于推理)
        """单独获取图像特征"""
        with torch.no_grad(): # 不计算梯度，节省计算资源
            image_features = self.image_encoder(images) # 编码
            image_embedding = self.image_projection(image_features) # 投射
            return image_embedding / image_embedding.norm(dim=-1, keepdim=True) # 归一化并返回

    def get_text_features(self, input_ids, attention_mask): # 辅助函数：单独获取文本特征 (用于推理)
        """单独获取文本特征"""
        with torch.no_grad(): # 不计算梯度
            text_features = self.text_encoder(input_ids, attention_mask) # 编码
            text_embedding = self.text_projection(text_features) # 投射
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True) # 归一化并返回

# =============================================================================
# 数据集类定义 (Dataset Class Definition)
# =============================================================================

class SimpleImageTextDataset(Dataset): # 定义数据集类
    """
    简单的图文对数据集
    """
    def __init__(self, data_dir, transform=None, max_length=77): # 初始化
        self.data_dir = data_dir # 数据集目录
        self.transform = transform # 图像预处理流程
        self.max_length = max_length # 文本最大长度
        
        # 加载数据集描述文件
        with open(os.path.join(data_dir, "dataset.json"), "r") as f: # 打开 JSON 文件
            self.data = json.load(f) # 加载数据
        
        # 使用简单的本地 tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=5000) # 实例化自定义的 tokenizer
        
        # 从所有文本中构建词汇表
        texts = [item["text"] for item in self.data] # 提取所有文本描述
        self.tokenizer.build_vocab(texts) # 构建词汇表
        
        print(f"Loaded {len(self.data)} samples from {data_dir}") # 打印加载的样本数量

    def __len__(self): # 返回数据集大小
        return len(self.data) # 返回样本总数

    def __getitem__(self, idx): # 定义如何根据索引获取单个样本
        item = self.data[idx] # 获取对应索引的数据项
        
        # 加载图片 - 处理路径
        image_path = item["image_path"] # 获取图像路径
        # 如果是绝对路径但文件不存在，尝试拼接成相对路径
        if not os.path.exists(image_path):
            filename = os.path.basename(image_path) # 提取文件名
            image_path = os.path.join(self.data_dir, "images", filename) # 在数据目录中查找
        
        image = Image.open(image_path).convert("RGB") # 打开图像并转换为 RGB 格式
        
        if self.transform: # 如果定义了图像变换
            image = self.transform(image) # 应用变换
        
        # 处理文本
        text = item["text"] # 获取文本描述
        encoded = self.tokenizer( # 使用 tokenizer 对文本进行编码
            text,
            truncation=True, # 开启截断
            padding='max_length', # 填充到最大长度
            max_length=self.max_length, # 指定最大长度
            return_tensors='pt' # 返回 PyTorch 张量
        )
        
        return { # 返回一个包含图像、文本和编码后文本的字典
            'image': image,
            'text': text,
            'input_ids': encoded['input_ids'].squeeze(), # 移除多余的维度
            'attention_mask': encoded['attention_mask'].squeeze() # 移除多余的维度
        }

# =============================================================================
# 工具函数 (Utility Functions)
# =============================================================================

def check_gpu_memory(): # 检查 GPU 显存使用情况
    """检查GPU内存使用情况"""
    if torch.cuda.is_available(): # 如果 CUDA 可用
        allocated = torch.cuda.memory_allocated() / 1024**3  # 已分配显存 (GB)
        reserved = torch.cuda.memory_reserved() / 1024**3   # 已预留显存 (GB)
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB") # 打印信息
        return allocated, reserved # 返回值
    return 0, 0 # 如果 CUDA 不可用，返回 0

def clear_gpu_cache(): # 清理 GPU 缓存
    """清理GPU缓存"""
    if torch.cuda.is_available(): # 如果 CUDA 可用
        torch.cuda.empty_cache() # 调用 PyTorch 的接口清理未使用的缓存

def save_model(model, optimizer, epoch, loss, save_path): # 保存模型检查点
    """保存模型检查点"""
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本所在目录
    full_save_path = os.path.join(current_dir, save_path) # 拼接成完整保存路径
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True) # 确保保存目录存在
    torch.save({ # 保存一个字典
        'epoch': epoch, # 当前训练轮次
        'model_state_dict': model.state_dict(), # 模型的状态字典
        'optimizer_state_dict': optimizer.state_dict(), # 优化器的状态字典
        'loss': loss, # 当前的损失值
    }, full_save_path) # 保存到指定路径
    print(f"Model saved to {full_save_path}") # 打印保存信息

def load_model_if_exists(model, optimizer, checkpoint_path): # 加载模型检查点
    """如果存在已训练的模型检查点，则加载它"""
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前脚本目录
    full_checkpoint_path = os.path.join(current_dir, checkpoint_path) # 拼接完整路径
    
    if os.path.exists(full_checkpoint_path): # 如果检查点文件存在
        print(f"发现已存在的模型检查点: {full_checkpoint_path}") # 打印信息
        print("正在加载模型...")
        
        checkpoint = torch.load(full_checkpoint_path, map_location='cpu') # 加载检查点到 CPU (避免 GPU 内存问题)
        model.load_state_dict(checkpoint['model_state_dict']) # 加载模型权重
        
        if optimizer is not None: # 如果提供了优化器
            optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # 加载优化器状态
        
        epoch = checkpoint.get('epoch', 0) # 获取保存的 epoch，若没有则默认为 0
        loss = checkpoint.get('loss', 0.0) # 获取保存的 loss，若没有则默认为 0.0
        
        print(f"模型加载成功！训练轮次: {epoch + 1}, 损失: {loss:.4f}") # 打印加载成功信息
        return True, epoch, loss # 返回成功标志、epoch 和 loss
    else: # 如果检查点文件不存在
        print(f"未找到模型检查点: {full_checkpoint_path}") # 打印信息
        return False, 0, float('inf') # 返回失败标志和默认值

def plot_training_curve(losses, save_path="training_curve.png"): # 绘制训练损失曲线
    """绘制训练曲线"""
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录
    full_save_path = os.path.join(current_dir, save_path) # 拼接完整保存路径
    
    plt.figure(figsize=(10, 6)) # 创建一个新的图形
    plt.plot(losses) # 绘制损失曲线
    plt.title('CLIP Training Loss') # 设置标题
    plt.xlabel('Step') # 设置 x 轴标签
    plt.ylabel('Loss') # 设置 y 轴标签
    plt.grid(True) # 显示网格
    plt.savefig(full_save_path) # 保存图像文件
    plt.close()  # 关闭图形，避免在内存中累积
    print(f"Training curve saved to {full_save_path}") # 打印保存信息

def compute_similarity(image_features, text_features): # 计算图文特征的相似度
    """计算图像和文本特征的相似度"""
    # 再次归一化 (虽然模型输出已归一化，但作为独立函数，保持稳健性)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度 (点积)
    similarity = torch.matmul(image_features, text_features.t()) # 矩阵乘法
    return similarity

def test_model_inference(model, dataloader, device, num_samples=5): # 测试模型推理效果
    """测试模型推理效果"""
    model.eval() # 将模型设置为评估模式
    
    with torch.no_grad(): # 在此块内不计算梯度
        for i, batch in enumerate(dataloader): # 遍历数据加载器
            if i >= num_samples: # 如果达到指定的测试样本批次数
                break # 退出循环
                
            images = batch['image'].to(device) # 将图像移到指定设备
            input_ids = batch['input_ids'].to(device) # 将文本 ID 移到指定设备
            attention_mask = batch['attention_mask'].to(device) # 将注意力掩码移到指定设备
            texts = batch['text'] # 获取原始文本列表
            
            # 获取特征
            image_features = model.get_image_features(images) # 获取图像特征
            text_features = model.get_text_features(input_ids, attention_mask) # 获取文本特征
            
            # 计算相似度
            similarity = compute_similarity(image_features, text_features) # 计算批内图文相似度矩阵
            
            print(f"\nBatch {i+1}:") # 打印批次信息
            for j in range(min(3, len(texts))):  # 只显示当前批次的前 3 个样本
                print(f"Text: {texts[j]}") # 打印当前文本
                print(f"Similarity scores: {similarity[j].cpu().numpy()}") # 打印该文本与批内所有图像的相似度分数
                predicted_idx = similarity[j].argmax().item() # 找到相似度最高的图像索引
                print(f"Best match: {texts[predicted_idx]} (score: {similarity[j, predicted_idx]:.4f})") # 打印最佳匹配的文本和分数
                print("-" * 50) # 分隔符

# =============================================================================
# 简单的本地 Tokenizer (Simple Local Tokenizer)
# =============================================================================

class SimpleTokenizer: # 定义一个简单的 tokenizer
    """
    简单的本地 tokenizer，避免网络连接问题
    """
    def __init__(self, vocab_size=5000): # 初始化
        self.vocab_size = vocab_size # 词汇表大小
        # 预定义特殊 token
        self.word2idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx2word = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_built = False # 词汇表是否已构建的标志
        
    def build_vocab(self, texts): # 构建词汇表
        """从文本构建词汇表"""
        if self.vocab_built: # 如果已经构建过，则直接返回
            return
            
        word_counts = Counter() # 创建一个计数器
        for text in texts: # 遍历所有文本
            words = self.tokenize_text(text) # 对文本进行分词
            word_counts.update(words) # 更新词频
        
        # 选择最常见的 N 个词 (N = vocab_size - 4)
        most_common = word_counts.most_common(self.vocab_size - 4)
        
        for i, (word, _) in enumerate(most_common): # 遍历最常见的词
            idx = i + 4  # 索引从 4 开始，因为前 4 个被特殊 token 占用
            self.word2idx[word] = idx # 添加到词到索引的映射
            self.idx2word[idx] = word # 添加到索引到词的映射
        
        self.vocab_built = True # 标记为已构建
        print(f"Built vocabulary with {len(self.word2idx)} words") # 打印词汇表大小
    
    def tokenize_text(self, text): # 简单的分词函数
        """简单的文本分词"""
        text = text.lower() # 转换为小写
        # 使用正则表达式移除所有非字母数字和空格的字符
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split() # 按空格分割
    
    def __call__(self, text, truncation=True, padding='max_length', max_length=77, return_tensors='pt'): # 使类实例可调用
        """模拟 transformers tokenizer 的接口"""
        words = self.tokenize_text(text) # 分词
        
        # 转换为索引
        input_ids = [self.word2idx.get('[CLS]', 2)]  # 添加起始 token [CLS]
        for word in words: # 遍历单词
            input_ids.append(self.word2idx.get(word, self.word2idx['[UNK]'])) # 转换为索引，未知词用 [UNK]
        
        # 截断
        if truncation and len(input_ids) > max_length - 1: # 如果开启截断且长度超限
            input_ids = input_ids[:max_length - 1] # 截断到最大长度减一 (为 [SEP] 留出位置)
        
        # 添加结束 token
        input_ids.append(self.word2idx.get('[SEP]', 3)) # 添加结束 token [SEP]
        
        # 填充
        attention_mask = [1] * len(input_ids) # 创建注意力掩码，有效部分为 1
        while len(input_ids) < max_length: # 如果长度不足
            input_ids.append(self.word2idx['[PAD]']) # 用 [PAD] 的索引填充
            attention_mask.append(0) # 对应位置的掩码为 0
        
        if return_tensors == 'pt': # 如果要求返回 PyTorch 张量
            import torch # 导入 torch
            return {
                'input_ids': torch.tensor([input_ids]), # 转换为张量并增加 batch 维度
                'attention_mask': torch.tensor([attention_mask]) # 转换为张量并增加 batch 维度
            }
        else: # 否则返回列表
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# =============================================================================
# 主训练函数 (Main Training Function)
# =============================================================================

def main(): # 主函数
    # 配置参数 - 针对 4060 显卡优化
    parser = argparse.ArgumentParser(description='Train CLIP model') # 创建参数解析器
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32 for 4060)') # 批大小
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs') # 训练轮数
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # 学习率
    parser.add_argument('--data_dir', type=str, default='sample_data', help='Data directory') # 数据目录
    parser.add_argument('--embedding_dim', type=int, default=256, help='Shared embedding dimension') # 共享嵌入维度
    parser.add_argument('--force_train', action='store_true', help='强制重新训练，即使存在已训练的模型') # 强制训练标志
    parser.add_argument('--test_only', action='store_true', help='仅进行测试，不训练') # 仅测试标志
    
    args = parser.parse_args() # 解析命令行参数
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu" # 判断使用 CUDA 还是 CPU
    print(f"Using device: {device}") # 打印设备信息
    if device == "cuda": # 如果使用 CUDA
        print(f"GPU: {torch.cuda.get_device_name()}") # 打印 GPU 型号
        check_gpu_memory() # 检查初始显存
    
    # 检查数据集是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录
    data_path = os.path.join(current_dir, args.data_dir) # 获取数据路径
    
    if not os.path.exists(data_path): # 如果数据路径不存在
        print(f"数据集目录 {data_path} 不存在！") # 打印错误信息
        print("请先运行 create_dataset.py 来创建数据集") # 提示用户
        return # 退出程序
    
    # 数据预处理
    image_transforms = transforms.Compose([ # 定义一系列图像变换
        transforms.Resize((224, 224)), # 调整图像大小到 224x224
        transforms.ToTensor(), # 将 PIL 图像或 NumPy 数组转换为张量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 标准化，使用 ImageNet 的均值和标准差
    ])
    
    # 创建数据集和数据加载器
    dataset = SimpleImageTextDataset( # 实例化数据集
        data_dir=data_path,
        transform=image_transforms
    )
    
    dataloader = DataLoader( # 实例化数据加载器
        dataset, 
        batch_size=args.batch_size, # 设置批大小
        shuffle=True, # 每个 epoch 开始时打乱数据
        num_workers=2,  # 使用 2 个子进程加载数据，减少内存占用
        pin_memory=True if device == "cuda" else False # 如果使用 GPU，将数据锁在内存中，加速传输
    )
    
    print(f"Dataset size: {len(dataset)}") # 打印数据集大小
    print(f"Batch size: {args.batch_size}") # 打印批大小
    print(f"Number of batches: {len(dataloader)}") # 打印批次数
    
    # 创建模型 - 使用轻量级配置
    print("Creating models...") # 打印信息
    try: # 尝试加载预训练权重
        image_encoder = ImageEncoder(model_name='resnet18', pretrained=True)
    except Exception as e: # 如果失败 (例如网络问题)
        print(f"无法下载预训练权重: {e}") # 打印错误
        print("使用随机初始化的权重...") # 提示用户
        image_encoder = ImageEncoder(model_name='resnet18', pretrained=False) # 使用未预训练的模型
    
    text_encoder = TextEncoder(vocab_size=5000, hidden_size=256, num_layers=2) # 创建文本编码器
    model = CLIP(image_encoder, text_encoder, shared_embedding_dim=args.embedding_dim) # 创建 CLIP 主模型
    model = model.to(device) # 将模型移动到指定设备
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters()) # 总参数量
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # 可训练参数量
    print(f"Total parameters: {total_params:,}") # 打印总参数量
    print(f"Trainable parameters: {trainable_params:,}") # 打印可训练参数量
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01) # 使用 AdamW 优化器
    
    # 检查是否存在已训练的模型
    model_loaded = False # 模型加载成功标志
    best_loss = float('inf') # 最佳损失，初始化为无穷大
    start_epoch = 0 # 开始的 epoch
    
    if not args.force_train: # 如果不强制重新训练
        # 定义检查点文件的查找顺序
        checkpoint_files = [
            "checkpoints/best_model.pt",
            "checkpoints/final_model.pt",
        ]
        
        current_dir = os.path.dirname(os.path.abspath(__file__)) # 获取当前目录
        checkpoints_dir = os.path.join(current_dir, "checkpoints") # 检查点目录
        if os.path.exists(checkpoints_dir): # 如果目录存在
            epoch_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("epoch_") and f.endswith(".pt")] # 查找所有 epoch 文件
            epoch_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)  # 按 epoch 编号降序排列
            checkpoint_files.extend([f"checkpoints/{f}" for f in epoch_files]) # 将 epoch 文件添加到查找列表
        
        for checkpoint_file in checkpoint_files: # 遍历所有可能的检查点文件
            model_loaded, loaded_epoch, loaded_loss = load_model_if_exists(model, optimizer, checkpoint_file) # 尝试加载
            if model_loaded: # 如果加载成功
                best_loss = loaded_loss # 更新最佳损失
                start_epoch = loaded_epoch + 1 # 设置开始的 epoch
                print(f"使用检查点: {checkpoint_file}") # 打印使用的检查点信息
                break # 成功加载后即退出循环
    
    # 决定是否需要训练
    should_train = not model_loaded or args.force_train # 如果模型未加载或强制训练，则需要训练
    if args.test_only: # 如果是仅测试模式
        should_train = False # 不进行训练
        if not model_loaded: # 如果模型也未加载
            print("错误：指定了仅测试模式，但未找到已训练的模型！") # 打印错误
            print("请先训练模型或使用 --force_train 参数") # 提示用户
            return # 退出
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) # 使用余弦退火调度器
    
    if should_train: # 如果需要训练
        print(f"\n开始训练CLIP模型... (从第 {start_epoch + 1} 轮开始)") # 打印训练开始信息
        model.train() # 将模型设置为训练模式
        
        losses = [] # 用于记录每一步的损失
        
        for epoch in range(start_epoch, args.epochs): # 开始 epoch 循环
            epoch_losses = [] # 用于记录当前 epoch 的损失
        
            pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}') # 使用 tqdm 创建进度条
            
            for batch_idx, batch in enumerate(pbar): # 遍历批次
                # 数据移动到 GPU
                images = batch['image'].to(device)
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # 前向传播
                image_features, text_features = model(images, input_ids, attention_mask)
                
                # 计算相似度矩阵
                logit_scale = model.logit_scale.exp() # 获取温度系数并取指数
                logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale # 计算图像到文本的 logits
                logits_per_text = logits_per_image.t() # 转置得到文本到图像的 logits
                
                # 计算损失 (对比损失)
                batch_size = images.shape[0] # 获取批大小
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=device) # 创建标签，对角线上的样本为正样本
                
                loss_image = nn.functional.cross_entropy(logits_per_image, ground_truth) # 计算图像侧的交叉熵损失
                loss_text = nn.functional.cross_entropy(logits_per_text, ground_truth) # 计算文本侧的交叉熵损失
                total_loss = (loss_image + loss_text) / 2 # 总损失为两者平均
                
                # 反向传播
                optimizer.zero_grad() # 清空之前的梯度
                total_loss.backward() # 计算当前梯度
                
                # 梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step() # 更新模型参数
                
                # 记录损失
                losses.append(total_loss.item()) # 记录总损失
                epoch_losses.append(total_loss.item()) # 记录 epoch 损失
                
                # 更新进度条的后缀信息
                pbar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                    'Scale': f'{logit_scale.item():.2f}'
                })
                
                # 定期清理 GPU 缓存，释放一些显存
                if batch_idx % 50 == 0:
                    clear_gpu_cache()
            
            # 更新学习率
            scheduler.step() # 在每个 epoch 结束后更新学习率
            
            # 计算 epoch 平均损失
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            print(f"\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model, optimizer, epoch, avg_loss, "checkpoints/best_model.pt")
            
            # 定期保存检查点 (例如每 5 个 epoch)
            if (epoch + 1) % 5 == 0:
                save_model(model, optimizer, epoch, avg_loss, f"checkpoints/epoch_{epoch+1}.pt")
            
            # 检查 GPU 内存
            if device == "cuda":
                check_gpu_memory()
    
        print("训练完成！") # 打印完成信息
        
        # 绘制训练曲线
        if losses:
            plot_training_curve(losses, "training_curve.png")
        
        # 保存最终模型
        save_model(model, optimizer, args.epochs-1, best_loss, "checkpoints/final_model.pt")
    else: # 如果不训练
        print("跳过训练，使用已加载的模型进行测试")
    
    # 测试模型（无论是否训练都执行）
    print("\n测试模型推理效果...")
    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False) # 创建一个小的测试数据加载器
    test_model_inference(model, test_dataloader, device, num_samples=3) # 调用测试函数

if __name__ == "__main__": # 如果该脚本作为主程序运行
    main() # 调用主函数
```

---

# CLIP 模型代码的深度逻辑与结构分析

## 摘要

这份代码不仅是一个深度学习模型的实现，更是一个小型、完整且遵循良好软件工程实践的项目。其核心思想是 **解耦 (Decoupling)** 和 **模块化 (Modularization)**。代码结构清晰地分为了四个主要层次：**数据层**、**模型层**、**控制/训练层** 和 **工具层**。这种分层设计使得每一部分职责单一，易于理解、修改和扩展。

---

## I. 宏观架构：分层与解耦

代码的整体结构可以看作一个经典的金字塔模型，上层依赖下层，各层之间通过明确的接口进行通信。

```
+---------------------------------+
|      控制/训练层 (main.py)      |  <-- 程序的总指挥，连接数据和模型，执行训练流程
+---------------------------------+
                |
                v (调用/实例化)
+---------------------------------+
|         模型层 (Classes)        |  <-- 核心计算引擎 (CLIP, ImageEncoder, TextEncoder)
+---------------------------------+
                |
                v (消耗/处理)
+---------------------------------+
|         数据层 (Classes)        |  <-- 数据的来源和预处理 (Dataset, DataLoader, Tokenizer)
+---------------------------------+
|         工具层 (Functions)      |  <-- 可重用的辅助功能 (save/load, plot, check_gpu)
+---------------------------------+
```

1.  **数据层 (Data Layer)**:
    *   **组件**: `SimpleTokenizer`, `SimpleImageTextDataset`, `DataLoader`.
    *   **职责**: 负责所有与数据相关的任务：从磁盘读取原始数据 (`dataset.json`, 图像文件)，将文本转换为数字 ID (tokenization)，对图像进行预处理和增强 (transforms)，最后将它们打包成模型可以直接使用的 `batch`。
    *   **设计优势**: 数据层是完全独立的。它不知道上层会有什么样的模型来使用这些数据。这意味着，如果我们想换一个不同的模型（比如 VGG + GRU），我们完全不需要修改数据层的任何代码。

2.  **模型层 (Model Layer)**:
    *   **组件**: `ImageEncoder`, `TextEncoder`, `CLIP`.
    *   **职责**: 定义了模型的计算图（Computational Graph）。它接收数据层提供的张量 (tensors)，执行一系列复杂的数学运算（卷积、注意力、线性变换等），并输出最终的特征嵌入。
    *   **设计优势**: 模型层对数据来源一无所知。它只关心输入张量的形状和类型。这种抽象使得模型具有高度的可移植性，可以轻松地在不同的项目或数据集上重用。

3.  **控制/训练层 (Control/Training Layer)**:
    *   **组件**: `main()` 函数。
    *   **职责**: 它是整个项目的“大脑”和“指挥官”。它负责：
        *   **实例化**: 创建数据层和模型层的对象。
        *   **连接**: 将 `Dataset` 对象送入 `DataLoader`，再将 `DataLoader` 的输出送入 `CLIP` 模型。
        *   **执行**: 管理训练循环 (epoch 和 batch 遍历)，调用损失函数，执行反向传播和优化器步骤。
        *   **状态管理**: 处理模型的保存、加载、学习率调度和训练过程的可视化。
    *   **设计优势**: 将训练逻辑与模型定义分离开来。这使得我们可以轻松地改变训练策略（例如，从标准的监督学习切换到强化学习或生成对抗网络训练），而无需修改底层的 `CLIP` 模型结构。

4.  **工具层 (Utility Layer)**:
    *   **组件**: `check_gpu_memory`, `save_model`, `plot_training_curve` 等独立函数。
    *   **职责**: 提供通用的、无状态的辅助功能。
    *   **设计优势**: 这些函数是可重用的代码片段，可以被项目中的任何部分调用，甚至可以被复制到其他项目中，提高了代码的复用性。

---

## II. 核心组件深度解析

### `ImageEncoder` & `TextEncoder`：专家的分工

这两个类体现了 **单一职责原则**。

*   **`ImageEncoder`**: 它的唯一任务是理解图像。它通过 `ResNet` 的卷积层逐步从像素中提取层次化的特征：从边缘、纹理到更复杂的物体部件，最终形成一个能代表图像核心内容的向量。移除最后一层是关键，因为我们需要的不是一个指向特定类别的“答案”（如“猫”或“狗”），而是一个丰富的、可供比较的“描述”。
*   **`TextEncoder`**: 它的唯一任务是理解文本。通过 `Transformer` 的自注意力机制，它能捕捉到句子中单词之间的复杂依赖关系（例如，“a man throwing a frisbee in a park” 中，“throwing” 的主语是 “man”，宾语是 “frisbee”）。返回 `[CLS]` token 的输出是一种惯例，它被训练为聚合整个句子的语义信息。

### `CLIP` 类：多模态的融合与对齐

`CLIP` 类是整个项目的核心，其内部逻辑精妙地解决了多模态学习的关键挑战。

1.  **挑战**: 图像和文本是两种完全不同的数据模态，它们的原始特征空间（像素空间 vs. 词汇空间）毫无关联。如何让它们可以相互比较？
2.  **解决方案：共享嵌入空间 (Shared Embedding Space)**
    
    *   **投射头 (Projection Head)**: `image_projection` 和 `text_projection` 是实现这一目标的关键。它们就像是两种不同语言（图像语言和文本语言）的“翻译官”。无论 `ImageEncoder` 和 `TextEncoder` 输出的特征维度是多少，这两个投射头都会将它们“翻译”到一个统一的、维度相同的向量空间中。这个空间就是共享嵌入空间。
    *   **对齐 (Alignment)**: 在这个共享空间里，如果一个图像和一个文本描述的是相同的内容，它们的向量在空间中的位置就应该非常接近。这就是“对齐”的含义，也是对比学习的目标。
    
3.  **L2 归一化：将比较简化为方向一致性**
    
    *   **为什么需要归一化？** 在高维空间中，向量的长度（模）可能会成为一个干扰因素。一个特征很强的向量可能仅仅因为它长度较长而与许多其他向量有较高的点积，但这并不代表它们的语义更相似。
    *   **归一化的作用**: 通过 `v / v.norm()`，所有特征向量都被映射到了单位超球面的表面。此时，所有向量的长度都为 1。在这种情况下，两个向量之间的点积（Dot Product）在数学上等价于它们夹角的余弦值（Cosine Similarity）。
        $$
        \text{cos}(\theta) = \frac{A \cdot B}{\|A\| \|B\|}
        $$
        当 $\|A\| = \|B\| = 1$ 时，$\text{cos}(\theta) = A \cdot B$。
    *   **带来的好处**: 这意味着我们不再关心向量的长度，只关心它们的方向。如果两个向量指向超球面上的同一个方向（或非常接近的方向），它们的余弦相似度就接近 1，代表语义高度相关。这使得相似度度量更加纯粹和稳定。

---

## III. 核心算法：对比损失 (Contrastive Loss) 的智慧

CLIP 的训练灵魂在于其损失函数的设计。它没有使用传统的“输入 X，预测 Y”的监督学习范式，而是采用了一种更巧妙的自监督学习方法。

1.  **基本思想**: 在一个批次 (batch) 中，对于任何一个图像 $I_i$，其配对的文本 $T_i$ 是它的 **正样本 (Positive Sample)**。所有其他的文本 $T_j$ (其中 $j \neq i$) 都是它的 **负样本 (Negative Samples)**。反之亦然。

2.  **数学实现**:
    *   假设一个批次大小为 $N$，我们有图像特征 $\{img_1, img_2, ..., img_N\}$ 和文本特征 $\{txt_1, txt_2, ..., txt_N\}$。
    *   我们计算出一个 $N \times N$ 的相似度矩阵 $S$，其中 $S_{ij} = \text{cosine\_similarity}(img_i, txt_j)$。
    *   乘以可学习的温度系数 $e^T$ 进行缩放，得到 logits 矩阵 $L = S \times e^T$。
    *   **对于第 $i$ 行 (图像 $i$)**: 这一行是 `[L_i1, L_i2, ..., L_ii, ..., L_iN]`。我们的目标是让模型认为第 $i$ 个文本是正确的。这本质上是一个 $N$ 分类问题，其中正确的类别标签是 $i$。
    *   **交叉熵损失**: 因此，我们可以直接套用交叉熵损失函数。对于图像侧的损失，我们希望模型从 $N$ 个文本中正确地“挑出”与每个图像配对的那个。
        $$
        \mathcal{L}_{\text{image}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(L_{ii})}{\sum_{j=1}^{N} \exp(L_{ij})}
        $$
    *   同理，将 logits 矩阵 $L$ 转置，就可以计算文本侧的损失 $\mathcal{L}_{\text{text}}$。
    *   最终的总损失是两者的平均：$\mathcal{L}_{\text{total}} = (\mathcal{L}_{\text{image}} + \mathcal{L}_{\text{text}}) / 2$。

3.  **设计的高明之处**:
    *   **数据效率**: 每一个样本既作为一次正样本，又作为 $N-1$ 次负样本，极大地提高了数据利用率。
    *   **无需硬标签**: 它不需要人工标注“这张图里有什么”，只需要知道“这张图和这段文字是一对”即可。这使得利用海量的网络图文对进行预训练成为可能。
    *   **零样本能力 (Zero-shot Capability)**: 通过这种方式训练，模型学会的不是识别固定的几个类别，而是图文之间的通用语义关系。因此，在推理时，我们可以给它任何新的文本描述，它都能计算出与图像的相似度，从而实现对从未见过的类别的识别。

---

## IV. 总结：一份教科书式的项目实践

这份代码的结构和逻辑展示了构建一个成功的深度学习项目的关键要素：

*   **清晰的抽象层次**: 将复杂系统分解为独立的、易于管理的部分。
*   **面向接口编程**: 各层之间通过定义好的数据格式（如字典、张量）进行交互，而不是紧密耦合在一起。
*   **核心算法的深刻理解**: 对比损失的巧妙运用是模型能够成功学习的关键。
*   **鲁棒性和易用性**: 包含了检查点管理、参数配置、异常处理和结果可视化，构成了一个完整的、用户友好的开发和实验流程。

通过学习和理解这份代码，我们不仅能掌握 CLIP 模型的原理，更能学到如何组织和构建一个高质量的机器学习项目。

------

### 代码里ResNet 和Transformer的进一步解释：

**ResNet 是预训练的，而这个代码里的 Transformer 是从零开始训练的。但最关键的是，在我们的 CLIP 训练过程中，它们 *全都会* 被进一步训练（这个过程通常称为“微调”或 Fine-tuning）。**

---

## 深度解析：模型层中的预训练与从零训练

您的疑问非常准确地指向了模型初始化的两种不同策略。让我们逐一分析 `ImageEncoder` 和 `TextEncoder`。

### 1. `ImageEncoder` (ResNet): 站在巨人的肩膀上

这是您代码中的相关部分：
```python
class ImageEncoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True): # 注意这个 pretrained=True
        super().__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained) # 关键在这里
            # ...
```

*   **`pretrained=True` 的含义是什么？**
    当您设置 `pretrained=True` 时，`torchvision` 库会自动下载一个已经在 **ImageNet 数据集** 上训练好的 ResNet-18 模型权重。ImageNet 是一个包含超过 120 万张图像和 1000 个类别的大规模数据集。

*   **这个预训练好的 ResNet 已经具备了什么能力？**
    它已经是一个非常强大的通用视觉特征提取器。它不是一张“白纸”，而是已经具备了基础的视觉理解能力，例如：
    *   **底层特征**: 识别边缘、颜色、纹理。
    *   **中层特征**: 识别形状、图案，甚至是物体的部件（如眼睛、轮子）。
    *   **高层特征**: 识别完整的物体概念（如猫、汽车、树）。

*   **那在我们的 CLIP 训练中，它还需要训练吗？**
    **需要！** 这就是所谓的 **微调 (Fine-tuning)**。虽然 ResNet 已经很强大了，但它的“知识”是为了进行 1000 分类任务而优化的。我们的任务是学习图文对齐，目标略有不同。因此，在 CLIP 的训练过程中：
    1.  我们从这个强大的预训练起点开始，而不是从随机权重开始。
    2.  CLIP 的对比损失所产生的梯度会反向传播回 `ImageEncoder`。
    3.  这些梯度会 **微调** ResNet 的权重，使其提取的视觉特征不仅能识别物体，而且更适合与文本描述进行对齐。

    **这个过程就像一个已经大学毕业的通才（预训练的 ResNet），为了适应一份新工作（图文对齐），而去学习一些更专业的技能（微调）。**

### 2. `TextEncoder` (Transformer): 从零开始，艰苦奋斗

现在我们看文本编码器：
```python
class TextEncoder(nn.Module):
    def __init__(self, vocab_size=5000, hidden_size=256, num_layers=2):
        super().__init__()
        # ...
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # ...
        encoder_layer = nn.TransformerEncoderLayer(...)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
```

*   **它是否是预训练的？**
    **不是。** 在您的这段代码中，`nn.Embedding`, `nn.TransformerEncoderLayer` 等所有组件都是使用 PyTorch 的默认方式进行 **随机初始化** 的。

*   **这意味着什么？**
    这个 `TextEncoder` 在训练开始时是一张完全的“白纸”。它对语言一无所知：
    
    *   它不知道单词的含义。
    *   它不理解语法结构。
    *   它无法分辨“猫追老鼠”和“老鼠追猫”的区别。
    
*   **它如何学习？**
    它完全依赖于 CLIP 的对比学习任务来从零开始学习关于语言的一切。
    1.  当一个文本描述 $T_i$ 与其对应的图像 $I_i$ 配对成功时，模型会得到一个正向的奖励信号。
    2.  当它与不匹配的图像 $I_j$ 配对时，会得到一个负向的惩罚信号。
    3.  通过亿万次这样的“试错”，`TextEncoder` 会逐渐学习到：
        *   哪些词在语义上是相似的（例如，“小狗”和“犬”）。
        *   如何组合单词来表达复杂的含义。
        *   如何生成一个能够代表整句话核心思想的特征向量。

    **这个过程就像一个婴儿从零开始学习一门语言，完全通过观察世界（图像）和倾听描述（文本）之间的关联来学习。**

> **注**: 在更大型的 CLIP 实现中（如 OpenAI 的版本），文本编码器通常也使用预训练好的模型（例如 GPT-2 或 BERT 的变体），这样可以获得更好的性能。但在这个为教学和优化目的而简化的代码中，从零开始训练文本编码器是一个非常合理且常见的做法。

### 总结与对比

| 特性         | `ImageEncoder` (ResNet)                                      | `TextEncoder` (Transformer)                                  |
| :----------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| **初始状态** | **预训练 (Pre-trained)**                                     | **随机初始化 (From Scratch)**                                |
| **初始知识** | 具备强大的通用视觉理解能力                                   | 对语言一无所知，是一张白纸                                   |
| **训练过程** | **微调 (Fine-tuning)**：在现有知识基础上进行调整和优化，以适应新任务。 | **从零训练 (Training from scratch)**：学习所有关于语言的知识。 |
| **学习效率** | 起点高，学习速度相对较快。                                   | 起点低，需要更多数据和时间来学习。                           |
| **最终目标** | 共同为 CLIP 的对比学习任务服务，输出可以相互对齐的特征向量。 | 共同为 CLIP 的对比学习任务服务，输出可以相互对齐的特征向量。 |

所以，您的理解非常到位。模型层并不是一个整体，它的不同部分可以有不同的“出身”和学习策略。这种结合 **预训练+微调** 和 **从零训练** 的混合模式是深度学习中非常强大和常用的一种技术。
