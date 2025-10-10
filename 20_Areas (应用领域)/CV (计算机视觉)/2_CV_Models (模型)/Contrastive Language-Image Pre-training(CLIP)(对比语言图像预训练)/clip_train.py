import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import re
from collections import Counter
import os
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# =============================================================================
# 模型定义部分
# =============================================================================

class ImageEncoder(nn.Module):
    """
    图像编码器：使用torchvision的预训练视觉模型
    针对4060显卡优化，使用较小的模型
    """
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        # 使用torchvision的轻量级模型
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            # 移除最后的分类层
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 512
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.output_dim = 512
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        print(f"Image encoder output dim: {self.output_dim}")

    def forward(self, images):
        # 输入: [batch_size, 3, 224, 224]
        # 输出: [batch_size, output_dim]
        features = self.model(images)
        return features.flatten(1)  # 展平除batch维度外的所有维度

class TextEncoder(nn.Module):
    """
    文本编码器：使用简单的嵌入层和Transformer
    """
    def __init__(self, vocab_size=5000, hidden_size=256, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_dim = hidden_size
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.positional_encoding = nn.Embedding(77, hidden_size)  # 最大长度77
        
        # 简单的Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        print(f"Text encoder output dim: {self.output_dim}")

    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.shape
        
        # 词嵌入
        token_embeddings = self.embedding(input_ids)
        
        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        pos_embeddings = self.positional_encoding(positions)
        
        # 组合嵌入
        embeddings = token_embeddings + pos_embeddings
        
        # 创建注意力掩码（Transformer期望的格式）
        # True表示被忽略的位置
        src_key_padding_mask = (attention_mask == 0)
        
        # Transformer编码
        encoded = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # 返回[CLS] token的表示（第一个位置）
        return encoded[:, 0, :]

class CLIP(nn.Module):
    """
    CLIP主模型：针对4060显卡优化的版本
    """
    def __init__(self, image_encoder, text_encoder, shared_embedding_dim=256):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        # 投射层
        self.image_projection = nn.Sequential(
            nn.Linear(self.image_encoder.output_dim, shared_embedding_dim, bias=False),
            nn.LayerNorm(shared_embedding_dim)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(self.text_encoder.output_dim, shared_embedding_dim, bias=False),
            nn.LayerNorm(shared_embedding_dim)
        )

        # 可学习的温度系数
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, input_ids, attention_mask):
        # 编码
        image_features_raw = self.image_encoder(images)
        text_features_raw = self.text_encoder(input_ids, attention_mask)

        # 投射到共享空间
        image_embedding = self.image_projection(image_features_raw)
        text_embedding = self.text_projection(text_features_raw)

        # L2归一化
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)

        return image_embedding, text_embedding

    def get_image_features(self, images):
        """单独获取图像特征"""
        with torch.no_grad():
            image_features = self.image_encoder(images)
            image_embedding = self.image_projection(image_features)
            return image_embedding / image_embedding.norm(dim=-1, keepdim=True)

    def get_text_features(self, input_ids, attention_mask):
        """单独获取文本特征"""
        with torch.no_grad():
            text_features = self.text_encoder(input_ids, attention_mask)
            text_embedding = self.text_projection(text_features)
            return text_embedding / text_embedding.norm(dim=-1, keepdim=True)

# =============================================================================
# 数据集类定义
# =============================================================================

class SimpleImageTextDataset(Dataset):
    """
    简单的图文对数据集
    """
    def __init__(self, data_dir, transform=None, max_length=77):
        self.data_dir = data_dir
        self.transform = transform
        self.max_length = max_length
        
        # 加载数据集
        with open(os.path.join(data_dir, "dataset.json"), "r") as f:
            self.data = json.load(f)
        
        # 使用简单的本地tokenizer
        self.tokenizer = SimpleTokenizer(vocab_size=5000)
        
        # 构建词汇表
        texts = [item["text"] for item in self.data]
        self.tokenizer.build_vocab(texts)
        
        print(f"Loaded {len(self.data)} samples from {data_dir}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 加载图片 - 处理路径
        image_path = item["image_path"]
        # 如果是绝对路径但文件不存在，尝试使用相对路径
        if not os.path.exists(image_path):
            # 提取文件名，在数据目录中查找
            filename = os.path.basename(image_path)
            image_path = os.path.join(self.data_dir, "images", filename)
        
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # 处理文本
        text = item["text"]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'image': image,
            'text': text,
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }

# =============================================================================
# 工具函数
# =============================================================================

def check_gpu_memory():
    """检查GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        return allocated, reserved
    return 0, 0

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_model(model, optimizer, epoch, loss, save_path):
    """保存模型检查点"""
    # 确保保存路径在当前目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(current_dir, save_path)
    os.makedirs(os.path.dirname(full_save_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, full_save_path)
    print(f"Model saved to {full_save_path}")

def load_model_if_exists(model, optimizer, checkpoint_path):
    """如果存在已训练的模型检查点，则加载它"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_checkpoint_path = os.path.join(current_dir, checkpoint_path)
    
    if os.path.exists(full_checkpoint_path):
        print(f"发现已存在的模型检查点: {full_checkpoint_path}")
        print("正在加载模型...")
        
        checkpoint = torch.load(full_checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint.get('epoch', 0)
        loss = checkpoint.get('loss', 0.0)
        
        print(f"模型加载成功！训练轮次: {epoch + 1}, 损失: {loss:.4f}")
        return True, epoch, loss
    else:
        print(f"未找到模型检查点: {full_checkpoint_path}")
        return False, 0, float('inf')

def plot_training_curve(losses, save_path="training_curve.png"):
    """绘制训练曲线"""
    # 确保保存在当前目录下
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_path = os.path.join(current_dir, save_path)
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('CLIP Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(full_save_path)
    plt.close()  # 关闭图形避免内存泄漏
    print(f"Training curve saved to {full_save_path}")

def compute_similarity(image_features, text_features):
    """计算图像和文本特征的相似度"""
    # 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # 计算余弦相似度
    similarity = torch.matmul(image_features, text_features.t())
    return similarity

def test_model_inference(model, dataloader, device, num_samples=5):
    """测试模型推理效果"""
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts = batch['text']
            
            # 获取特征
            image_features = model.get_image_features(images)
            text_features = model.get_text_features(input_ids, attention_mask)
            
            # 计算相似度
            similarity = compute_similarity(image_features, text_features)
            
            print(f"\nBatch {i+1}:")
            for j in range(min(3, len(texts))):  # 只显示前3个样本
                print(f"Text: {texts[j]}")
                print(f"Similarity scores: {similarity[j].cpu().numpy()}")
                predicted_idx = similarity[j].argmax().item()
                print(f"Best match: {texts[predicted_idx]} (score: {similarity[j, predicted_idx]:.4f})")
                print("-" * 50)

# =============================================================================
# 简单的本地Tokenizer
# =============================================================================

class SimpleTokenizer:
    """
    简单的本地tokenizer，避免网络连接问题
    """
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3}
        self.idx2word = {0: '[PAD]', 1: '[UNK]', 2: '[CLS]', 3: '[SEP]'}
        self.vocab_built = False
        
    def build_vocab(self, texts):
        """从文本构建词汇表"""
        if self.vocab_built:
            return
            
        word_counts = Counter()
        for text in texts:
            words = self.tokenize_text(text)
            word_counts.update(words)
        
        # 选择最常见的词
        most_common = word_counts.most_common(self.vocab_size - 4)  # 减去特殊token
        
        for i, (word, _) in enumerate(most_common):
            idx = i + 4  # 从4开始，因为前4个是特殊token
            self.word2idx[word] = idx
            self.idx2word[idx] = word
        
        self.vocab_built = True
        print(f"Built vocabulary with {len(self.word2idx)} words")
    
    def tokenize_text(self, text):
        """简单的文本分词"""
        text = text.lower()
        # 简单的分词：只保留字母数字，用空格分割
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        return text.split()
    
    def __call__(self, text, truncation=True, padding='max_length', max_length=77, return_tensors='pt'):
        """模拟transformers tokenizer的接口"""
        words = self.tokenize_text(text)
        
        # 转换为索引
        input_ids = [self.word2idx.get('[CLS]', 2)]  # 开始token
        for word in words:
            input_ids.append(self.word2idx.get(word, self.word2idx['[UNK]']))
        
        # 截断
        if truncation and len(input_ids) > max_length - 1:
            input_ids = input_ids[:max_length - 1]
        
        # 添加结束token
        input_ids.append(self.word2idx.get('[SEP]', 3))
        
        # 填充
        attention_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(self.word2idx['[PAD]'])
            attention_mask.append(0)
        
        if return_tensors == 'pt':
            import torch
            return {
                'input_ids': torch.tensor([input_ids]),
                'attention_mask': torch.tensor([attention_mask])
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

# =============================================================================
# 主训练函数
# =============================================================================

def main():
    # 配置参数 - 针对4060显卡优化
    parser = argparse.ArgumentParser(description='Train CLIP model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32 for 4060)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='sample_data', help='Data directory')
    parser.add_argument('--embedding_dim', type=int, default=256, help='Shared embedding dimension')
    parser.add_argument('--force_train', action='store_true', help='强制重新训练，即使存在已训练的模型')
    parser.add_argument('--test_only', action='store_true', help='仅进行测试，不训练')
    
    args = parser.parse_args()
    
    # 检查GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        check_gpu_memory()
    
    # 检查数据集是否存在
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, args.data_dir)
    
    if not os.path.exists(data_path):
        print(f"数据集目录 {data_path} 不存在！")
        print("请先运行 create_dataset.py 来创建数据集")
        return
    
    # 数据预处理
    image_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集和数据加载器
    dataset = SimpleImageTextDataset(
        data_dir=data_path,
        transform=image_transforms
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,  # 减少num_workers以节省内存
        pin_memory=True if device == "cuda" else False
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    
    # 创建模型 - 使用轻量级配置
    print("Creating models...")
    try:
        # 尝试使用预训练权重
        image_encoder = ImageEncoder(model_name='resnet18', pretrained=True)
    except Exception as e:
        print(f"无法下载预训练权重: {e}")
        print("使用随机初始化的权重...")
        image_encoder = ImageEncoder(model_name='resnet18', pretrained=False)
    
    text_encoder = TextEncoder(vocab_size=5000, hidden_size=256, num_layers=2)
    model = CLIP(image_encoder, text_encoder, shared_embedding_dim=args.embedding_dim)
    model = model.to(device)
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # 检查是否存在已训练的模型
    model_loaded = False
    best_loss = float('inf')
    start_epoch = 0
    
    if not args.force_train:
        # 按优先级检查模型文件
        checkpoint_files = [
            "checkpoints/best_model.pt",
            "checkpoints/final_model.pt",
        ]
        
        # 检查是否有epoch检查点
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoints_dir = os.path.join(current_dir, "checkpoints")
        if os.path.exists(checkpoints_dir):
            epoch_files = [f for f in os.listdir(checkpoints_dir) if f.startswith("epoch_") and f.endswith(".pt")]
            epoch_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]), reverse=True)  # 按epoch降序排列
            checkpoint_files.extend([f"checkpoints/{f}" for f in epoch_files])
        
        for checkpoint_file in checkpoint_files:
            model_loaded, loaded_epoch, loaded_loss = load_model_if_exists(model, optimizer, checkpoint_file)
            if model_loaded:
                best_loss = loaded_loss
                start_epoch = loaded_epoch + 1
                print(f"使用检查点: {checkpoint_file}")
                break
    
    # 决定是否需要训练
    should_train = not model_loaded or args.force_train
    if args.test_only:
        should_train = False
        if not model_loaded:
            print("错误：指定了仅测试模式，但未找到已训练的模型！")
            print("请先训练模型或使用 --force_train 参数")
            return
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if should_train:
        # 训练循环
        print(f"\n开始训练CLIP模型... (从第 {start_epoch + 1} 轮开始)")
        model.train()
        
        losses = []
        
        for epoch in range(start_epoch, args.epochs):
            epoch_losses = []
        
        # 使用tqdm显示进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, batch in enumerate(pbar):
            # 数据移动到GPU
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # 前向传播
            image_features, text_features = model(images, input_ids, attention_mask)
            
            # 计算相似度矩阵
            logit_scale = model.logit_scale.exp()
            logits_per_image = torch.matmul(image_features, text_features.t()) * logit_scale
            logits_per_text = logits_per_image.t()
            
            # 计算损失
            batch_size = images.shape[0]
            ground_truth = torch.arange(batch_size, dtype=torch.long, device=device)
            
            loss_image = nn.functional.cross_entropy(logits_per_image, ground_truth)
            loss_text = nn.functional.cross_entropy(logits_per_text, ground_truth)
            total_loss = (loss_image + loss_text) / 2
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            losses.append(total_loss.item())
            epoch_losses.append(total_loss.item())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.2e}',
                'Scale': f'{logit_scale.item():.2f}'
            })
            
            # 定期清理GPU缓存
            if batch_idx % 50 == 0:
                clear_gpu_cache()
        
        # 更新学习率
        scheduler.step()
        
        # 计算epoch平均损失
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        print(f"\nEpoch {epoch+1} 完成, 平均损失: {avg_loss:.4f}")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, optimizer, epoch, avg_loss, "checkpoints/best_model.pt")
        
        # 定期保存检查点
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, avg_loss, f"checkpoints/epoch_{epoch+1}.pt")
        
        # 检查GPU内存
        if device == "cuda":
            check_gpu_memory()
    
        print("训练完成！")
        
        # 绘制训练曲线
        if losses:
            plot_training_curve(losses, "training_curve.png")
        
        # 最终保存
        save_model(model, optimizer, args.epochs-1, best_loss, "checkpoints/final_model.pt")
    else:
        print("跳过训练，使用已加载的模型进行测试")
    
    # 测试模型（无论是否训练都执行）
    print("\n测试模型推理效果...")
    test_dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    test_model_inference(model, test_dataloader, device, num_samples=3)

if __name__ == "__main__":
    main()
