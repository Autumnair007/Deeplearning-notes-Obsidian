import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 设置随机种子，以确保结果的可重复性
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


# 定义Multi-Head Self-Attention模块
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, print_tensor_info=False):
        batch_size, tokens, embed_dim = x.shape

        if print_tensor_info:
            print(f"\n------ 多头自注意力机制 (Multi-Head Self-Attention) ------")
            print(f"输入张量形状: {x.shape} [批次大小, token数量, 嵌入维度]")
            print(f"第一个token的值 (示例): {x[0, 0, :5]}... (只显示前5个值)")

        qkv = self.qkv(x)  # [batch_size, tokens, 3 * embed_dim]
        if print_tensor_info:
            print(f"\n1. QKV线性投影后形状: {qkv.shape}")
            print(f"   这一步将每个token从 {embed_dim} 维投影到 {3 * embed_dim} 维，包含Q、K、V三部分")

        qkv = qkv.reshape(batch_size, tokens, 3, self.num_heads, self.head_dim)
        if print_tensor_info:
            print(f"\n2. 重塑QKV张量形状: {qkv.shape}")
            print(f"   将3*embed_dim拆分成[3, num_heads, head_dim]，其中3代表Q、K、V")

        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, tokens, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each [batch_size, num_heads, tokens, head_dim]
        if print_tensor_info:
            print(f"\n3. 拆分后Q形状: {q.shape}, K形状: {k.shape}, V形状: {v.shape}")
            print(f"   现在Q、K、V被分离，每个头都有自己的 {self.head_dim} 维度空间")

        # Scaled dot-product attention
        k_t = k.transpose(-2, -1)  # [batch_size, num_heads, head_dim, tokens]
        if print_tensor_info:
            print(f"\n4. 转置K后形状: {k_t.shape}")
            print(f"   转置是为了准备矩阵乘法计算注意力分数")

        dots = (q @ k_t) / (self.head_dim ** 0.5)  # [batch_size, num_heads, tokens, tokens]
        if print_tensor_info:
            print(f"\n5. 注意力分数矩阵形状: {dots.shape}")
            print(
                f"   用Q乘以K的转置，并除以缩放因子sqrt({self.head_dim})={self.head_dim ** 0.5:.2f}，得到token间的相关性分数")

        attn = F.softmax(dots, dim=-1)
        if print_tensor_info:
            print(f"\n6. Softmax后注意力权重形状: {attn.shape}")
            print(f"   对每行应用softmax，将分数转换为概率分布")
            if attn.shape[2] < 10:  # 只有在token数量较少时打印
                print(f"   第一个头的第一个token的注意力权重: {attn[0, 0, 0]}")

        out = attn @ v  # [batch_size, num_heads, tokens, head_dim]
        if print_tensor_info:
            print(f"\n7. 注意力加权后形状: {out.shape}")
            print(f"   将注意力权重与V相乘，得到加权后的特征表示")

        out = out.transpose(1, 2)  # [batch_size, tokens, num_heads, head_dim]
        if print_tensor_info:
            print(f"\n8. 重排多头结果形状: {out.shape}")
            print(f"   将多个头的结果重排，准备合并")

        out = out.reshape(batch_size, tokens, embed_dim)  # [batch_size, tokens, embed_dim]
        if print_tensor_info:
            print(f"\n9. 合并多头结果形状: {out.shape}")
            print(f"   将所有头的结果合并回原始嵌入维度")

        out = self.proj(out)
        if print_tensor_info:
            print(f"\n10. 最终投影后形状: {out.shape}")
            print(f"    通过最后的线性层，将多头注意力的输出映射回原始维度")

        return out


# 定义MLP模块
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, print_tensor_info=False):
        if print_tensor_info:
            print(f"\n------ MLP层 ------")
            print(f"输入张量形状: {x.shape}")

        x = self.fc1(x)
        if print_tensor_info:
            print(f"\n1. 第一个线性层后形状: {x.shape}")
            print(f"   将维度从 {self.fc1.in_features} 扩展到 {self.fc1.out_features}")

        x = self.act(x)
        if print_tensor_info:
            print(f"\n2. GELU激活后形状: {x.shape}")
            print(f"   应用GELU激活函数，形状不变但激活了特征")

        x = self.fc2(x)
        if print_tensor_info:
            print(f"\n3. 第二个线性层后形状: {x.shape}")
            print(f"   将维度从 {self.fc2.in_features} 重新映射回 {self.fc2.out_features}")

        return x


# 定义Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, print_tensor_info=False):
        if print_tensor_info:
            print(f"\n====== Transformer编码器块 ======")
            print(f"输入张量形状: {x.shape}")

        # 第一个模块：Layer Norm + Self-Attention + Residual
        x_ln1 = self.ln1(x)
        if print_tensor_info:
            print(f"\n1. 第一层归一化后形状: {x_ln1.shape}")
            print(f"   对每个token执行LayerNorm，保持形状不变但归一化了特征")

        attn_out = self.self_attn(x_ln1, print_tensor_info)
        attn_out = self.dropout(attn_out)
        x = x + attn_out  # 残差连接
        if print_tensor_info:
            print(f"\n2. 残差连接后形状: {x.shape}")
            print(f"   将注意力输出加回原始输入，创建残差连接")

        # 第二个模块：Layer Norm + MLP + Residual
        x_ln2 = self.ln2(x)
        if print_tensor_info:
            print(f"\n3. 第二层归一化后形状: {x_ln2.shape}")
            print(f"   再次应用LayerNorm，准备送入MLP")

        mlp_out = self.mlp(x_ln2, print_tensor_info)
        mlp_out = self.dropout(mlp_out)
        x = x + mlp_out  # 残差连接
        if print_tensor_info:
            print(f"\n4. MLP残差连接后形状: {x.shape}")
            print(f"   将MLP输出加回输入，创建第二个残差连接")

        return x


# 定义Vision Transformer模型
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            embed_dim=256,
            depth=6,
            num_heads=8,
            mlp_ratio=4,
            dropout=0.1,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.depth = depth

        # 计算序列长度：每个图像生成的patches数量
        self.num_patches = (img_size // patch_size) ** 2

        # 线性映射层：将每个patch转换为embed_dim维度的向量
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        # 加上一个额外的[class]标记
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer编码器
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, embed_dim * mlp_ratio, dropout
            )
            for _ in range(depth)
        ])

        self.ln = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, print_tensor_info=False):
        if print_tensor_info:
            print("\n\n========== VISION TRANSFORMER 前向传播过程 ==========")
            print(f"输入图像形状: {x.shape}  [批次大小, 通道数, 高度, 宽度]")
            print(f"ViT配置: 嵌入维度={self.embed_dim}, 头数={self.num_heads}, Transformer层数={self.depth}")

        batch_size = x.shape[0]

        # 将图像分成patches并进行线性映射
        x = self.patch_embed(x)  # [B, C, H/P, W/P]
        if print_tensor_info:
            print(f"\n1. 图像分块后形状: {x.shape}")
            print(f"   将 {self.patch_size}x{self.patch_size} 的图像块映射为 {self.embed_dim} 维向量")
            print(f"   从 [B, 3, 32, 32] 变为 [B, {self.embed_dim}, {32 // self.patch_size}, {32 // self.patch_size}]")

        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        if print_tensor_info:
            print(f"\n2. 展平并转置后形状: {x.shape}")
            print(f"   将空间维度展平并转置，得到序列长度={self.num_patches}的序列")
            print(f"   每个序列元素是一个 {self.embed_dim} 维的向量，表示一个图像块")

        # 添加类标记
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, C]
        if print_tensor_info:
            print(f"\n3. 类标记形状: {cls_token.shape}")
            print(f"   创建可学习的[CLS]标记，用于最终分类")

        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, C]
        if print_tensor_info:
            print(f"\n4. 添加类标记后形状: {x.shape}")
            print(f"   在序列前添加[CLS]标记，序列长度增加1")

        # 添加位置编码
        if print_tensor_info:
            print(f"\n5. 位置编码形状: {self.pos_embed.shape}")
            print(f"   可学习的位置编码，为每个位置提供位置信息")

        x = x + self.pos_embed
        if print_tensor_info:
            print(f"\n6. 添加位置编码后形状: {x.shape}")
            print(f"   将位置信息添加到token嵌入中，使模型了解序列中的位置关系")

        x = self.dropout(x)
        if print_tensor_info:
            print(f"\n7. Dropout后形状: {x.shape}")
            print(f"   应用dropout，防止过拟合")

        # 通过Transformer编码器
        for i, block in enumerate(self.blocks):
            if print_tensor_info and i == 0:
                print(f"\n--- 进入第1个Transformer块 (共{len(self.blocks)}个) ---")
                x = block(x, print_tensor_info)
            elif print_tensor_info and i == len(self.blocks) - 1:
                print(f"\n--- 进入最后一个Transformer块 ---")
                x = block(x, print_tensor_info)
            else:
                x = block(x, False)

        if print_tensor_info:
            print(f"\n8. Transformer编码后形状: {x.shape}")
            print(f"   通过{len(self.blocks)}个Transformer块后，形状保持不变，但特征更丰富")

        x = self.ln(x)
        if print_tensor_info:
            print(f"\n9. 最终LayerNorm后形状: {x.shape}")
            print(f"   对特征进行最后的归一化")

        # 使用[class]标记进行分类
        x = x[:, 0]
        if print_tensor_info:
            print(f"\n10. 提取CLS标记形状: {x.shape}")
            print(f"    只使用第一个位置的[CLS]标记表示整个图像的特征")

        x = self.head(x)
        if print_tensor_info:
            print(f"\n11. 分类头输出形状: {x.shape}")
            print(f"    将[CLS]标记映射到{x.shape[-1]}个类别的logits")
            print("\n========== 前向传播结束 ==========\n")

        return x


# 数据加载和预处理函数
def load_cifar10(batch_size=128, data_dir='cifar10_data'):
    # 定义数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 加载训练集和测试集
    print("Checking if CIFAR-10 dataset exists...")
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("CIFAR-10 dataset loaded successfully!")
    return train_loader, test_loader


# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 添加一个标志，只在第一个批次打印张量信息
    tensor_info_printed = False

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # 在第一个epoch的第一个批次打印张量信息
        print_tensor_info = (epoch == 0 and i == 0 and not tensor_info_printed)
        if print_tensor_info:
            tensor_info_printed = True
            print("\n\n================ 第一次训练时的张量变化说明 ================")
            print("注意：此信息仅在第一个epoch的第一个批次显示一次，用于教学目的\n")

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images, print_tensor_info)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

        if print_tensor_info:
            print("\n============ 张量变化说明完毕 ============")
            print("继续训练过程...\n")

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


# 评估函数
def evaluate(model, test_loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    test_acc = 100. * correct / total
    return test_loss, test_acc


# 绘制训练过程图表
def plot_metrics(train_losses, train_accs, test_losses, test_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, test_losses, 'r-', label='Testing Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, test_accs, 'r-', label='Testing Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('vit_training_metrics.png')
    plt.show()


# 主函数
def main():
    # 定义超参数
    batch_size = 128
    num_epochs = 10
    learning_rate = 3e-4
    weight_decay = 1e-4

    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size)

    # 实例化模型
    model = VisionTransformer(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        embed_dim=192,
        depth=6,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1
    ).to(device)

    # 打印模型架构概要
    print("\n===== Vision Transformer架构概要 =====")
    print(f"图像大小: 32x32")
    print(f"Patch大小: 4x4 (每张图片分成64个patch)")
    print(f"嵌入维度: 192")
    print(f"Transformer层数: 6")
    print(f"注意力头数: 8")
    print(f"MLP比例: 4 (MLP隐藏层维度为 192*4 = 768)")
    print(f"参数总数: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("=====================================\n")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练和评估
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print("-" * 50)

        scheduler.step()

    print("Training completed!")

    # 保存模型
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print("Model saved to 'vit_cifar10.pth'")

    # 绘制训练过程图表
    plot_metrics(train_losses, train_accs, test_losses, test_accs)


if __name__ == "__main__":
    main()