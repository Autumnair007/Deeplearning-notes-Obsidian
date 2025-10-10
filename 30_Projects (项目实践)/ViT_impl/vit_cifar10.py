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
print(f"使用设备: {device}")


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


# 定义多头自注意力机制
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim  # 嵌入维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = embed_dim // num_heads  # 每个头的维度

        # 确保嵌入维度可以被头数整除
        assert self.head_dim * num_heads == embed_dim, "嵌入维度必须能被头数整除"

        # QKV投影：将输入映射到查询(Q)、键(K)和值(V)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        # 输出投影
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, print_info=False):
        batch_size, tokens, embed_dim = x.shape

        if print_info:
            print("\n当前阶段: 自注意力计算")

        # 1. 生成查询(Q)、键(K)、值(V)
        qkv = self.qkv(x)  # [batch_size, tokens, 3 * embed_dim]
        qkv = qkv.reshape(batch_size, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, tokens, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 分别获取Q、K、V

        # 2. 计算注意力分数
        k_t = k.transpose(-2, -1)  # 转置K，准备计算点积
        dots = (q @ k_t) / (self.head_dim ** 0.5)  # 缩放点积注意力
        attn = F.softmax(dots, dim=-1)  # 应用softmax获取注意力权重

        # 3. 使用注意力权重加权值(V)
        out = attn @ v  # [batch_size, num_heads, tokens, head_dim]
        out = out.transpose(1, 2)  # [batch_size, tokens, num_heads, head_dim]
        out = out.reshape(batch_size, tokens, embed_dim)  # 合并多头结果

        # 4. 最终投影
        out = self.proj(out)

        return out


# 定义MLP（多层感知机）模块
class MLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # 第一个全连接层
        self.act = nn.GELU()  # GELU激活函数
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # 第二个全连接层

    def forward(self, x, print_info=False):
        if print_info:
            print("\n当前阶段: MLP处理")

        # 1. 第一个全连接层
        x = self.fc1(x)
        # 2. 应用激活函数
        x = self.act(x)
        # 3. 第二个全连接层
        x = self.fc2(x)

        return x


# 定义Transformer编码器块
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim, dropout=0.1):
        super().__init__()
        # 第一个LayerNorm
        self.ln1 = nn.LayerNorm(embed_dim)
        # 自注意力机制
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads)
        # 第二个LayerNorm
        self.ln2 = nn.LayerNorm(embed_dim)
        # MLP模块
        self.mlp = MLP(embed_dim, mlp_hidden_dim)
        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, print_info=False):
        if print_info:
            print("\n当前阶段: Transformer编码器块处理")

        # 1. 第一个子层: LayerNorm + 自注意力 + 残差连接
        x_ln1 = self.ln1(x)
        attn_out = self.self_attn(x_ln1, print_info)
        attn_out = self.dropout(attn_out)
        x = x + attn_out  # 残差连接

        # 2. 第二个子层: LayerNorm + MLP + 残差连接
        x_ln2 = self.ln2(x)
        mlp_out = self.mlp(x_ln2, print_info)
        mlp_out = self.dropout(mlp_out)
        x = x + mlp_out  # 残差连接

        return x


# 定义Vision Transformer模型
class VisionTransformer(nn.Module):
    def __init__(
            self,
            img_size=32,  # 图像尺寸
            patch_size=4,  # 图像块大小
            in_channels=3,  # 输入通道数
            num_classes=10,  # 分类类别数
            embed_dim=256,  # 嵌入维度
            depth=6,  # Transformer块的数量
            num_heads=8,  # 注意力头数
            mlp_ratio=4,  # MLP隐藏层维度和嵌入维度的比率
            dropout=0.1,  # Dropout率
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

        # 类别标记 [CLS]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.dropout = nn.Dropout(dropout)

        # Transformer编码器块
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim, num_heads, embed_dim * mlp_ratio, dropout
            )
            for _ in range(depth)
        ])

        # 最终层归一化
        self.ln = nn.LayerNorm(embed_dim)

        # 分类头
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化位置编码和类别标记
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    # 初始化权重
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, print_info=False):
        if print_info:
            print("\n当前阶段: 模型前向传播")

        batch_size = x.shape[0]

        # 1. 图像分块并嵌入
        x = self.patch_embed(x)  # [B, C, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]

        # 2. 添加类别标记 [CLS]
        cls_token = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, C]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, C]

        # 3. 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)

        # 4. 通过Transformer编码器
        for i, block in enumerate(self.blocks):
            if print_info and (i == 0 or i == len(self.blocks) - 1):
                x = block(x, print_info and i == 0)  # 只在第一个块打印详情
            else:
                x = block(x, False)

        # 5. 最终层归一化
        x = self.ln(x)

        # 6. 使用类别标记进行分类
        x = x[:, 0]  # 取[CLS]标记对应的输出

        # 7. 分类头
        x = self.head(x)

        return x


# 数据加载和预处理函数
def load_cifar10(batch_size=128, data_dir='cifar10_data'):
    # 定义数据转换
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # 标准化
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    # 加载训练集和测试集
    print("当前阶段: 数据下载与准备")
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

    print("当前阶段: 数据加载完成")
    return train_loader, test_loader


# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 只在第一个epoch的第一个批次打印张量信息
    tensor_info_printed = False

    pbar = tqdm(train_loader, desc=f"当前阶段: 第 {epoch + 1} 轮训练")
    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # 只有在第一个epoch的第一个批次打印信息
        print_info = (epoch == 0 and i == 0 and not tensor_info_printed)
        if print_info:
            tensor_info_printed = True
            print("\n当前阶段: 首批数据处理")

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images, print_info)
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
            '损失': running_loss / (pbar.n + 1),
            '准确率': 100. * correct / total
        })

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
        for images, labels in tqdm(test_loader, desc="当前阶段: 模型评估"):
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

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, test_losses, 'r-', label='测试损失')
    plt.title('损失随轮次变化')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='训练准确率')
    plt.plot(epochs, test_accs, 'r-', label='测试准确率')
    plt.title('准确率随轮次变化')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('vit_training_metrics.png')
    plt.show()


# 主函数
def main():
    # 定义超参数
    batch_size = 128  # 批次大小
    num_epochs = 10  # 训练轮次
    learning_rate = 3e-4  # 学习率
    weight_decay = 1e-4  # 权重衰减

    # 加载数据
    train_loader, test_loader = load_cifar10(batch_size)

    # 实例化模型
    model = VisionTransformer(
        img_size=32,  # 图像尺寸
        patch_size=4,  # 图像块大小，将图像分成8x8=64个块
        in_channels=3,  # 输入通道数（彩色图像为3）
        num_classes=10,  # 分类类别数（CIFAR-10有10个类别）
        embed_dim=192,  # 嵌入维度
        depth=6,  # Transformer块的数量
        num_heads=8,  # 注意力头数
        mlp_ratio=4,  # MLP隐藏层维度和嵌入维度的比率
        dropout=0.1  # Dropout率
    ).to(device)

    print("当前阶段: 模型初始化完成")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 使用余弦退火学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 记录训练过程的指标
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []

    print("当前阶段: 开始训练")
    for epoch in range(num_epochs):
        # 训练一个轮次
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        # 在测试集上评估
        test_loss, test_acc = evaluate(model, test_loader, criterion)

        # 记录指标
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # 打印当前轮次的结果
        print(f"当前阶段: 完成第 {epoch + 1}/{num_epochs} 轮训练")
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%")
        print(f"测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%")
        print("-" * 50)

        # 更新学习率
        scheduler.step()

    print("当前阶段: 训练完成")

    # 保存模型
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print("当前阶段: 模型保存完成")

    # 绘制训练过程图表
    print("当前阶段: 生成训练过程图表")
    plot_metrics(train_losses, train_accs, test_losses, test_accs)


if __name__ == "__main__":
    main()