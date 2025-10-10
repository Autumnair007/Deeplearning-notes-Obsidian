import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import math
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
import os


# ============================================================================
# 配置参数
# ============================================================================

class Config:
    # 模型参数
    MODEL_SIZE = 'small'  # 为了在CIFAR-10上快速训练
    PATCH_SIZE = 4  # CIFAR-10图像较小，使用较小的patch
    IMAGE_SIZE = 32  # CIFAR-10原始尺寸

    # 模型维度
    MODEL_DIMS = {
        'small': {'dim': 384, 'depth': 6, 'heads': 6},  # 简化模型
        'base': {'dim': 768, 'depth': 12, 'heads': 12},
        'large': {'dim': 1024, 'depth': 24, 'heads': 16}
    }

    # 投影头参数
    PROJECTION_DIM = 256  # 简化投影维度

    # 训练参数
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.04
    TOTAL_EPOCHS = 2

    # Multi-crop参数 - 修改为使用相同尺寸避免位置编码问题
    GLOBAL_CROPS_NUMBER = 2
    LOCAL_CROPS_NUMBER = 6
    GLOBAL_CROP_SIZE = 32  # 保持与原图相同尺寸
    LOCAL_CROP_SIZE = 32  # 暂时使用相同尺寸，避免位置编码问题

    # 损失权重
    LAMBDA_IMG = 1.0
    LAMBDA_PATCH = 1.0
    LAMBDA_KOLEO = 0.1

    # EMA参数
    EMA_START = 0.996
    EMA_END = 1.0

    # 掩码参数
    MASK_RATIO = 0.15

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据路径
    DATA_PATH = './cifar10_data'


# ============================================================================
# Vision Transformer 组件
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = rearrange(x, 'b e h w -> b (h w) e')
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.dropout(x)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.,
                 qkv_bias=True, dropout=0.):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, dropout=dropout
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # 初始化权重
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, x, w, h):
        """插值位置编码以处理不同尺寸的输入"""
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size

        # 添加一个小数以避免浮点数错误
        w0, h0 = w0 + 0.1, h0 + 0.1

        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sx, sy),
            mode='bicubic',
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # 使用插值位置编码
        x = x + self.interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # 返回CLS token和patch tokens
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]

        return cls_token, patch_tokens


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.projection(x)


# ============================================================================
# 数据增强和数据集
# ============================================================================

class MultiCropTransform:
    def __init__(self, global_size=32, local_size=32, global_crops=2, local_crops=6):
        self.global_size = global_size
        self.local_size = local_size
        self.global_crops = global_crops
        self.local_crops = local_crops

        # 全局裁剪变换 - 使用更强的数据增强
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # 局部裁剪变换 - 更小的裁剪比例模拟局部视图
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.2, 0.6)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, image):
        crops = []
        # 生成全局裁剪
        for _ in range(self.global_crops):
            crops.append(self.global_transform(image))
        # 生成局部裁剪
        for _ in range(self.local_crops):
            crops.append(self.local_transform(image))
        return crops


class MultiCropDataset(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # 忽略标签，这是自监督学习
        crops = self.transform(image)
        return crops


# ============================================================================
# 损失函数
# ============================================================================

def sinkhorn_knopp_centering(features, num_iters=3, epsilon=0.05):
    """Sinkhorn-Knopp centering algorithm"""
    with torch.no_grad():
        Q = torch.exp(features / epsilon).t()  # (output_dim, batch_size)

        B = Q.shape[1]  # batch size
        K = Q.shape[0]  # output dim

        # 初始化
        Q /= torch.sum(Q)

        for _ in range(num_iters):
            # 行归一化
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows + 1e-6
            Q /= K

            # 列归一化
            sum_of_cols = torch.sum(Q, dim=0, keepdim=True)
            Q /= sum_of_cols + 1e-6
            Q /= B

        Q *= B
        return Q.t()


def dino_loss(student_output, teacher_output, teacher_temp=0.04, student_temp=0.1):
    """DINO loss calculation"""
    # 温度缩放
    student_out = student_output / student_temp
    teacher_out = F.softmax(teacher_output / teacher_temp, dim=-1)

    # 计算交叉熵
    loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
    return loss.mean()


def koleo_regularizer(features):
    """KoLeo regularization loss"""
    # L2归一化
    features = F.normalize(features, p=2, dim=-1)

    # 计算余弦相似度矩阵
    similarity_matrix = torch.matmul(features, features.t())

    # 移除对角线元素
    mask = torch.eye(similarity_matrix.shape[0], device=similarity_matrix.device).bool()
    similarity_matrix = similarity_matrix.masked_fill(mask, 0)

    # 计算KoLeo损失（最小化特征间的相似性）
    loss = torch.sum(similarity_matrix ** 2) / (features.shape[0] * (features.shape[0] - 1))
    return loss


def generate_random_mask(batch_size, num_patches, mask_ratio):
    """生成随机掩码"""
    num_masked = int(mask_ratio * num_patches)
    mask = torch.zeros(batch_size, num_patches, dtype=torch.bool)

    for i in range(batch_size):
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[i, masked_indices] = True

    return mask


# ============================================================================
# 调度器
# ============================================================================

class CosineScheduler:
    def __init__(self, start_val, end_val, total_steps):
        self.start_val = start_val
        self.end_val = end_val
        self.total_steps = total_steps

    def get_value(self, step):
        if step >= self.total_steps:
            return self.end_val

        progress = step / self.total_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.end_val + (self.start_val - self.end_val) * cosine_decay


# ============================================================================
# 训练函数
# ============================================================================

def train_dinov2():
    config = Config()
    print(f"使用设备: {config.DEVICE}")

    # 创建数据集
    transform = MultiCropTransform(
        global_size=config.GLOBAL_CROP_SIZE,
        local_size=config.LOCAL_CROP_SIZE,
        global_crops=config.GLOBAL_CROPS_NUMBER,
        local_crops=config.LOCAL_CROPS_NUMBER
    )

    # 加载CIFAR-10数据集
    cifar10_dataset = CIFAR10(
        root=config.DATA_PATH,
        train=True,
        download=True,
        transform=None  # 我们使用自定义的transform
    )

    dataset = MultiCropDataset(cifar10_dataset, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    # 创建模型
    model_config = config.MODEL_DIMS[config.MODEL_SIZE]

    # 学生网络
    student_network = VisionTransformer(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        embed_dim=model_config['dim'],
        depth=model_config['depth'],
        num_heads=model_config['heads']
    ).to(config.DEVICE)

    # 教师网络
    teacher_network = VisionTransformer(
        img_size=config.IMAGE_SIZE,
        patch_size=config.PATCH_SIZE,
        embed_dim=model_config['dim'],
        depth=model_config['depth'],
        num_heads=model_config['heads']
    ).to(config.DEVICE)

    # 初始化教师网络权重
    teacher_network.load_state_dict(student_network.state_dict())
    for param in teacher_network.parameters():
        param.requires_grad = False

    # 投影头
    img_level_head = ProjectionHead(
        input_dim=model_config['dim'],
        output_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)

    patch_level_head = ProjectionHead(
        input_dim=model_config['dim'],
        output_dim=config.PROJECTION_DIM
    ).to(config.DEVICE)

    # 优化器
    params = list(student_network.parameters()) + \
             list(img_level_head.parameters()) + \
             list(patch_level_head.parameters())

    optimizer = torch.optim.AdamW(
        params,
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.TOTAL_EPOCHS
    )

    # EMA调度器
    total_steps = len(dataloader) * config.TOTAL_EPOCHS
    ema_scheduler = CosineScheduler(
        start_val=config.EMA_START,
        end_val=config.EMA_END,
        total_steps=total_steps
    )

    # 训练循环
    global_step = 0
    num_patches = (config.IMAGE_SIZE // config.PATCH_SIZE) ** 2

    print(f"每个图像的patches数量: {num_patches}")
    print(f"总训练步数: {total_steps}")
    print("开始训练...")

    for epoch in range(config.TOTAL_EPOCHS):
        total_loss_epoch = 0
        img_loss_epoch = 0
        patch_loss_epoch = 0
        koleo_loss_epoch = 0

        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch + 1}/{config.TOTAL_EPOCHS}')

        for batch_idx, crops_batch in enumerate(progress_bar):
            # crops_batch是一个list，包含每个样本的多个裁剪
            batch_size = len(crops_batch[0])
            total_crops = config.GLOBAL_CROPS_NUMBER + config.LOCAL_CROPS_NUMBER

            # 重新组织数据：从 [crop_idx][batch_idx] 到 [view_idx, batch_size, ...]
            all_views = []
            for crop_idx in range(total_crops):
                view_batch = torch.stack([crops_batch[crop_idx][i] for i in range(batch_size)])
                all_views.append(view_batch.to(config.DEVICE))

            # 分离全局视图和局部视图
            global_views = all_views[:config.GLOBAL_CROPS_NUMBER]
            local_views = all_views[config.GLOBAL_CROPS_NUMBER:]
            all_views = global_views + local_views

            # 教师网络前向传播（只处理全局视图）
            with torch.no_grad():
                teacher_cls_outputs = []
                teacher_patch_outputs = []

                for global_view in global_views:
                    cls_token, patch_tokens = teacher_network(global_view)
                    teacher_cls_outputs.append(cls_token)
                    teacher_patch_outputs.append(patch_tokens)

                # 应用投影头
                teacher_cls_proj = []
                teacher_patch_proj = []
                for i in range(len(teacher_cls_outputs)):
                    teacher_cls_proj.append(img_level_head(teacher_cls_outputs[i]))
                    teacher_patch_proj.append(patch_level_head(teacher_patch_outputs[i]))

                # Sinkhorn-Knopp centering
                for i in range(len(teacher_cls_proj)):
                    teacher_cls_proj[i] = sinkhorn_knopp_centering(teacher_cls_proj[i])

            # 生成掩码
            mask = generate_random_mask(batch_size, num_patches, config.MASK_RATIO).to(config.DEVICE)

            # 学生网络前向传播（处理所有视图）
            student_cls_outputs = []
            student_patch_outputs = []

            for view in all_views:
                cls_token, patch_tokens = student_network(view)
                student_cls_outputs.append(cls_token)
                student_patch_outputs.append(patch_tokens)

            # 应用投影头
            student_cls_proj = []
            student_patch_proj = []
            for i in range(len(student_cls_outputs)):
                student_cls_proj.append(img_level_head(student_cls_outputs[i]))
                student_patch_proj.append(patch_level_head(student_patch_outputs[i]))

            # 计算损失
            total_loss = 0

            # 图像级损失（DINO loss）
            img_loss = 0
            num_comparisons = 0
            for teacher_idx in range(len(teacher_cls_proj)):
                for student_idx in range(len(student_cls_proj)):
                    if student_idx != teacher_idx or teacher_idx >= config.GLOBAL_CROPS_NUMBER:
                        # 避免完全相同的全局视图比较，但允许全局-局部比较
                        loss = dino_loss(
                            student_cls_proj[student_idx],
                            teacher_cls_proj[teacher_idx]
                        )
                        img_loss += loss
                        num_comparisons += 1

            if num_comparisons > 0:
                img_loss /= num_comparisons

            total_loss += config.LAMBDA_IMG * img_loss

            # 补丁级损失（简化版本，只使用第一个全局视图）
            if len(teacher_patch_proj) > 0 and len(student_patch_proj) > 0:
                teacher_patches = teacher_patch_proj[0]  # 第一个全局视图
                student_patches = student_patch_proj[0]  # 对应的学生视图

                # 在掩码位置计算损失
                masked_teacher = teacher_patches[mask]
                masked_student = student_patches[mask]

                if masked_teacher.numel() > 0:
                    patch_loss = F.mse_loss(masked_student, masked_teacher)
                    total_loss += config.LAMBDA_PATCH * patch_loss
                else:
                    patch_loss = torch.tensor(0.0)
            else:
                patch_loss = torch.tensor(0.0)

            # KoLeo正则化损失
            all_student_cls = torch.cat(student_cls_outputs, dim=0)
            koleo_loss = koleo_regularizer(all_student_cls)
            total_loss += config.LAMBDA_KOLEO * koleo_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            # 优化器步骤
            optimizer.step()

            # 更新教师网络（EMA）
            with torch.no_grad():
                m = ema_scheduler.get_value(global_step)
                for student_param, teacher_param in zip(student_network.parameters(), teacher_network.parameters()):
                    teacher_param.data.mul_(m).add_((1 - m) * student_param.data)

            # 记录损失
            total_loss_epoch += total_loss.item()
            img_loss_epoch += img_loss.item() if isinstance(img_loss, torch.Tensor) else img_loss
            patch_loss_epoch += patch_loss.item() if isinstance(patch_loss, torch.Tensor) else patch_loss
            koleo_loss_epoch += koleo_loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                'Total': f'{total_loss.item():.3f}',
                'Img': f'{img_loss.item() if isinstance(img_loss, torch.Tensor) else img_loss:.3f}',
                'Patch': f'{patch_loss.item() if isinstance(patch_loss, torch.Tensor) else patch_loss:.3f}',
                'KoLeo': f'{koleo_loss.item():.3f}',
                'EMA': f'{m:.3f}'
            })

            global_step += 1

        # 更新学习率
        lr_scheduler.step()

        # 打印epoch统计
        num_batches = len(dataloader)
        print(f"\nEpoch {epoch + 1} 完成:")
        print(f"  平均总损失: {total_loss_epoch / num_batches:.4f}")
        print(f"  平均图像损失: {img_loss_epoch / num_batches:.4f}")
        print(f"  平均补丁损失: {patch_loss_epoch / num_batches:.4f}")
        print(f"  平均KoLeo损失: {koleo_loss_epoch / num_batches:.4f}")
        print(f"  当前学习率: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存模型（每10个epoch）
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'student_state_dict': student_network.state_dict(),
                'teacher_state_dict': teacher_network.state_dict(),
                'img_head_state_dict': img_level_head.state_dict(),
                'patch_head_state_dict': patch_level_head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }

            os.makedirs('checkpoints', exist_ok=True)
            torch.save(checkpoint, f'checkpoints/dinov2_epoch_{epoch + 1}.pth')
            print(f"  模型已保存到 checkpoints/dinov2_epoch_{epoch + 1}.pth")

    print("训练完成！")

    # 保存最终模型
    final_checkpoint = {
        'student_state_dict': student_network.state_dict(),
        'teacher_state_dict': teacher_network.state_dict(),
        'img_head_state_dict': img_level_head.state_dict(),
        'patch_head_state_dict': patch_level_head.state_dict(),
        'config': config
    }

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(final_checkpoint, 'checkpoints/dinov2_final.pth')
    print("最终模型已保存到 checkpoints/dinov2_final.pth")


# ============================================================================
# 主函数
# ============================================================================

if __name__ == "__main__":
    print("开始DINOv2训练...")
    print("=" * 50)
    train_dinov2()