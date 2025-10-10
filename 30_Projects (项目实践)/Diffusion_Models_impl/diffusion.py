import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math

# 超参数升级
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
epochs = 10
timesteps = 1000
learning_rate = 1e-4

# 改进的噪声调度（余弦调度）
def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

betas = cosine_beta_schedule(timesteps).to(device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# 改进的UNet结构（包含残差连接和注意力机制）
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = h + self.time_proj(t)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)

# 修改后的UNet类
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # 下采样
        self.down1 = ResidualBlock(1, 64, 256)
        self.down2 = ResidualBlock(64, 128, 256)
        self.down3 = ResidualBlock(128, 256, 256)
        
        # 中间层（需要单独处理时间步参数）
        self.mid_block1 = ResidualBlock(256, 256, 256)
        self.mid_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.mid_bn = nn.BatchNorm2d(256)
        self.mid_block2 = ResidualBlock(256, 256, 256)
        
        # 上采样 - 修复输入通道数
        self.up3 = ResidualBlock(256 + 128, 128, 256)  # 256(中间层) + 128(down2输出)
        self.up2 = ResidualBlock(128 + 64, 64, 256)    # 128(up3输出) + 64(down1输出)
        self.up1 = ResidualBlock(64, 64, 256)
        
        self.final = nn.Conv2d(64, 1, 1)
        
    def forward(self, x, t):
        # 时间编码
        t_embed = self.time_embed(timestep_embedding(t, 128))
        
        # 下采样
        x1 = self.down1(x, t_embed)
        x2 = F.avg_pool2d(x1, 2)
        x2 = self.down2(x2, t_embed)
        x3 = F.avg_pool2d(x2, 2)
        x3 = self.down3(x3, t_embed)
        
        # 中间处理（显式传递时间参数）
        m = self.mid_block1(x3, t_embed)
        m = self.mid_conv(m)
        m = self.mid_bn(m)
        m = F.silu(m)
        m = self.mid_block2(m, t_embed)
        
        # 上采样
        m = F.interpolate(m, scale_factor=2)
        m = torch.cat([m, x2], dim=1)
        m = self.up3(m, t_embed)
        m = F.interpolate(m, scale_factor=2)
        m = torch.cat([m, x1], dim=1)
        m = self.up2(m, t_embed)
        m = self.up1(m, t_embed)
        
        return self.final(m)

def timestep_embedding(t, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
dataset = datasets.MNIST("./mnist_data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(device)
model = UNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(epochs):
    for step, (images, _) in enumerate(loader):
        images = images.to(device)
        batch_size = images.size(0)
        
        # 随机时间步
        t = torch.randint(0, timesteps, (batch_size,), device=device)
        
        # 前向加噪
        noise = torch.randn_like(images)
        sqrt_alpha_cumprod = sqrt_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        sqrt_one_minus = sqrt_one_minus_alphas_cumprod.gather(-1, t).view(-1, 1, 1, 1)
        noisy_images = sqrt_alpha_cumprod * images + sqrt_one_minus * noise
        
        # 预测噪声
        pred_noise = model(noisy_images, t)
        
        # 损失计算
        loss = F.mse_loss(pred_noise, noise)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
    print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

# 改进的采样过程
@torch.no_grad()
def sample(num_samples=16):
    model.eval()
    samples = torch.randn(num_samples, 1, 28, 28, device=device)
    
    for i in reversed(range(timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        pred_noise = model(samples, t)
        
        alpha_t = alphas[i]
        alpha_cumprod_t = alphas_cumprod[i]
        beta_t = betas[i]
        
        if i > 0:
            noise = torch.randn_like(samples)
        else:
            noise = torch.zeros_like(samples)
            
        samples = (samples - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * pred_noise) / torch.sqrt(alpha_t)
        samples = samples + torch.sqrt(beta_t) * noise
    
    samples = samples.clamp(-1, 1).cpu()
    samples = (samples + 1) * 0.5  # 转换到[0,1]范围
    return samples

# 生成并显示图像
generated = sample(16)

plt.figure(figsize=(10, 10))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(generated[i].squeeze(), cmap="gray")
    plt.axis("off")
plt.tight_layout()
plt.show()