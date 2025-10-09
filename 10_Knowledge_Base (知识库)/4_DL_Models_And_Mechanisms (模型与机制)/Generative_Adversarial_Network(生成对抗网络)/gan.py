# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器模块
import torchvision  # 计算机视觉相关工具
from torchvision import transforms  # 图像变换
from torch.utils.data import DataLoader  # 数据加载器
import matplotlib.pyplot as plt  # 绘图库
import numpy as np  # 数值计算库
from tqdm import tqdm  # 进度条工具

"""
GAN (生成对抗网络) 基本原理：
1. 由两个网络组成：生成器(Generator)和判别器(Discriminator)
2. 生成器：接收随机噪声，生成假样本
3. 判别器：判断输入样本是真实数据还是生成器生成的假样本
4. 两者通过对抗训练共同提升：生成器试图生成更真实的样本欺骗判别器，
   判别器则不断提升判别能力
5. 最终目标是生成器能生成足以乱真的样本
"""

# 设置随机种子保证可重复性
torch.manual_seed(42)

# 超参数设置
batch_size = 256  # 增大batch_size可以提升训练稳定性
epochs = 100  # 增加训练轮数以获得更好的生成效果
latent_dim = 128  # 增加潜在空间维度，提供更强的生成能力
lr = 0.0002  # 学习率，较小的学习率有助于稳定训练
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动选择GPU或CPU

# 改进的数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为Tensor
    transforms.Normalize((0.5,), (0.5,)),  # 将像素值归一化到[-1,1]范围
    transforms.RandomRotation(5)  # 添加数据增强，随机旋转±5度
])

# 下载并加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root='./mnist_data',  # 数据集保存路径
    train=True,  # 使用训练集
    download=True,  # 如果不存在则下载
    transform=transform  # 应用上面定义的数据变换
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 每个epoch打乱数据
    num_workers=0  # 禁用多进程以避免在某些环境中的问题
)

"""
生成器(Generator)结构说明：
1. 使用转置卷积(ConvTranspose2d)进行上采样
2. 输入是潜在空间中的随机噪声(latent_dim维)
3. 通过多层上采样逐渐增大特征图尺寸
4. 使用BatchNorm2d加速收敛并稳定训练
5. 最终使用Tanh激活函数将输出限制在[-1,1]范围
6. 最后裁剪到28x28以匹配MNIST图像尺寸
"""


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 输入: latent_dim维向量，重塑为(latent_dim,1,1)的张量
            # 转置卷积：输入通道latent_dim，输出通道512，kernel_size=4，stride=1，padding=0
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),  # 输出尺寸: 4x4
            nn.BatchNorm2d(512),  # 批归一化
            nn.ReLU(True),  # ReLU激活函数

            # 上采样层1：4x4 -> 8x8
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # 输出尺寸: 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 上采样层2：8x8 -> 16x16
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 输出尺寸: 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 上采样层3：16x16 -> 32x32
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),  # 输出尺寸: 32x32
            nn.Tanh(),  # 使用Tanh将输出限制在[-1,1]范围
            transforms.CenterCrop(28)  # 中心裁剪到28x28，匹配MNIST尺寸
        )

    def forward(self, input):
        # 将输入噪声重塑为(batch_size, latent_dim, 1, 1)的张量
        input = input.view(-1, latent_dim, 1, 1)
        return self.main(input)


"""
判别器(Discriminator)结构说明：
1. 使用普通卷积层进行下采样
2. 输入是28x28的灰度图像
3. 通过多层卷积逐渐减小特征图尺寸
4. 使用LeakyReLU激活函数防止梯度消失
5. 最终输出一个0-1之间的概率值(使用Sigmoid)
"""


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入: 1x28x28图像
            # 卷积层1：28x28 -> 14x14
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),  # 输出尺寸: 14x14
            nn.LeakyReLU(0.2, inplace=True),  # LeakyReLU，负斜率0.2

            # 卷积层2：14x14 -> 7x7
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 输出尺寸: 7x7
            nn.BatchNorm2d(256),  # 批归一化
            nn.LeakyReLU(0.2, inplace=True),

            # 卷积层3：7x7 -> 3x3
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 输出尺寸: 3x3
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 最后一层卷积：3x3 -> 1x1
            nn.Conv2d(512, 1, 3, 1, 0, bias=False),  # 输出尺寸: 1x1
            nn.Sigmoid()  # 使用Sigmoid输出概率值
        )

    def forward(self, input):
        # 将输出展平为(batch_size,)的形状
        return self.main(input).view(-1)


# 初始化模型并移动到设备(GPU/CPU)
generator = Generator().to(device)
discriminator = Discriminator().to(device)

"""
损失函数说明：
使用二元交叉熵损失(BCELoss)：
- 判别器试图最大化对真实样本和生成样本的正确分类概率
- 生成器试图最小化判别器对生成样本的判别准确率
"""
adversarial_loss = nn.BCELoss()

# 优化器设置
# 使用Adam优化器，betas参数控制动量项
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

"""
训练过程说明：
1. 交替训练判别器和生成器
2. 使用标签平滑(real_labels=0.9, fake_labels=0.1)提高稳定性
3. 每5个epoch可视化一次生成结果
"""
for epoch in range(epochs):
    # 使用tqdm进度条
    progress_bar = tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc=f"Epoch {epoch + 1}/{epochs}",
                        ncols=100)

    for i, (imgs, _) in progress_bar:
        real_imgs = imgs.to(device)
        batch_size = real_imgs.size(0)

        # 使用标签平滑(Label Smoothing)提高稳定性
        real_labels = torch.full((batch_size,), 0.9, device=device)  # 真实样本标签设为0.9
        fake_labels = torch.full((batch_size,), 0.1, device=device)  # 生成样本标签设为0.1

        # ==================== 训练判别器 ====================
        for _ in range(1):  # 可以调整为多次迭代以增强判别器
            optimizer_D.zero_grad()  # 清空梯度

            # 计算真实图片的损失
            real_output = discriminator(real_imgs)  # 判别器对真实图片的输出
            d_loss_real = adversarial_loss(real_output, real_labels)  # 真实图片的损失

            # 计算生成图片的损失
            z = torch.randn(batch_size, latent_dim, 1, 1, device=device)  # 生成随机噪声
            fake_imgs = generator(z)  # 生成假图片
            fake_output = discriminator(fake_imgs.detach())  # 判别器对假图片的输出(使用detach避免影响生成器)
            d_loss_fake = adversarial_loss(fake_output, fake_labels)  # 假图片的损失

            # 总判别器损失(真实和假图片损失的平均)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()  # 反向传播
            optimizer_D.step()  # 更新判别器参数

        # ==================== 训练生成器 ====================
        optimizer_G.zero_grad()  # 清空梯度

        # 生成器试图让判别器将假图片判断为真
        gen_output = discriminator(fake_imgs)  # 重新计算判别器输出(不使用detach)
        g_loss = adversarial_loss(gen_output, real_labels)  # 生成器损失

        g_loss.backward()  # 反向传播
        optimizer_G.step()  # 更新生成器参数

        # 更新进度条显示
        progress_bar.set_postfix({
            "D_loss": f"{d_loss.item():.4f}",  # 判别器损失
            "G_loss": f"{g_loss.item():.4f}"  # 生成器损失
        })

    # ==================== 可视化生成结果 ====================
    if (epoch + 1) % 5 == 0:  # 每5个epoch显示一次
        with torch.no_grad():  # 禁用梯度计算
            # 生成16个随机噪声样本
            test_z = torch.randn(16, latent_dim, 1, 1, device=device)
            generated = generator(test_z).cpu().numpy()  # 生成图片并转为numpy数组

        # 绘制生成结果
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated[i][0], cmap='gray')  # 显示灰度图
            plt.axis('off')
        plt.tight_layout()
        plt.show()