import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 设置设备：如果有GPU，则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义编码器 (Encoder)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # 全连接层：输入784维，输出256维，激活函数为ReLU
        self.fc1 = nn.Linear(784, 256)
        # 全连接层：输入256维，输出512维，激活函数为Tanh
        self.fc2 = nn.Linear(256, 512)
        # 全连接层：输入512维，输出2维，生成潜变量均值
        self.fc3_mean = nn.Linear(512, 2)
        # 全连接层：输入512维，输出2维，生成潜变量方差（对数形式）
        self.fc3_logvar = nn.Linear(512, 2)
        # 激活函数
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    # 前向传播
    def forward(self, x):
        h1 = self.relu(self.fc1(x))     # 输入经过全连接层和ReLU激活函数
        h2 = self.tanh(self.fc2(h1))    # 经过第二个全连接层和Tanh激活函数
        mean = self.fc3_mean(h2)        # 生成潜变量均值
        logvar = self.fc3_logvar(h2)    # 生成潜变量方差
        return mean, logvar             # 返回均值和方差


# 编码器的作用是将输入的高维数据（MNIST图片）压缩为低维潜变量（均值和对数方差），
# 这些潜变量是生成器用来重建图像的关键。

# 定义解码器 (Decoder)
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # 全连接层：从2维潜变量到512维
        self.fc1 = nn.Linear(2, 512)
        # 全连接层：从512维到784维（28x28的图片展平成784维）
        self.fc2 = nn.Linear(512, 784)
        # 激活函数：ReLU
        self.relu = nn.ReLU()
        # 激活函数：Sigmoid，用于生成0到1之间的像素值
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, z):
        h2 = self.relu(self.fc1(z))  # 将潜变量通过全连接层并应用ReLU激活
        return self.sigmoid(self.fc2(h2))  # 将输出映射到(0,1)之间的像素值

# 解码器的作用是将低维潜变量转换回高维空间（即原始784维图像），
# 并输出经过Sigmoid函数的结果，用于生成最终的图像。

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 初始化编码器和解码器
        self.encoder = Encoder()
        self.decoder = Decoder()

    # 重参数化技巧：从均值和方差生成潜变量z
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)  # 标准差 = exp(0.5 * logvar)
        eps = torch.randn_like(std)    # 随机噪声，符合标准正态分布
        return mean + eps * std        # 潜变量z = 均值 + 噪声 * 标准差

    # 前向传播
    def forward(self, x):
        mean, logvar = self.encoder(x)  # 编码器生成均值和对数方差
        z = self.reparameterize(mean, logvar)  # 通过重参数化技巧得到潜变量z
        recon_x = self.decoder(z)  # 解码器生成重构图像
        return recon_x, mean, logvar  # 返回重构图像、均值和对数方差

# VAE模型通过组合编码器和解码器，将输入的高维数据编码为潜变量z，并解码为原始数据。
# 这里使用了重参数化技巧（reparameterization trick）来允许梯度传播。


# 损失函数：包括重构损失和KL散度
def loss_function(recon_x, x, mean, logvar):
    # 二元交叉熵损失，用于衡量重构图像与原始图像之间的差异
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL散度，用于衡量潜变量z的分布与标准正态分布之间的差异
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD  # 总损失为重构损失和KL散度的和

# 损失函数由两部分组成：重构损失和KL散度，前者衡量生成图像和输入图像的相似度，
# 后者确保潜变量z符合标准正态分布，从而保持生成的多样性。


# 加载MNIST数据集，并将其转换为Tensor格式
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# 使用DataLoader将数据集分批次加载，每个批次64张图片
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 数据加载器负责加载并预处理MNIST手写数字数据集。我们将数据集按批次进行加载，
# 每个批次含64张图片，并随机打乱顺序，便于后续训练。


# 初始化VAE模型并将其移动到设备（CPU或GPU）
vae = VAE().to(device)
# 使用Adam优化器来优化模型参数，学习率设为1e-5
optimizer = optim.Adam(vae.parameters(), lr=1e-5)

# 在训练模型之前，需要初始化模型并设置优化器。Adam优化器是一种常用的优化算法，
# 它能够自适应调整学习率以加速模型的收敛。


# 训练模型
num_epochs = 50  # 训练50轮
losses = []  # 用于保存每轮的总损失
kl_losses = []  # 用于保存每轮的KL散度

for epoch in range(num_epochs):
    total_loss = 0  # 累计总损失
    total_kl = 0  # 累计KL散度
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)  # 将图片展平为784维向量并移动到设备

        optimizer.zero_grad()  # 清空梯度
        recon_batch, mean, logvar = vae(data)  # 前向传播得到重构图像、均值和对数方差

        loss = loss_function(recon_batch, data, mean, logvar)  # 计算损失
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新模型参数

        total_loss += loss.item()  # 累加总损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()).item()  # 计算KL散度
        total_kl += kl_loss  # 累加KL散度

    # 保存每轮的平均损失和KL散度
    losses.append(total_loss / len(train_loader.dataset))
    kl_losses.append(total_kl / len(train_loader.dataset))
    print(f'Epoch {epoch + 1}, Loss: {total_loss / len(train_loader.dataset):.4f}, KL Loss: {total_kl / len(train_loader.dataset):.4f}')

# 训练过程分为50轮。每一轮中，我们将数据按批次输入模型，计算损失并更新模型参数。
# 每轮结束后，记录总损失和KL散度，用于后续绘图和分析。


# 绘制重构损失和KL散度随迭代次数的变化图
plt.figure(figsize=(10, 5))
plt.plot(losses, label='Total Loss')  # 绘制总损失曲线
plt.plot(kl_losses, label='KL Divergence')  # 绘制KL散度曲线
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 训练完成后，绘制每轮的总损失和KL散度曲线，便于观察模型的训练情况。


# 生成10张手写数字图片
with torch.no_grad():  # 生成时不需要计算梯度
    z = torch.randn(10, 2).to(device)  # 从标准正态分布中采样10个2维潜变量
    sample = vae.decoder(z).cpu().view(10, 28, 28)  # 使用解码器生成10张图片并重塑为28x28

    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)  # 创建子图，2行5列
        plt.imshow(sample[i], cmap='gray')  # 显示生成的手写数字图片
        plt.axis('off')  # 隐藏坐标轴
    plt.show()

# 在训练完VAE后，我们可以通过从标准正态分布中随机采样潜变量z，
# 并使用解码器生成新的手写数字图片，展示模型的生成能力。
