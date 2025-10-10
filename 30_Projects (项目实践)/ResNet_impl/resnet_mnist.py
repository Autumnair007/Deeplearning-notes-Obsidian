# 导入PyTorch库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入神经网络函数模块（包含ReLU等激活函数）
import torch.nn.functional as F
# 导入视觉数据集和相关转换
from torchvision import datasets, transforms
# 导入数据加载器
from torch.utils.data import DataLoader

# 定义残差块（ResNet的基本构建块）
class BasicBlock(nn.Module):
    # 扩展系数，用于调整输出通道数
    expansion = 1

    # 初始化函数
    def __init__(self, in_channels, out_channels, stride=1):
        # 调用父类初始化
        super().__init__()
        # 第一个卷积层：3x3卷积，可能有下采样(stride>1)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # 第一个批归一化层
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层：3x3卷积，保持尺寸
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # 第二个批归一化层
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接（shortcut），默认是恒等映射
        self.shortcut = nn.Sequential()
        # 如果需要调整维度（下采样或通道数变化）
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                # 1x1卷积调整维度
                nn.Conv2d(in_channels, self.expansion * out_channels,
                          kernel_size=1, stride=stride, bias=False),
                # 批归一化
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    # 前向传播函数
    def forward(self, x):
        # 第一层：卷积 -> 批归一化 -> ReLU激活
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二层：卷积 -> 批归一化
        out = self.bn2(self.conv2(out))
        # 添加快捷连接
        out += self.shortcut(x)
        # ReLU激活
        out = F.relu(out)
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        # 初始输入通道数
        self.in_channels = 64

        # 第一层：3x3卷积，输入通道1（灰度图），输出通道64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # 批归一化
        self.bn1 = nn.BatchNorm2d(64)
        # 创建四个残差块层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 自适应平均池化，输出1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，输出类别数
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.verbose_once = True  # 添加一个标志，控制是否打印一次

    # 创建残差层
    def _make_layer(self, block, out_channels, num_blocks, stride):
        # 第一个块可能有下采样，其余步长为1
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        # 创建每个残差块
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            # 更新输入通道数
            self.in_channels = out_channels * block.expansion
        # 返回序列化的层
        return nn.Sequential(*layers)

    # 前向传播
    def forward(self, x):  # 移除 verbose 参数
        if self.verbose_once:
            print(f"输入: {x.shape}")  # 打印输入形状
        # 第一层：卷积 -> 批归一化 -> ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        if self.verbose_once:
            print(f"第一层卷积后: {out.shape}")  # 打印第一层输出形状
        # 通过四个残差层
        out = self.layer1(out)
        if self.verbose_once:
            print(f"第一层残差块后: {out.shape}")  # 打印第一层残差块输出形状
        out = self.layer2(out)
        if self.verbose_once:
            print(f"第二层残差块后: {out.shape}")  # 打印第二层残差块输出形状
        out = self.layer3(out)
        if self.verbose_once:
            print(f"第三层残差块后: {out.shape}")  # 打印第三层残差块输出形状
        out = self.layer4(out)
        if self.verbose_once:
            print(f"第四层残差块后: {out.shape}")  # 打印第四层残差块输出形状
        # 全局平均池化
        out = self.avgpool(out)
        if self.verbose_once:
            print(f"全局平均池化后: {out.shape}")  # 打印池化后形状
        # 展平
        out = out.view(out.size(0), -1)
        if self.verbose_once:
            print(f"展平后: {out.shape}")  # 打印展平后形状
        # 全连接层
        out = self.linear(out)
        if self.verbose_once:
            print(f"全连接层后: {out.shape}")  # 打印全连接层输出形状
            self.verbose_once = False  # 打印完一次后关闭标志
        return out

# 创建ResNet18模型
def ResNet18():
    # 使用BasicBlock，每层2个块，共4层
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 数据预处理
transform = transforms.Compose([
    # 转换为Tensor
    transforms.ToTensor(),
    # 标准化（MNIST的均值和标准差）
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载MNIST训练集
train_dataset = datasets.MNIST(
    root='./mnist_data',  # 数据存储路径
    train=True,  # 训练集
    download=True,  # 如果不存在则下载
    transform=transform)  # 应用预处理

# 加载MNIST测试集
test_dataset = datasets.MNIST(
    root='./mnist_data',
    train=False,  # 测试集
    download=True,
    transform=transform)

# 设置批量大小
batch_size = 64
# 创建训练数据加载器
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 打乱数据
    num_workers=0)  # 不使用多线程

# 创建测试数据加载器
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # 不打乱
    num_workers=0)

# 选择设备（GPU或CPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 创建模型并移到设备上
model = ResNet18().to(device)
# 使用Adam优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# 使用交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练函数
def train(epoch):  # 移除 verbose 参数
    # 设置为训练模式
    model.train()
    total = 0  # 总样本数
    correct = 0  # 正确预测数
    epoch_loss = 0  # 累计损失
    # 遍历训练数据
    for batch_idx, (data, target) in enumerate(train_loader):
        # 数据移到设备
        data, target = data.to(device), target.to(device)
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        output = model(data)  # 移除 verbose 参数
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        # 累计损失
        epoch_loss += loss.item()
        # 计算准确率
        _, predicted = torch.max(output.data, 1)  # 获取预测类别
        total += target.size(0)  # 累计样本数
        correct += (predicted == target).sum().item()  # 累计正确数

    # 打印每个epoch的平均损失和准确率
    print(f'Train Epoch: {epoch}\tAverage Loss: {epoch_loss / len(train_loader):.6f}\t'
          f'Accuracy: {100 * correct / total:.2f}%')

# 测试函数
def test():  # 移除 verbose 参数
    # 设置为评估模式
    model.eval()
    test_loss = 0  # 测试损失
    correct = 0  # 正确预测数
    total = 0  # 总样本数
    # 不计算梯度
    with torch.no_grad():
        # 遍历测试数据
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # 前向传播
            output = model(data)  # 移除 verbose 参数
            # 累计损失
            test_loss += criterion(output, target).item()
            # 计算准确率
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    # 计算平均损失
    test_loss /= len(test_loader)
    # 打印测试结果
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {100 * correct / total:.2f}%\n')

# 训练和测试循环
epochs = 10  # 训练轮数
for epoch in range(1, epochs + 1):
    train(epoch)  # 移除 verbose 参数
    test()  # 移除 verbose 参数

# 保存模型参数
torch.save(model.state_dict(), 'resnet_mnist.pth')
