# 导入必要的库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import numpy as np  # 数值计算库
import matplotlib.pyplot as plt  # 绘图库
from sklearn.preprocessing import MinMaxScaler  # 数据归一化
import pandas as pd  # 数据处理库
import os  # 操作系统接口
from urllib.request import urlretrieve  # 文件下载

# 超参数设置
SEQ_LENGTH = 12  # 输入序列长度（12个月为一个周期）
HIDDEN_SIZE = 64  # RNN隐藏层神经元数量
EPOCHS = 200  # 训练轮次
BATCH_SIZE = 8  # 每个批次的样本数
LEARNING_RATE = 0.01  # 学习率

# 自动下载数据集（航空乘客数据）
DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
DATA_PATH = 'airline-passengers.csv'

if not os.path.exists(DATA_PATH):
    try:
        print("正在下载数据集...")
        urlretrieve(DATA_URL, DATA_PATH)  # 下载CSV文件
        print("下载完成！")
    except Exception as e:
        print(f"下载失败: {e}")
        exit()

# 加载数据
df = pd.read_csv(DATA_PATH, usecols=['Passengers'])  # 只读取乘客数列
data = df.values.astype('float32')  # 转换为numpy数组并指定数据类型

# 数据预处理
scaler = MinMaxScaler(feature_range=(-1, 1))  # 创建归一化器（范围-1到1）
data_normalized = scaler.fit_transform(data)  # 应用归一化

def create_sequences(data, seq_length):
    """创建时间序列样本"""
    X, Y = [], []
    for i in range(len(data) - seq_length - 1):
        # 每个输入样本是连续的seq_length个数据点
        X.append(data[i:(i + seq_length)])
        # 对应的输出是下一个时间点的数据
        Y.append(data[i + seq_length])
    return np.array(X), np.array(Y)

# 划分训练集测试集
train_size = int(len(data_normalized) * 0.8)  # 80%作为训练集
train_data = data_normalized[:train_size]
test_data = data_normalized[train_size:]

# 创建序列样本
X_train, y_train = create_sequences(train_data, SEQ_LENGTH)
X_test, y_test = create_sequences(test_data, SEQ_LENGTH)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)  # 形状变为(N_samples, SEQ_LENGTH, 1)
y_train = torch.FloatTensor(y_train)  # 形状变为(N_samples, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test)

# 定义简单RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,  # 输入特征维度（乘客数只有1个特征）
            hidden_size=hidden_size,  # 隐藏层维度
            batch_first=True  # 输入格式为(batch, seq, feature)
        )
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层输出预测值

    def forward(self, x):
        # RNN前向传播
        out, _ = self.rnn(x)  # 输出形状：(batch_size, seq_length, hidden_size)
        out = out[:, -1, :]  # 取最后一个时间步的输出（序列预测任务常用做法）
        out = self.fc(out)  # 通过全连接层得到预测值
        return out


# 初始化模型、损失函数和优化器
model = RNNModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 训练循环
for epoch in range(EPOCHS):
    model.train()  # 设置训练模式
    for i in range(0, len(X_train), BATCH_SIZE):
        # 获取当前批次数据 --------------------------------------------------
        # 这是关于batch的关键部分：
        # 假设X_train的形状是(N, SEQ_LENGTH, 1)
        # 每个batch的形状是(BATCH_SIZE, SEQ_LENGTH, 1)
        # 例如当BATCH_SIZE=8时：
        # batch_X.shape = (8, 12, 1) 表示8个样本，每个样本12个月的数据，每个时间点1个特征
        # batch_y.shape = (8, 1) 对应8个样本的下个月乘客数
        batch_X = X_train[i:i + BATCH_SIZE]
        batch_y = y_train[i:i + BATCH_SIZE]

        # 前向传播
        outputs = model(batch_X)  # outputs形状：(BATCH_SIZE, 1)
        loss = criterion(outputs, batch_y)

        # 反向传播
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

    # 每20个epoch打印损失
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}')

# 预测
model.eval()  # 设置评估模式
with torch.no_grad():  # 禁用梯度计算
    test_predict = model(X_test)

# 反归一化（将预测值转换回原始量纲）
test_predict = scaler.inverse_transform(test_predict.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Passengers', color='blue')
plt.plot(test_predict, label='Predicted Passengers', color='red', linestyle='--')
plt.title('Air Passengers Prediction with Simple RNN')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.legend()
plt.show()