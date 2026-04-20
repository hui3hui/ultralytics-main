import os

import matplotlib.pyplot as plt  # 用于画图
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# 模型训练

# ================= 1. 配置参数 (Hyperparameters) =================
# 这些参数可以写进论文的"实验设置"章节
SEQUENCE_LENGTH = 30  # 时间窗口：看过去30帧来判断动作
INPUT_SIZE = 34  # 特征数：17个关键点 * 2个坐标(x,y)
HIDDEN_SIZE = 64  # LSTM隐藏层神经元数量
NUM_LAYERS = 2  # LSTM堆叠层数
NUM_CLASSES = 2  # 类别数：0=正常(Normal), 1=跌倒(Fall)
BATCH_SIZE = 32  # 每一批训练多少个样本
EPOCHS = 50  # 训练总轮数
LEARNING_RATE = 0.001  # 学习率

# 检查GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 正在使用计算设备: {device}")


# ================= 2. 数据集定义 =================
class PoseDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 滑动窗口切分函数
def create_sequences(data, seq_length):
    sequences = []
    if len(data) < seq_length:
        return np.array([])

    for i in range(len(data) - seq_length):
        seq = data[i : i + seq_length]
        sequences.append(seq)
    return np.array(sequences)


# ================= 3. 加载与预处理数据 =================
print("🔄 正在加载 CSV 数据...")

# 检查文件
if not os.path.exists("fall_data.csv") or not os.path.exists("normal_data.csv"):
    print("❌ 错误：找不到 CSV 文件！请先运行 batch_extract.py 生成数据。")
    exit()

# 读取 CSV
df_fall = pd.read_csv("fall_data.csv")
df_normal = pd.read_csv("normal_data.csv")

# 转换为 numpy 数组
fall_data = df_fall.values
normal_data = df_normal.values

print(f"   原始数据量 -> 跌倒帧数: {len(fall_data)}, 正常帧数: {len(normal_data)}")

# 制作序列 (Sliding Window)
# 结果形状: (样本数, 30, 34)
X_fall = create_sequences(fall_data, SEQUENCE_LENGTH)
X_normal = create_sequences(normal_data, SEQUENCE_LENGTH)

print(f"📊 切分后序列 -> 跌倒样本: {len(X_fall)}, 正常样本: {len(X_normal)}")

if len(X_fall) == 0 or len(X_normal) == 0:
    print("❌ 数据不足以构成一个序列！请录制更长的视频。")
    exit()

# 创建标签 (Labels)
# 跌倒 = 1, 正常 = 0
y_fall = np.ones(len(X_fall))
y_normal = np.zeros(len(X_normal))

# 合并数据
X = np.concatenate((X_fall, X_normal), axis=0)
y = np.concatenate((y_fall, y_normal), axis=0)

# 简单的打乱数据并划分 训练集(80%) / 测试集(20%)
# 为了简单起见，这里直接用 DataLoader 的 shuffle，不单独切分验证集了
dataset = PoseDataset(X, y)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ================= 4. 定义 LSTM 模型 =================
class FallLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # batch_first=True -> 输入格式 (batch, seq, feature)
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        self.dropout = nn.Dropout(0.5)  # 防止过拟合

    def forward(self, x):
        # 初始化隐藏状态 (h0, c0)
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(device)

        # LSTM 前向传播
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # 取最后一个时间步的输出
        out = out[:, -1, :]

        # Dropout 和 全连接分类
        out = self.dropout(out)
        out = self.fc(out)
        return out


# 初始化模型
model = FallLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ================= 5. 开始训练 =================
print("\n🚀 开始训练模型...")
loss_history = []
acc_history = []

for epoch in range(EPOCHS):
    total_loss = 0
    correct = 0
    total = 0

    model.train()  # 切换到训练模式

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()
        # 2. 前向传播
        outputs = model(inputs)
        # 3. 计算损失
        loss = criterion(outputs, labels)
        # 4. 反向传播
        loss.backward()
        # 5. 更新参数
        optimizer.step()

        # 统计数据
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # 计算本轮平均 Loss 和 Accuracy
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    loss_history.append(avg_loss)
    acc_history.append(accuracy)

    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

# ================= 6. 保存结果 =================
# 保存模型权重
torch.save(model.state_dict(), "fall_detection_model.pth")
print("\n✅ 模型已保存为: fall_detection_model.pth")

# 绘制训练曲线 (可以直接放入论文)
plt.figure(figsize=(12, 5))

# Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(loss_history, label="Training Loss", color="red")
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(acc_history, label="Training Accuracy", color="blue")
plt.title("Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)

plt.savefig("training_results.png")
print("📊 训练曲线图已保存为: training_results.png (可用于论文插图)")
plt.show()
