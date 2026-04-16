import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
#混淆矩阵，准确率评定
# ================= 配置 =================
SEQUENCE_LENGTH = 30
INPUT_SIZE = 34
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 2
BATCH_SIZE = 32

# ================= 模型定义 (保持一致) =================
class FallLSTM(nn.Module):
    def __init__(self):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
        
    def forward(self, x):
        h0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        c0 = torch.zeros(NUM_LAYERS, x.size(0), HIDDEN_SIZE).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ================= 数据处理 =================
def create_sequences(data, seq_length):
    sequences = []
    if len(data) < seq_length: return np.array([])
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
    return np.array(sequences)

# 加载数据
print("🔄 正在加载所有数据进行最终评估...")
df_fall = pd.read_csv('fall_data.csv')
df_normal = pd.read_csv('normal_data.csv')

X_fall = create_sequences(df_fall.values, SEQUENCE_LENGTH)
X_normal = create_sequences(df_normal.values, SEQUENCE_LENGTH)

# 创建标签
y_fall = np.ones(len(X_fall))
y_normal = np.zeros(len(X_normal))

# 合并
X = np.concatenate((X_fall, X_normal), axis=0)
y = np.concatenate((y_fall, y_normal), axis=0)

# 转 Tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y, dtype=torch.long).to(device)

# ================= 加载模型与预测 =================
model = FallLSTM().to(device)
try:
    model.load_state_dict(torch.load('fall_detection_model.pth'))
    model.eval()
except:
    print("❌ 找不到模型文件！")
    exit()

print("🚀 开始预测...")
with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs.data, 1)

# 转回 CPU
y_true = y_tensor.cpu().numpy()
y_pred = predicted.cpu().numpy()

# ================= 画混淆矩阵 =================
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fall'], 
            yticklabels=['Normal', 'Fall'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ 混淆矩阵已保存为 confusion_matrix.png")

# 打印详细报告
report = classification_report(y_true, y_pred, target_names=['Normal', 'Fall'])
print("\n" + "="*30)
print("实验最终报告 (请复制到论文中)")
print("="*30)
print(report)