import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque
import time
#实时检测
# 模型定义 
class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
        super(FallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5) 
        
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ================= 初始化 =================
SEQUENCE_LENGTH = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Loading models...")
yolo_model = YOLO('yolov8n-pose.pt')
lstm_model = FallLSTM().to(device)
lstm_model.load_state_dict(torch.load('fall_detection_model.pth', map_location=device))
lstm_model.eval()

# ================= 🌟 新增：调节窗口 =================
window_name = 'Fall Detection System'
cv2.namedWindow(window_name)

def nothing(x):
    pass

# 创建滑块：范围 0-100，默认 70 (即 0.7)
cv2.createTrackbar('Sensitivity', window_name, 70, 100, nothing)

# ================= 主循环 =================
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

sequence = deque(maxlen=SEQUENCE_LENGTH)

# 🌟 状态稳定器
fall_counter = 0     # 记录连续跌倒帧数
TRIGGER_FRAME = 3    # 只有连续 3 帧都说是跌倒，才真的报警
alarm_state = False  # 当前是否报警

while True:
    ret, frame = cap.read()
    if not ret: break

    # 获取当前滑块的值 (0-100 -> 0.0-1.0)
    current_threshold = cv2.getTrackbarPos('Sensitivity', window_name) / 100.0

    results = yolo_model(frame, classes=0, verbose=False, conf=0.5)
    annotated_frame = results[0].plot()
    
    current_prob = 0.0
    
    if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
        keypoints = results[0].keypoints.xyn[0].cpu().numpy().flatten()
        sequence.append(keypoints)
        
        if len(sequence) == SEQUENCE_LENGTH:
            input_seq = torch.tensor(np.array([sequence]), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = lstm_model(input_seq)
                probabilities = torch.softmax(output, dim=1)
                fall_prob = probabilities[0][1].item()
                current_prob = fall_prob
                
            # 🌟 核心：防抖动逻辑
            if fall_prob > current_threshold:
                fall_counter += 1
            else:
                fall_counter = 0 # 一旦断了，就重置

            # 只有连续积累超过 TRIGGER_FRAME 帧，才切换状态
            if fall_counter >= TRIGGER_FRAME:
                alarm_state = True
            elif fall_counter == 0:
                alarm_state = False

    # ================= 绘制 UI =================
    # 1. 顶部状态条
    if alarm_state:
        color = (0, 0, 255)
        text = f"FALL DETECTED! ({current_prob:.2f})"
    else:
        color = (0, 255, 0)
        text = f"Normal ({1-current_prob:.2f})"
        
    cv2.rectangle(annotated_frame, (0, 0), (1280, 60), color, -1)
    cv2.putText(annotated_frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    # 2. 绘制概率进度条 (更直观)
    # 底部画一个灰条，上面覆盖一个红条表示跌倒概率
    cv2.rectangle(annotated_frame, (20, 650), (320, 680), (100, 100, 100), -1)
    bar_width = int(current_prob * 300)
    # 概率越高越红
    bar_color = (0, int(255*(1-current_prob)), int(255*current_prob)) 
    cv2.rectangle(annotated_frame, (20, 650), (20 + bar_width, 680), bar_color, -1)
    cv2.putText(annotated_frame, f"Fall Prob: {current_threshold:.2f} (Thresh)", (330, 675), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow(window_name, annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()