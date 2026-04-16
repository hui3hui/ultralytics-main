import cv2
import pandas as pd
import os
from ultralytics import YOLO
#追加自制视频

# ================= 配置 =================
video_path = "hard_normal.mp4"     # 你刚录的视频
csv_path = "normal_data.csv"       # 要追加的目标文件
# =======================================

print("🔄 Loading YOLOv8...")
model = YOLO('yolov8n-pose.pt')

if not os.path.exists(video_path):
    print(f"❌ 错误：找不到视频 {video_path}，请先运行录制脚本！")
    exit()

print(f"🚀 开始处理视频: {video_path} ...")
cap = cv2.VideoCapture(video_path)
new_data = []

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # 提取骨架 (只保留置信度高的)
    results = model(frame, verbose=False, conf=0.5)
    
    if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
        # 取第一个人的归一化坐标
        keypoints = results[0].keypoints.xyn[0].cpu().numpy()
        flat_keypoints = keypoints.flatten()
        new_data.append(flat_keypoints)

cap.release()

if len(new_data) == 0:
    print("❌ 视频里没检测到人，无法追加数据！")
    exit()

# ================= 核心：追加到 CSV =================
# 1. 转换为 DataFrame
col_names = [f"{axis}{i}" for i in range(17) for axis in ['x', 'y']]
df_new = pd.DataFrame(new_data, columns=col_names)

# 2. 读取旧文件以确认存在 (可选)
if os.path.exists(csv_path):
    # mode='a' 表示 append (追加)
    # header=False 表示不重复写入表头(x0, y0...)
    df_new.to_csv(csv_path, mode='a', header=False, index=False)
    print(f"✅ 成功追加 {len(df_new)} 行困难样本到 {csv_path}！")
    print("💪 现在的 normal_data.csv 更强壮了！")
else:
    # 如果旧文件不存在，就直接存为新文件
    df_new.to_csv(csv_path, index=False)
    print(f"⚠️ {csv_path} 不存在，已创建新文件。")