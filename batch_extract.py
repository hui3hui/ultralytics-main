import cv2
import pandas as pd
import os
import glob
from ultralytics import YOLO
#批处理视频

# 1. 加载模型
model = YOLO('yolov8n-pose.pt')

def process_folder(folder_path, label_tag):
    """
    读取文件夹下所有视频，提取骨架，合并成一个大列表
    folder_path: 视频文件夹路径
    label_tag: 只是为了打印日志用
    """
    video_files = glob.glob(os.path.join(folder_path, "*.mp4")) + \
                  glob.glob(os.path.join(folder_path, "*.avi")) # 支持mp4和avi
    
    all_data = []
    print(f"📂 正在处理 {label_tag} 数据，共找到 {len(video_files)} 个视频...")

    for video_file in video_files:
        cap = cv2.VideoCapture(video_file)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            # 这里的 conf=0.5 是为了过滤掉背景里的杂乱识别
            results = model(frame, verbose=False, conf=0.5)
            
            # 只有检测到人且有关键点时才保存
            if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
                # 取第一个人的归一化坐标
                keypoints = results[0].keypoints.xyn[0].cpu().numpy()
                
                # 再次检查是否有全0数据 (有时YOLO会输出全0)
                if keypoints.any(): 
                    flat_keypoints = keypoints.flatten()
                    all_data.append(flat_keypoints)
            
            frame_idx += 1
        
        cap.release()
        print(f"  -> {os.path.basename(video_file)} 处理完毕")

    return all_data

# ================= 配置路径 =================
# ⚠️ 请修改这里的路径为你电脑上实际的路径
fall_folder = "dataset/fall"   # 跌倒视频的文件夹
normal_folder = "dataset/adl"  # 正常视频的文件夹

# ================= 开始处理 =================

# 1. 处理跌倒数据
print("--- 开始提取跌倒数据 ---")
fall_list = process_folder(fall_folder, "跌倒(Fall)")

# 保存 fall_data.csv
col_names = [f"{axis}{i}" for i in range(17) for axis in ['x', 'y']] # 生成 x0, y0, ...
if len(fall_list) > 0:
    df_fall = pd.DataFrame(fall_list, columns=col_names)
    df_fall.to_csv("fall_data.csv", index=False)
    print(f"✅ 成功生成 fall_data.csv，共 {len(df_fall)} 帧数据")
else:
    print("❌ 跌倒文件夹里没读到数据，请检查路径！")

# 2. 处理正常数据
print("\n--- 开始提取正常数据 ---")
normal_list = process_folder(normal_folder, "正常(ADL)")

# 保存 normal_data.csv
if len(normal_list) > 0:
    df_normal = pd.DataFrame(normal_list, columns=col_names)
    df_normal.to_csv("normal_data.csv", index=False)
    print(f"✅ 成功生成 normal_data.csv，共 {len(df_normal)} 帧数据")
else:
    print("❌ 正常文件夹里没读到数据，请检查路径！")