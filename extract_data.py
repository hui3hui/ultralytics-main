import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
#数据处理（不用了）
# 1. 加载模型
model = YOLO('yolov8n-pose.pt')

def process_video(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    data_list = []
    frame_count = 0
    valid_frame_count = 0

    print(f"🔄 开始处理视频: {video_path}...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # YOLO 推理 (verbose=False 关闭日志)
        results = model(frame, verbose=False)
        
        # ==================== 核心修改部分 Start ====================
        # 这里的逻辑是：先看有没有检测结果，再看检测结果里有没有框/点
        try:
            # 检查是否检测到了物体 (results[0].boxes.id 或 results[0].keypoints.data)
            # 如果 keypoints.data 的形状是 [0, 17, 3]，说明没人
            if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
                
                # 取出归一化坐标 xyn (x, y normalized)
                # shape 通常是 (N, 17, 2)，N是人数。我们要取第一个人 [0]
                keypoints_xyn = results[0].keypoints.xyn[0].cpu().numpy()
                
                # 双重检查：确保取出来的数组真的有内容
                if keypoints_xyn.shape == (17, 2):
                    # 展平为一行: [x1, y1, x2, y2, ...]
                    flat_keypoints = keypoints_xyn.flatten()
                    data_list.append(flat_keypoints)
                    valid_frame_count += 1
                else:
                    # 极少情况：有人但关键点数据异常
                    pass
            else:
                # 这一帧没人，跳过
                # print(f"⚠️ 第 {frame_count} 帧未检测到人")
                pass

        except Exception as e:
            print(f"❌ 第 {frame_count} 帧处理出错: {e}")
            continue
        # ==================== 核心修改部分 End ====================

        if frame_count % 50 == 0:
            print(f"   已扫描 {frame_count} 帧 | 有效采集: {valid_frame_count} 帧")

    cap.release()

    if len(data_list) == 0:
        print("❌ 错误：整个视频都没有检测到任何人！请检查视频是否有人物。")
        return

    # 保存 CSV
    col_names = []
    for i in range(17):
        col_names.append(f"x{i}")
        col_names.append(f"y{i}")
    
    df = pd.DataFrame(data_list, columns=col_names)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ 处理完成！")
    print(f"📊 总扫描帧数: {frame_count}")
    print(f"💾 有效骨架数据: {valid_frame_count} 行")
    print(f"📂 结果已保存至: {output_csv}")

# 请修改此处路径
video_file = "test_fall.mp4" 
output_file = "fall_data.csv"

process_video(video_file, output_file)