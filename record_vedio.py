import time

import cv2

# 录制视频
# 配置
output_file = "hard_normal.mp4"
duration = 30  # 录制时长(秒)

cap = cv2.VideoCapture(0)
# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 获取实际参数以创建写入器
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = 30.0  # 默认FPS

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

print("🎥 准备开始录制！请在摄像头前做【蹲下、系鞋带、弯腰】等动作。")
print("3秒后开始...")
time.sleep(1)
print("2...")
time.sleep(1)
print("1...")
time.sleep(1)
print("🔴 录制开始！(持续30秒)")

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 写入文件
    out.write(frame)

    # 屏幕显示录制倒计时
    elapsed = time.time() - start_time
    remaining = duration - elapsed

    cv2.putText(frame, f"REC {remaining:.1f}s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Recording...", frame)

    if cv2.waitKey(1) & 0xFF == ord("q") or remaining <= 0:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ 录制完成！视频已保存为: {output_file}")
