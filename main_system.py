import sys
import winsound  # Windows系统自带的声音库
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ultralytics import YOLO


# 可视化界面
# ================= 1. 核心算法线程 (不卡界面) =================
class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)  # 发送图像信号
    update_log_signal = pyqtSignal(str)  # 发送日志信号
    update_status_signal = pyqtSignal(str, float)  # 发送状态信号

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.sensitivity = 0.75  # 默认灵敏度

        # === 模型初始化 (复用你之前的代码) ===
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定义 LSTM (必须与训练一致)
        class FallLSTM(nn.Module):
            def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
                super(FallLSTM, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)

            def forward(self, x):
                h0 = torch.zeros(2, x.size(0), 64).to(x.device)
                c0 = torch.zeros(2, x.size(0), 64).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = out[:, -1, :]
                out = self.fc(out)
                return out

        try:
            self.yolo_model = YOLO("yolov8n-pose.pt")
            self.lstm_model = FallLSTM().to(self.device)
            self.lstm_model.load_state_dict(torch.load("fall_detection_model.pth", map_location=self.device))
            self.lstm_model.eval()
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading models: {e}")

    def run(self):
        if not self.model_loaded:
            self.update_log_signal.emit("❌ 模型加载失败，请检查文件！")
            return

        cap = cv2.VideoCapture(0)
        # 降低分辨率提速
        cap.set(3, 800)
        cap.set(4, 600)

        sequence = deque(maxlen=30)
        fall_counter = 0
        TRIGGER_FRAME = 3

        self.update_log_signal.emit("🎥 摄像头已启动，系统监测中...")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. 检测
            results = self.yolo_model(frame, classes=0, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()

            current_status = "Normal"
            prob = 0.0

            # 2. 逻辑处理
            if results[0].keypoints is not None and results[0].keypoints.data.shape[0] > 0:
                keypoints = results[0].keypoints.xyn[0].cpu().numpy().flatten()
                sequence.append(keypoints)

                if len(sequence) == 30:
                    input_seq = torch.tensor(np.array([sequence]), dtype=torch.float32).to(self.device)
                    with torch.no_grad():
                        output = self.lstm_model(input_seq)
                        probs = torch.softmax(output, dim=1)
                        fall_prob = probs[0][1].item()
                        prob = fall_prob

                    if fall_prob > self.sensitivity:
                        fall_counter += 1
                    else:
                        fall_counter = 0

                    # ... 原有代码 ...
                    if fall_counter >= TRIGGER_FRAME:
                        current_status = "FALL"

                        # --- 🌟 新增：报警逻辑 ---
                        if fall_counter == TRIGGER_FRAME:
                            # 1. 写日志
                            now = datetime.now().strftime("%H:%M:%S")
                            self.update_log_signal.emit(f"⚠️ [{now}] 跌倒确认！正在触发声光报警...")
                            self.update_log_signal.emit("📨 [模拟] 已发送短信至监护人: 老人在客厅跌倒！")

                            # 2. 发出声音 (频率1000Hz, 持续500毫秒) -> "嘀————"
                            # 注意：这个Beep会轻微阻塞线程，但在独立线程里跑问题不大，反而能起到限流作用
                            winsound.Beep(2000, 500)
                        # -----------------------
            # 发送信号给界面
            self.change_pixmap_signal.emit(annotated_frame)
            self.update_status_signal.emit(current_status, prob)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, value):
        self.sensitivity = value / 100.0


# ================= 2. 主界面 GUI =================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的行人跌倒检测系统 - 毕业设计")
        self.resize(1200, 800)
        self.setStyleSheet("background-color: #f0f0f0;")

        # === 布局 ===
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # --- 左侧：视频区 ---
        video_group = QGroupBox("实时监控画面")
        video_layout = QVBoxLayout()
        self.video_label = QLabel()
        self.video_label.setFixedSize(800, 600)
        self.video_label.setStyleSheet("background-color: black; border: 2px solid #666;")
        self.video_label.setAlignment(Qt.AlignCenter)  # 居中
        video_layout.addWidget(self.video_label)

        # 状态大字
        self.status_label = QLabel("系统就绪")
        self.status_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: green; padding: 10px;")
        video_layout.addWidget(self.status_label)

        video_group.setLayout(video_layout)
        main_layout.addWidget(video_group, stretch=2)

        # --- 右侧：控制与日志 ---
        right_layout = QVBoxLayout()

        # 1. 控制面板
        control_group = QGroupBox("系统控制")
        control_layout = QVBoxLayout()

        self.btn_start = QPushButton("启动系统")
        self.btn_start.setMinimumHeight(50)
        self.btn_start.setStyleSheet("background-color: #2196F3; color: white; font-size: 16px; border-radius: 5px;")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setMinimumHeight(50)
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; border-radius: 5px;")
        self.btn_stop.clicked.connect(self.stop_detection)
        self.btn_stop.setEnabled(False)

        # 灵敏度滑块
        slider_label = QLabel("灵敏度调节:")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 95)  # 0.5 - 0.95
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.change_sensitivity)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addWidget(slider_label)
        control_layout.addWidget(self.slider)
        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        # 2. 日志面板
        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: white; font-family: Consolas;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, stretch=1)

        main_layout.addLayout(right_layout, stretch=1)

        # === 线程初始化 ===
        self.thread = None

    def start_detection(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.log_text.append("--- 系统启动 ---")

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.update_log)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.sensitivity = self.slider.value() / 100.0  # 初始灵敏度
        self.thread.start()

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("监控已停止")
        self.status_label.setText("系统就绪")
        self.status_label.setStyleSheet("color: green;")
        self.log_text.append("--- 系统停止 ---")

    def change_sensitivity(self):
        val = self.slider.value()
        self.log_text.append(f"🔧 灵敏度调整为: {val / 100.0}")
        if self.thread:
            self.thread.update_sensitivity(val)

    def update_image(self, cv_img):
        """将OpenCV图像转换为Qt图像显示."""
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_log(self, text):
        self.log_text.append(text)

    def update_status(self, status, prob):
        if status == "FALL":
            self.status_label.setText(f"检测到跌倒! ({prob:.2f})")
            self.status_label.setStyleSheet("background-color: red; color: white;")
        else:
            self.status_label.setText(f"正常监控中 ({1 - prob:.2f})")
            self.status_label.setStyleSheet("background-color: #e0e0e0; color: green;")

    def convert_cv_qt(self, cv_img):
        """转换函数."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(800, 600, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
