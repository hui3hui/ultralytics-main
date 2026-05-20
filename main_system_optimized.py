import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import winsound
from datetime import datetime
from collections import deque
from ultralytics import YOLO

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QWidget, QTextEdit,
    QSlider, QGroupBox, QFrame, QProgressBar
)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_log_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str, float)
    update_runtime_signal = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.sensitivity = 0.75

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            self.yolo_model = YOLO('yolov8n-pose.pt')
            self.lstm_model = FallLSTM().to(self.device)
            self.lstm_model.load_state_dict(
                torch.load('fall_detection_model.pth', map_location=self.device)
            )
            self.lstm_model.eval()
            self.model_loaded = True
        except Exception as e:
            self.model_loaded = False
            print(f"Error loading models: {e}")

    def run(self):
        if not self.model_loaded:
            self.update_log_signal.emit("❌ 模型加载失败，请检查 yolov8n-pose.pt 和 fall_detection_model.pth 文件。")
            self.update_runtime_signal.emit("摄像头：未启动", "模型状态：加载失败", "告警状态：未触发")
            return

        cap = cv2.VideoCapture(0)
        cap.set(3, 800)
        cap.set(4, 600)

        sequence = deque(maxlen=30)
        fall_counter = 0
        trigger_frame = 3

        self.update_log_signal.emit("🎥 摄像头已启动，系统监测中...")
        self.update_runtime_signal.emit("摄像头：已启动", "模型状态：加载成功", "告警状态：未触发")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.update_log_signal.emit("⚠️ 无法读取摄像头画面。")
                break

            results = self.yolo_model(frame, classes=0, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()

            current_status = "Normal"
            prob = 0.0

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

                    if fall_counter >= trigger_frame:
                        current_status = "FALL"
                        self.update_runtime_signal.emit("摄像头：已启动", "模型状态：加载成功", "告警状态：已触发")

                        if fall_counter == trigger_frame:
                            now = datetime.now().strftime("%H:%M:%S")
                            self.update_log_signal.emit(f"⚠️ [{now}] 跌倒确认！正在触发声光报警...")
                            self.update_log_signal.emit("📨 [模拟] 已发送短信至监护人：老人/行人发生跌倒，请及时查看。")
                            winsound.Beep(2000, 500)
                    else:
                        self.update_runtime_signal.emit("摄像头：已启动", "模型状态：加载成功", "告警状态：未触发")
            else:
                self.update_runtime_signal.emit("摄像头：已启动", "模型状态：加载成功", "告警状态：未触发")

            self.change_pixmap_signal.emit(annotated_frame)
            self.update_status_signal.emit(current_status, prob)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, value):
        self.sensitivity = value / 100.0


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("行人跌倒检测系统")
        self.resize(1380, 860)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f7fb;
            }
            QGroupBox {
                font-size: 16px;
                font-weight: bold;
                color: #1f2937;
                border: 1px solid #d8dee9;
                border-radius: 14px;
                margin-top: 12px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px 0 6px;
            }
            QLabel {
                color: #1f2937;
                font-size: 14px;
            }
            QTextEdit {
                background: #0f172a;
                color: #e2e8f0;
                border: none;
                border-radius: 12px;
                padding: 10px;
                font-family: Consolas, Monaco, monospace;
                font-size: 13px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #d1d5db;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2563eb;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(18, 18, 18, 18)
        root_layout.setSpacing(16)

        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1d4ed8, stop:1 #0f766e
                );
                border-radius: 18px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(24, 18, 24, 18)

        title_box = QVBoxLayout()
        title = QLabel("行人跌倒检测系统")
        title.setStyleSheet("color: white; font-size: 28px; font-weight: 700;")
        subtitle = QLabel("YOLOv8-Pose + LSTM 时序行为识别")
        subtitle.setStyleSheet("color: rgba(255,255,255,0.88); font-size: 14px;")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)

        self.system_badge = QLabel("系统就绪")
        self.system_badge.setAlignment(Qt.AlignCenter)
        self.system_badge.setFixedSize(120, 42)
        self.system_badge.setStyleSheet("""
            QLabel {
                background: rgba(255,255,255,0.18);
                color: white;
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 12px;
                font-size: 14px;
                font-weight: 600;
            }
        """)

        header_layout.addLayout(title_box)
        header_layout.addStretch()
        header_layout.addWidget(self.system_badge)
        root_layout.addWidget(header)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        root_layout.addLayout(body_layout)

        left_group = QGroupBox("实时监控画面")
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(16, 20, 16, 16)
        left_layout.setSpacing(14)

        self.video_label = QLabel("等待启动摄像头...")
        self.video_label.setFixedSize(860, 620)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #111827;
                color: #cbd5e1;
                border: 2px solid #1f2937;
                border-radius: 18px;
                font-size: 20px;
            }
        """)
        left_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        self.status_card = QFrame()
        self.status_card.setStyleSheet("""
            QFrame {
                background-color: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 16px;
            }
        """)
        status_layout = QVBoxLayout(self.status_card)
        status_layout.setContentsMargins(18, 14, 18, 14)

        self.status_title = QLabel("当前状态")
        self.status_title.setStyleSheet("font-size: 14px; color: #065f46;")
        self.status_label = QLabel("系统就绪")
        self.status_label.setStyleSheet("font-size: 28px; font-weight: 700; color: #047857;")
        self.prob_label = QLabel("跌倒概率：0.00")
        self.prob_label.setStyleSheet("font-size: 14px; color: #065f46;")

        status_layout.addWidget(self.status_title)
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.prob_label)
        left_layout.addWidget(self.status_card)

        left_group.setLayout(left_layout)
        body_layout.addWidget(left_group, stretch=3)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(16)
        body_layout.addLayout(right_layout, stretch=1)

        control_group = QGroupBox("系统控制")
        control_layout = QVBoxLayout()
        control_layout.setContentsMargins(16, 20, 16, 16)
        control_layout.setSpacing(12)

        self.btn_start = QPushButton("启动系统")
        self.btn_start.setMinimumHeight(48)
        self.btn_start.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #1d4ed8;
            }
            QPushButton:disabled {
                background-color: #93c5fd;
            }
        """)
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("停止检测")
        self.btn_stop.setMinimumHeight(48)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #ef4444;
                color: white;
                border: none;
                border-radius: 12px;
                font-size: 16px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #dc2626;
            }
            QPushButton:disabled {
                background-color: #fca5a5;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_detection)

        self.slider_title = QLabel("检测灵敏度：0.75")
        self.slider_title.setStyleSheet("font-size: 14px; font-weight: 600;")

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 95)
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.change_sensitivity)

        self.sensitivity_progress = QProgressBar()
        self.sensitivity_progress.setRange(50, 95)
        self.sensitivity_progress.setValue(75)
        self.sensitivity_progress.setTextVisible(False)
        self.sensitivity_progress.setFixedHeight(10)
        self.sensitivity_progress.setStyleSheet("""
            QProgressBar {
                background: #e5e7eb;
                border: none;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: #2563eb;
                border-radius: 5px;
            }
        """)

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.setMinimumHeight(42)
        self.btn_clear_log.setStyleSheet("""
            QPushButton {
                background-color: #475569;
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #334155;
            }
        """)
        self.btn_clear_log.clicked.connect(self.clear_log)

        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_stop)
        control_layout.addSpacing(8)
        control_layout.addWidget(self.slider_title)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.sensitivity_progress)
        control_layout.addWidget(self.btn_clear_log)

        control_group.setLayout(control_layout)
        right_layout.addWidget(control_group)

        info_group = QGroupBox("运行信息")
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(16, 20, 16, 16)
        info_layout.setSpacing(10)

        self.camera_label = QLabel("摄像头：未启动")
        self.model_label = QLabel("模型状态：待加载")
        self.alert_label = QLabel("告警状态：未触发")

        for widget in [self.camera_label, self.model_label, self.alert_label]:
            widget.setStyleSheet("""
                QLabel {
                    background: #f8fafc;
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 10px;
                    font-size: 14px;
                }
            """)
            info_layout.addWidget(widget)

        info_group.setLayout(info_layout)
        right_layout.addWidget(info_group)

        log_group = QGroupBox("检测日志")
        log_layout = QVBoxLayout()
        log_layout.setContentsMargins(16, 20, 16, 16)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("系统日志将在这里显示...")
        log_layout.addWidget(self.log_text)

        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group, stretch=1)

        self.thread = None

    def start_detection(self):
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.system_badge.setText("运行中")
        self.camera_label.setText("摄像头：正在启动")
        self.model_label.setText("模型状态：正在加载")
        self.log_text.append("—— 系统启动 ——")

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.update_log)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.update_runtime_signal.connect(self.update_runtime_info)
        self.thread.sensitivity = self.slider.value() / 100.0
        self.thread.start()

    def stop_detection(self):
        if self.thread:
            self.thread.stop()
            self.thread = None

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.system_badge.setText("已停止")
        self.camera_label.setText("摄像头：未启动")
        self.model_label.setText("模型状态：待加载")
        self.alert_label.setText("告警状态：未触发")

        self.video_label.clear()
        self.video_label.setText("监控已停止")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.status_card.setStyleSheet("""
            QFrame {
                background-color: #ecfdf5;
                border: 1px solid #a7f3d0;
                border-radius: 16px;
            }
        """)
        self.status_label.setText("系统就绪")
        self.status_label.setStyleSheet("font-size: 28px; font-weight: 700; color: #047857;")
        self.prob_label.setText("跌倒概率：0.00")
        self.prob_label.setStyleSheet("font-size: 14px; color: #065f46;")
        self.log_text.append("—— 系统停止 ——")

    def change_sensitivity(self):
        value = self.slider.value()
        self.slider_title.setText(f"检测灵敏度：{value / 100:.2f}")
        self.sensitivity_progress.setValue(value)
        if self.thread:
            self.thread.update_sensitivity(value)
        self.log_text.append(f"🔧 灵敏度调整为：{value / 100:.2f}")

    def clear_log(self):
        self.log_text.clear()
        self.log_text.append("—— 日志已清空 ——")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_log(self, text):
        self.log_text.append(text)

    def update_runtime_info(self, camera_text, model_text, alert_text):
        self.camera_label.setText(camera_text)
        self.model_label.setText(model_text)
        self.alert_label.setText(alert_text)

    def update_status(self, status, prob):
        self.prob_label.setText(f"跌倒概率：{prob:.2f}")

        if status == "FALL":
            self.system_badge.setText("告警中")
            self.status_card.setStyleSheet("""
                QFrame {
                    background-color: #fef2f2;
                    border: 1px solid #fecaca;
                    border-radius: 16px;
                }
            """)
            self.status_label.setText("检测到跌倒")
            self.status_label.setStyleSheet("font-size: 28px; font-weight: 700; color: #dc2626;")
            self.prob_label.setStyleSheet("font-size: 14px; color: #991b1b;")
        else:
            self.system_badge.setText("运行中" if self.thread else "系统就绪")
            self.status_card.setStyleSheet("""
                QFrame {
                    background-color: #ecfdf5;
                    border: 1px solid #a7f3d0;
                    border-radius: 16px;
                }
            """)
            self.status_label.setText("正常监控")
            self.status_label.setStyleSheet("font-size: 28px; font-weight: 700; color: #047857;")
            self.prob_label.setStyleSheet("font-size: 14px; color: #065f46;")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qt_format.scaled(860, 620, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(scaled)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
