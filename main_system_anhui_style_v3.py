import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from collections import deque
from ultralytics import YOLO

try:
    import winsound
except Exception:
    winsound = None

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QTextEdit,
    QSlider,
    QFrame,
    QProgressBar,
    QGraphicsDropShadowEffect,
    QSizePolicy,
    QGridLayout,
)
from PyQt5.QtGui import (
    QImage,
    QPixmap,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QLinearGradient,
    QPen,
    QPolygonF,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime, QRectF, QPointF


class FallLSTM(nn.Module):
    def __init__(self, input_size=34, hidden_size=64, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class DetectionThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    update_log_signal = pyqtSignal(str)
    update_status_signal = pyqtSignal(str, float)
    update_runtime_signal = pyqtSignal(str, str, str)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.sensitivity = 0.75
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.load_error = ""

        try:
            self.yolo_model = YOLO("yolov8n-pose.pt")
            self.lstm_model = FallLSTM().to(self.device)
            self.lstm_model.load_state_dict(
                torch.load("fall_detection_model.pth", map_location=self.device)
            )
            self.lstm_model.eval()
            self.model_loaded = True
        except Exception as e:
            self.load_error = str(e)

    def run(self):
        if not self.model_loaded:
            self.update_log_signal.emit("❌ 模型加载失败，请检查 yolov8n-pose.pt 和 fall_detection_model.pth 文件。")
            self.update_log_signal.emit(f"📌 详细原因：{self.load_error or '未知错误'}")
            self.update_runtime_signal.emit("摄像头：未启动", "模型状态：加载失败", "告警状态：未触发")
            self.update_status_signal.emit("ERROR", 0.0)
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            self.update_log_signal.emit("❌ 摄像头启动失败，请检查设备是否被占用。")
            self.update_runtime_signal.emit("摄像头：启动失败", "模型状态：加载成功", "告警状态：未触发")
            self.update_status_signal.emit("ERROR", 0.0)
            return

        sequence = deque(maxlen=30)
        fall_counter = 0
        trigger_frame = 3

        self.update_log_signal.emit("🎥 摄像头已启动，系统进入实时监测。")
        self.update_runtime_signal.emit("摄像头：已启动", "模型状态：加载成功", "告警状态：未触发")

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.update_log_signal.emit("⚠️ 无法读取摄像头画面。")
                break

            results = self.yolo_model(frame, classes=0, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()

            current_status = "MONITORING"
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
                            self.update_log_signal.emit(f"⚠️ [{now}] 检测到疑似跌倒，已达到确认阈值。")
                            self.update_log_signal.emit("📨 [模拟] 已发送提醒信息至监护端，请及时查看。")
                            if winsound is not None:
                                try:
                                    winsound.Beep(2000, 500)
                                except Exception:
                                    pass
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


class BackgroundWidget(QWidget):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QLinearGradient(0, 0, self.width(), self.height())
        bg.setColorAt(0.0, QColor("#f8fbff"))
        bg.setColorAt(0.46, QColor("#eef5fc"))
        bg.setColorAt(1.0, QColor("#e9f1fb"))
        painter.fillRect(self.rect(), bg)

        # 右上水印圆环
        painter.setPen(QPen(QColor(99, 133, 175, 18), 2))
        for r in (130, 180, 240):
            painter.drawEllipse(self.width() - 360 - r // 2, 26 - r // 8, r, r)

        painter.setPen(QColor(76, 108, 151, 18))
        painter.setFont(QFont("KaiTi", 36, QFont.Bold))
        painter.drawText(QRectF(self.width() - 470, 36, 380, 120), Qt.AlignCenter, "徽风建大")
        painter.setPen(QColor(88, 116, 157, 20))
        painter.setFont(QFont("Microsoft YaHei", 12))
        painter.drawText(QRectF(self.width() - 450, 108, 350, 50), Qt.AlignCenter, "Anhui Jianzhu University · Smart Care")

        # 顶部山影
        painter.setPen(QPen(QColor(117, 153, 196, 15), 1.4))
        top_path = QPainterPath()
        top_path.moveTo(self.width() * 0.62, 138)
        top_path.cubicTo(self.width() * 0.70, 92, self.width() * 0.81, 174, self.width() * 0.90, 126)
        top_path.cubicTo(self.width() * 0.94, 104, self.width() * 0.97, 98, self.width() + 10, 122)
        painter.drawPath(top_path)

        # 左下山纹
        left_back = QPainterPath()
        left_back.moveTo(-50, self.height())
        left_back.lineTo(-50, self.height() - 132)
        left_back.cubicTo(78, self.height() - 220, 190, self.height() - 24, 304, self.height() - 116)
        left_back.cubicTo(360, self.height() - 160, 438, self.height() - 168, 522, self.height() - 102)
        left_back.lineTo(630, self.height())
        left_back.closeSubpath()
        painter.fillPath(left_back, QColor(98, 138, 187, 24))

        left_front = QPainterPath()
        left_front.moveTo(-20, self.height())
        left_front.lineTo(-20, self.height() - 88)
        left_front.cubicTo(100, self.height() - 168, 222, self.height() - 42, 340, self.height() - 82)
        left_front.cubicTo(446, self.height() - 118, 532, self.height() - 136, 648, self.height() - 64)
        left_front.lineTo(740, self.height())
        left_front.closeSubpath()
        painter.fillPath(left_front, QColor(121, 159, 204, 20))

        # 右下山纹
        right_back = QPainterPath()
        right_back.moveTo(self.width() + 48, self.height())
        right_back.lineTo(self.width() + 48, self.height() - 182)
        right_back.cubicTo(self.width() - 54, self.height() - 266, self.width() - 182, self.height() - 48,
                           self.width() - 320, self.height() - 124)
        right_back.cubicTo(self.width() - 408, self.height() - 170, self.width() - 520, self.height() - 158,
                           self.width() - 620, self.height() - 96)
        right_back.lineTo(self.width() - 730, self.height())
        right_back.closeSubpath()
        painter.fillPath(right_back, QColor(105, 146, 191, 22))

        painter.setPen(QPen(QColor(126, 160, 200, 42), 1.2))
        for y in (self.height() - 84, self.height() - 104, self.height() - 126):
            painter.drawLine(36, y, 268, y - 18)
            painter.drawLine(self.width() - 278, y - 6, self.width() - 36, y - 26)

        super().paintEvent(event)


class UniversitySeal(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(98, 98)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#dce8f6"))
        painter.drawEllipse(2, 2, 94, 94)

        painter.setBrush(QColor("#2d5b90"))
        painter.drawEllipse(8, 8, 82, 82)

        painter.setBrush(QColor("#f8fbff"))
        painter.drawEllipse(16, 16, 66, 66)

        painter.setBrush(QColor("#2b5a8d"))
        painter.drawEllipse(25, 25, 48, 48)

        painter.setPen(QPen(QColor("#f8fbff"), 2))
        painter.drawLine(49, 31, 49, 67)
        painter.drawLine(35, 46, 63, 46)
        painter.drawEllipse(41, 38, 16, 16)

        painter.setPen(QColor("#f8fbff"))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.drawText(QRectF(20, 72, 58, 12), Qt.AlignCenter, "AHJZU")
        painter.setFont(QFont("KaiTi", 8, QFont.Bold))
        painter.drawText(QRectF(12, 15, 72, 12), Qt.AlignCenter, "安徽建筑大学")

        painter.end()
        super().paintEvent(event)


class DecorativeBanner(QFrame):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)

        bg = QLinearGradient(0, 0, self.width(), 0)
        bg.setColorAt(0.0, QColor("#fcfeff"))
        bg.setColorAt(0.56, QColor("#f5f9fe"))
        bg.setColorAt(1.0, QColor("#edf4fd"))
        painter.setBrush(bg)
        painter.setPen(QPen(QColor("#dbe7f3"), 1))
        painter.drawRoundedRect(rect, 28, 28)

        painter.setPen(QPen(QColor(120, 155, 196, 22), 1.2))
        curve = QPainterPath()
        curve.moveTo(self.width() * 0.60, self.height() * 0.78)
        curve.cubicTo(self.width() * 0.70, self.height() * 0.36, self.width() * 0.84, self.height() * 0.98,
                      self.width() * 0.97, self.height() * 0.62)
        painter.drawPath(curve)

        painter.setPen(QColor(88, 118, 160, 22))
        painter.setFont(QFont("KaiTi", 24, QFont.Bold))
        painter.drawText(QRectF(self.width() - 320, 10, 240, 46), Qt.AlignRight | Qt.AlignVCenter, "智慧校安")
        painter.setFont(QFont("Microsoft YaHei", 12))
        painter.drawText(QRectF(self.width() - 320, 44, 240, 28), Qt.AlignRight | Qt.AlignVCenter, "Campus Smart Safety")

        super().paintEvent(event)


class BlueRibbon(QFrame):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(0, 0, -1, -1)

        bg = QLinearGradient(0, 0, self.width(), 0)
        bg.setColorAt(0.0, QColor("#1f4f83"))
        bg.setColorAt(0.52, QColor("#396d9f"))
        bg.setColorAt(1.0, QColor("#7da9cf"))
        painter.setBrush(bg)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 16, 16)

        # 左侧小旗角
        painter.setBrush(QColor("#1a406a"))
        pennant = QPolygonF([
            QPointF(0, rect.height() - 2),
            QPointF(26, rect.height() - 2),
            QPointF(0, rect.height() + 14),
        ])
        painter.drawPolygon(pennant)

        painter.setPen(QPen(QColor(255, 255, 255, 18), 1))
        for x in range(-40, self.width() + 60, 44):
            painter.drawLine(x, 0, x + 64, self.height())

        super().paintEvent(event)


class SectionCard(QFrame):
    def __init__(self, title: str, subtitle: str = ""):
        super().__init__()
        self.setObjectName("sectionCard")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(20, 20, 20, 18)
        self.main_layout.setSpacing(12)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        title_layout.addWidget(self.title_label)

        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setObjectName("cardSubtitle")
            title_layout.addWidget(self.subtitle_label)
        else:
            self.subtitle_label = None

        header_layout.addLayout(title_layout)
        header_layout.addStretch()

        self.header_right_label = QLabel("")
        self.header_right_label.setObjectName("headerTag")
        self.header_right_label.hide()
        header_layout.addWidget(self.header_right_label)
        self.main_layout.addLayout(header_layout)
        self.apply_shadow()

    def set_header_tag(self, text: str):
        if text:
            self.header_right_label.setText(text)
            self.header_right_label.show()
        else:
            self.header_right_label.hide()

    def apply_shadow(self):
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(28)
        shadow.setOffset(0, 10)
        shadow.setColor(QColor(33, 72, 122, 35))
        self.setGraphicsEffect(shadow)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.stopping_manually = False

        self.setWindowTitle("行人跌倒检测系统")
        self.resize(1460, 900)
        self.setMinimumSize(1320, 820)

        self.setStyleSheet(
            """
            QMainWindow {
                background: #eef4fb;
            }
            QWidget {
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
                color: #183153;
            }
            #rootWidget {
                background: transparent;
            }
            #sectionCard {
                background: rgba(255, 255, 255, 0.96);
                border: 1px solid #dfe9f4;
                border-radius: 24px;
            }
            #cardTitle {
                font-size: 20px;
                font-weight: 800;
                color: #17375e;
            }
            #cardSubtitle {
                font-size: 13px;
                color: #6f87a2;
            }
            #headerTag {
                background: #edf4ff;
                border: 1px solid #cfe0f5;
                border-radius: 12px;
                padding: 6px 12px;
                color: #2c5b8f;
                font-size: 12px;
                font-weight: 700;
            }
            #schoolNameCn {
                font-family: "STKaiti", "KaiTi", "Microsoft YaHei";
                font-size: 30px;
                font-weight: 900;
                color: #285283;
                letter-spacing: 2px;
            }
            #schoolNameEn {
                font-size: 14px;
                color: #6d85a1;
                font-weight: 600;
            }
            #projectTitle {
                font-size: 26px;
                font-weight: 900;
                color: #234a7d;
            }
            #projectSubtitle {
                font-size: 15px;
                font-weight: 700;
                color: #557398;
            }
            #projectMinor {
                font-size: 12px;
                color: #7d93ae;
            }
            #systemBadge {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2b5f99, stop:1 #6e9fca);
                border-radius: 18px;
                color: white;
                padding: 10px 18px;
                font-size: 15px;
                font-weight: 800;
            }
            #clockBadge {
                background: rgba(255,255,255,0.76);
                border: 1px solid #d6e4f4;
                border-radius: 14px;
                color: #3d5f89;
                padding: 8px 14px;
                font-size: 13px;
                font-weight: 700;
            }
            #ribbonTitle {
                color: white;
                font-size: 18px;
                font-weight: 900;
            }
            #ribbonTag {
                color: #eaf4ff;
                font-size: 13px;
                font-weight: 600;
            }
            #videoSurface {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0b1830, stop:1 #12284b);
                border: 1px solid #203a63;
                border-radius: 28px;
            }
            #videoHeaderStrip {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0c1c37, stop:1 #1d3964);
                border-top-left-radius: 18px;
                border-top-right-radius: 18px;
                border-bottom-left-radius: 10px;
                border-bottom-right-radius: 10px;
            }
            #videoHeaderText {
                color: #d9e8fa;
                font-size: 13px;
                font-weight: 700;
            }
            #videoMetaTag {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(201,220,242,0.18);
                border-radius: 12px;
                color: #dce9f9;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 700;
            }
            #videoLabel {
                background: transparent;
                color: #dce8f7;
                border: 2px dashed rgba(169, 191, 219, 0.28);
                border-radius: 18px;
                font-size: 20px;
                font-weight: 500;
            }
            #panelStrip {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #215766, stop:1 #41748f);
                border-top-left-radius: 16px;
                border-top-right-radius: 16px;
                border-bottom-left-radius: 8px;
                border-bottom-right-radius: 8px;
            }
            #panelStripText {
                color: #eef8fb;
                font-size: 14px;
                font-weight: 800;
            }
            #panelStripTag {
                color: #dff0f5;
                font-size: 12px;
                font-weight: 700;
            }
            #statusBoardNormal {
                background: #f9fcfa;
                border: 1px solid #d6eadf;
                border-radius: 18px;
            }
            #statusBoardAlert {
                background: #fff7f7;
                border: 1px solid #f1d1d1;
                border-radius: 18px;
            }
            #boardHintNormal {
                font-size: 13px;
                color: #2d6e4f;
                font-weight: 700;
            }
            #boardHintAlert {
                font-size: 13px;
                color: #9e3a3a;
                font-weight: 700;
            }
            #boardStatusNormal {
                font-size: 30px;
                color: #16804f;
                font-weight: 900;
            }
            #boardStatusAlert {
                font-size: 30px;
                color: #cf2f2f;
                font-weight: 900;
            }
            #boardSeparator {
                background: #dceadf;
                max-height: 1px;
                min-height: 1px;
            }
            #boardStatsCard {
                background: rgba(255,255,255,0.72);
                border: 1px solid #e1eaef;
                border-radius: 14px;
            }
            #boardStatsKey {
                font-size: 12px;
                color: #6f88a3;
                font-weight: 700;
            }
            #boardStatsValue {
                font-size: 16px;
                color: #214a78;
                font-weight: 900;
            }
            #boardFooterBar {
                background: rgba(255,255,255,0.70);
                border: 1px solid #dde9ea;
                border-radius: 12px;
                padding: 8px 12px;
            }
            #boardFooterText {
                color: #5e7b93;
                font-size: 12px;
                font-weight: 600;
            }
            #boardFooterTime {
                color: #3a6088;
                font-size: 12px;
                font-weight: 800;
            }
            #infoItem {
                background: #f8fbff;
                border: 1px solid #dde8f3;
                border-radius: 16px;
                padding: 12px 14px;
            }
            #infoText {
                font-size: 14px;
                font-weight: 700;
                color: #294c77;
            }
            QPushButton {
                border: none;
                border-radius: 16px;
                padding: 14px 18px;
                font-size: 16px;
                font-weight: 800;
            }
            QPushButton:hover {
                margin-top: -1px;
            }
            QPushButton:disabled {
                color: rgba(255,255,255,0.88);
            }
            #startButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #39a86b, stop:1 #61c987);
                color: white;
            }
            #startButton:hover { background: #3eaf71; }
            #startButton:disabled { background: #8fd3ab; }
            #stopButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #f06e67, stop:1 #f58d86);
                color: white;
            }
            #stopButton:hover { background: #ec6a63; }
            #stopButton:disabled { background: #f5b1ad; }
            #ghostButton {
                background: #f1f6fb;
                color: #3e608b;
                border: 1px solid #d7e4f2;
                font-size: 15px;
                font-weight: 800;
            }
            #ghostButton:hover {
                background: #eaf2fb;
            }
            #smallTag {
                background: #eef4fb;
                color: #5a7492;
                border-radius: 10px;
                padding: 3px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            #bottomSlogan {
                font-family: "STKaiti", "KaiTi", "Microsoft YaHei";
                font-size: 18px;
                color: #2d507c;
                font-weight: 900;
                letter-spacing: 2px;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #d4dfeb;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #4b7cc3;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: white;
                width: 18px;
                margin: -6px 0;
                border-radius: 9px;
                border: 2px solid #4b7cc3;
            }
            QProgressBar {
                background: #e6edf5;
                border: none;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: #4b7cc3;
                border-radius: 5px;
            }
            QTextEdit {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #101d37, stop:1 #172b4e);
                color: #d9e5f6;
                border: 1px solid #22385c;
                border-radius: 18px;
                padding: 12px;
                font-family: Consolas, Monaco, monospace;
                font-size: 13px;
            }
            """
        )

        root = BackgroundWidget()
        root.setObjectName("rootWidget")
        self.setCentralWidget(root)

        self.root_layout = QVBoxLayout(root)
        self.root_layout.setContentsMargins(16, 14, 16, 14)
        self.root_layout.setSpacing(12)

        self.build_header()
        self.build_body()
        self.apply_window_polish()
        self.setup_clock()
        self.reset_ui_state(initial=True)

    def build_header(self):
        self.header_banner = DecorativeBanner()
        banner_layout = QHBoxLayout(self.header_banner)
        banner_layout.setContentsMargins(22, 14, 22, 14)
        banner_layout.setSpacing(18)

        seal = UniversitySeal()

        school_layout = QVBoxLayout()
        school_layout.setSpacing(1)
        school_name_cn = QLabel("安徽建筑大学")
        school_name_cn.setObjectName("schoolNameCn")
        school_name_en = QLabel("ANHUI JIANZHU UNIVERSITY")
        school_name_en.setObjectName("schoolNameEn")
        school_layout.addWidget(school_name_cn)
        school_layout.addWidget(school_name_en)

        project_layout = QVBoxLayout()
        project_layout.setSpacing(3)
        project_title = QLabel("基于深度学习的行人跌倒检测系统")
        project_title.setObjectName("projectTitle")
        project_subtitle = QLabel("YOLOv8-Pose + LSTM 跌倒行为识别平台")
        project_subtitle.setObjectName("projectSubtitle")
        project_minor = QLabel("实时监控 · 智能判别 · 告警联动 · 可视化展示")
        project_minor.setObjectName("projectMinor")
        project_layout.addWidget(project_title)
        project_layout.addWidget(project_subtitle)
        project_layout.addWidget(project_minor)

        right_layout = QVBoxLayout()
        right_layout.setSpacing(8)
        right_layout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.system_badge = QLabel("系统就绪")
        self.system_badge.setObjectName("systemBadge")
        self.system_badge.setAlignment(Qt.AlignCenter)

        self.clock_badge = QLabel("")
        self.clock_badge.setObjectName("clockBadge")
        self.clock_badge.setAlignment(Qt.AlignCenter)

        project_hint = QLabel("校园安防 / 智慧监护 / 行人跌倒检测")
        project_hint.setObjectName("projectMinor")
        project_hint.setAlignment(Qt.AlignRight)

        right_layout.addWidget(self.system_badge, alignment=Qt.AlignRight)
        right_layout.addWidget(self.clock_badge, alignment=Qt.AlignRight)
        right_layout.addWidget(project_hint)

        banner_layout.addWidget(seal)
        banner_layout.addLayout(school_layout)
        banner_layout.addSpacing(10)
        banner_layout.addLayout(project_layout, stretch=1)
        banner_layout.addLayout(right_layout)

        self.ribbon_bar = BlueRibbon()
        ribbon_layout = QHBoxLayout(self.ribbon_bar)
        ribbon_layout.setContentsMargins(18, 8, 18, 8)
        ribbon_title = QLabel("智慧监护实时监控页面")
        ribbon_title.setObjectName("ribbonTitle")
        ribbon_tag = QLabel("安徽建筑大学 · 校园安防演示界面")
        ribbon_tag.setObjectName("ribbonTag")
        ribbon_layout.addWidget(ribbon_title)
        ribbon_layout.addStretch()
        ribbon_layout.addWidget(ribbon_tag)

        self.root_layout.addWidget(self.header_banner)
        self.root_layout.addWidget(self.ribbon_bar)

    def build_body(self):
        body_layout = QHBoxLayout()
        body_layout.setSpacing(16)
        self.root_layout.addLayout(body_layout, stretch=1)

        left_card = SectionCard("实时监控画面", "主视图 / 姿态识别 / 跌倒判别")
        body_layout.addWidget(left_card, stretch=8)

        self.video_surface = QFrame()
        self.video_surface.setObjectName("videoSurface")
        video_layout = QVBoxLayout(self.video_surface)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(10)

        video_strip = QFrame()
        video_strip.setObjectName("videoHeaderStrip")
        strip_layout = QHBoxLayout(video_strip)
        strip_layout.setContentsMargins(16, 10, 16, 10)
        strip_label = QLabel("实时监控画面 / Camera View")
        strip_label.setObjectName("videoHeaderText")
        strip_layout.addWidget(strip_label)
        strip_layout.addStretch()
        video_layout.addWidget(video_strip)

        meta_layout = QHBoxLayout()
        meta_layout.setSpacing(10)
        self.video_tag_mode = QLabel("模式：实时监控")
        self.video_tag_mode.setObjectName("videoMetaTag")
        self.video_tag_model = QLabel("模型：YOLOv8-Pose + LSTM")
        self.video_tag_model.setObjectName("videoMetaTag")
        self.video_tag_target = QLabel("目标：person / pose")
        self.video_tag_target.setObjectName("videoMetaTag")
        meta_layout.addWidget(self.video_tag_mode)
        meta_layout.addWidget(self.video_tag_model)
        meta_layout.addWidget(self.video_tag_target)
        meta_layout.addStretch()
        video_layout.addLayout(meta_layout)

        self.video_label = QLabel("等待启动摄像头…")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setMinimumSize(840, 460)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.video_label, 1)

        status_strip = QFrame()
        status_strip.setObjectName("panelStrip")
        status_strip_layout = QHBoxLayout(status_strip)
        status_strip_layout.setContentsMargins(16, 8, 16, 8)
        strip_text = QLabel("监测状态面板")
        strip_text.setObjectName("panelStripText")
        strip_tag = QLabel("状态实时刷新")
        strip_tag.setObjectName("panelStripTag")
        status_strip_layout.addWidget(strip_text)
        status_strip_layout.addStretch()
        status_strip_layout.addWidget(strip_tag)
        video_layout.addWidget(status_strip)

        self.status_board = QFrame()
        self.status_board.setObjectName("statusBoardNormal")
        self.status_board.setFixedHeight(210)
        board_layout = QVBoxLayout(self.status_board)
        board_layout.setContentsMargins(16, 12, 16, 12)
        board_layout.setSpacing(10)

        top_board_layout = QHBoxLayout()
        top_board_layout.setSpacing(10)
        board_title_box = QVBoxLayout()
        board_title_box.setSpacing(2)
        self.status_title = QLabel("当前状态")
        self.status_title.setObjectName("boardHintNormal")
        self.status_value = QLabel("正常监控")
        self.status_value.setObjectName("boardStatusNormal")
        board_title_box.addWidget(self.status_title)
        board_title_box.addWidget(self.status_value)
        top_board_layout.addLayout(board_title_box)
        top_board_layout.addStretch()

        self.board_status_chip = QLabel("系统待命")
        self.board_status_chip.setObjectName("smallTag")
        self.board_status_chip.setMinimumWidth(72)
        self.board_status_chip.setAlignment(Qt.AlignCenter)
        top_board_layout.addWidget(self.board_status_chip)
        board_layout.addLayout(top_board_layout)

        separator = QFrame()
        separator.setObjectName("boardSeparator")
        board_layout.addWidget(separator)

        self.board_prob_card, self.board_prob_value = self.create_board_stat_card("跌倒概率", "0.00")
        self.board_threshold_card, self.board_threshold_value = self.create_board_stat_card("当前阈值", "0.75")
        self.board_alert_card, self.board_alert_value = self.create_board_stat_card("告警联动", "未触发")
        self.board_mode_card, self.board_mode_value = self.create_board_stat_card("识别模式", "实时监控")

        stat_row_1 = QHBoxLayout()
        stat_row_1.setSpacing(10)
        stat_row_1.addWidget(self.board_prob_card)
        stat_row_1.addWidget(self.board_threshold_card)

        stat_row_2 = QHBoxLayout()
        stat_row_2.setSpacing(10)
        stat_row_2.addWidget(self.board_alert_card)
        stat_row_2.addWidget(self.board_mode_card)

        board_layout.addLayout(stat_row_1)
        board_layout.addLayout(stat_row_2)

        self.status_progress = QProgressBar()
        self.status_progress.setRange(0, 100)
        self.status_progress.setValue(0)
        self.status_progress.setTextVisible(False)
        self.status_progress.setFixedHeight(10)
        board_layout.addWidget(self.status_progress)

        footer_bar = QFrame()
        footer_bar.setObjectName("boardFooterBar")
        footer_layout = QHBoxLayout(footer_bar)
        footer_layout.setContentsMargins(12, 4, 12, 4)
        footer_layout.setSpacing(10)
        self.board_footer = QLabel("状态说明：系统待命，可启动摄像头进入实时监测。")
        self.board_footer.setWordWrap(True)
        self.board_footer.setObjectName("boardFooterText")
        self.board_time_value = QLabel("--:--:--")
        self.board_time_value.setObjectName("boardFooterTime")
        footer_layout.addWidget(self.board_footer, stretch=1)
        footer_layout.addWidget(self.board_time_value)
        board_layout.addWidget(footer_bar)
        video_layout.addWidget(self.status_board)

        left_card.main_layout.addWidget(self.video_surface)

        footer_hint = QLabel("崇德 · 笃学 · 励志 · 创新")
        footer_hint.setObjectName("bottomSlogan")
        footer_hint.setAlignment(Qt.AlignCenter)
        left_card.main_layout.addWidget(footer_hint)

        right_column = QVBoxLayout()
        right_column.setSpacing(14)
        body_layout.addLayout(right_column, stretch=3)

        control_card = SectionCard("系统控制", "启动 / 停止 / 灵敏度调整")
        control_card.setMinimumWidth(360)
        control_card.setMaximumHeight(250)
        right_column.addWidget(control_card, 0)

        self.btn_start = QPushButton("⚙  启动系统")
        self.btn_start.setMinimumHeight(44)
        self.btn_start.setObjectName("startButton")
        self.btn_start.clicked.connect(self.start_detection)

        self.btn_stop = QPushButton("✦  停止检测")
        self.btn_stop.setMinimumHeight(44)
        self.btn_stop.setObjectName("stopButton")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)

        slider_top = QHBoxLayout()
        self.slider_title = QLabel("检测灵敏度：0.75")
        self.slider_title.setStyleSheet("font-size: 15px; font-weight: 800; color: #36577f;")
        self.slider_tip = QLabel("建议 0.70 - 0.85")
        self.slider_tip.setObjectName("smallTag")
        slider_top.addWidget(self.slider_title)
        slider_top.addStretch()
        slider_top.addWidget(self.slider_tip)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 95)
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.change_sensitivity)

        self.sensitivity_progress = QProgressBar()
        self.sensitivity_progress.setRange(50, 95)
        self.sensitivity_progress.setValue(75)
        self.sensitivity_progress.setTextVisible(False)
        self.sensitivity_progress.setFixedHeight(10)

        self.btn_clear_log = QPushButton("清空日志")
        self.btn_clear_log.setMinimumHeight(40)
        self.btn_clear_log.setObjectName("ghostButton")
        self.btn_clear_log.clicked.connect(self.clear_log)

        control_card.main_layout.addWidget(self.btn_start)
        control_card.main_layout.addWidget(self.btn_stop)
        control_card.main_layout.addSpacing(6)
        control_card.main_layout.addLayout(slider_top)
        control_card.main_layout.addWidget(self.slider)
        control_card.main_layout.addWidget(self.sensitivity_progress)
        control_card.main_layout.addWidget(self.btn_clear_log)

        info_card = SectionCard("运行信息", "设备状态 / 模型状态 / 告警状态")
        info_card.setMinimumWidth(360)
        info_card.setMaximumHeight(220)
        right_column.addWidget(info_card, 0)

        self.camera_label = self.create_info_item("摄像头：未启动")
        self.model_label = self.create_info_item("模型状态：待加载")
        self.alert_label = self.create_info_item("告警状态：未触发")

        info_card.main_layout.addWidget(self.camera_label)
        info_card.main_layout.addWidget(self.model_label)
        info_card.main_layout.addWidget(self.alert_label)

        log_card = SectionCard("检测日志", "事件记录 / 阈值变化 / 告警追踪")
        log_card.setMinimumWidth(360)
        log_card.set_header_tag("滚动更新")
        right_column.addWidget(log_card, 1)

        self.log_text = QTextEdit()
        self.log_text.setMinimumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText("系统日志将在这里显示…")
        log_card.main_layout.addWidget(self.log_text)

    def create_board_stat_card(self, key: str, value: str):
        card = QFrame()
        card.setObjectName("boardStatsCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(3)
        key_label = QLabel(key)
        key_label.setObjectName("boardStatsKey")
        value_label = QLabel(value)
        value_label.setObjectName("boardStatsValue")
        layout.addWidget(key_label)
        layout.addWidget(value_label)
        return card, value_label

    def apply_shadow(self, widget, blur=26, y=8, alpha=28):
        shadow = QGraphicsDropShadowEffect(widget)
        shadow.setBlurRadius(blur)
        shadow.setOffset(0, y)
        shadow.setColor(QColor(38, 74, 117, alpha))
        widget.setGraphicsEffect(shadow)

    def apply_window_polish(self):
        self.apply_shadow(self.header_banner, blur=30, y=10, alpha=30)
        self.apply_shadow(self.ribbon_bar, blur=24, y=8, alpha=26)
        self.apply_shadow(self.video_surface, blur=28, y=10, alpha=28)
        self.apply_shadow(self.status_board, blur=18, y=6, alpha=20)

    def setup_clock(self):
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def update_clock(self):
        now = QDateTime.currentDateTime()
        self.clock_badge.setText(now.toString("yyyy-MM-dd  hh:mm:ss"))
        if hasattr(self, "board_time_value"):
            self.board_time_value.setText(now.toString("hh:mm:ss"))

    def create_info_item(self, text: str) -> QFrame:
        box = QFrame()
        box.setObjectName("infoItem")
        layout = QHBoxLayout(box)
        layout.setContentsMargins(14, 0, 14, 0)
        box.setMinimumHeight(48)
        label = QLabel(text)
        label.setObjectName("infoText")
        layout.addWidget(label)
        box.text_label = label
        return box

    def set_info_item_text(self, box: QFrame, text: str):
        box.text_label.setText(text)

    def append_log(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{now}] {text}")

    def start_detection(self):
        if self.thread is not None:
            return

        self.stopping_manually = False
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.system_badge.setText("系统运行中")
        self.video_tag_mode.setText("模式：系统启动中")
        self.set_info_item_text(self.camera_label, "摄像头：正在启动")
        self.set_info_item_text(self.model_label, "模型状态：正在加载")
        self.board_footer.setText("状态说明：系统启动中，正在连接设备并加载模型。")
        self.board_status_chip.setText("启动中")
        self.append_log("—— 系统启动 ——")

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.update_log)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.update_runtime_signal.connect(self.update_runtime_info)
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.sensitivity = self.slider.value() / 100.0
        self.thread.start()

    def stop_detection(self):
        if self.thread:
            self.stopping_manually = True
            self.thread.stop()
            self.thread = None

        self.reset_ui_state(initial=False)
        self.append_log("—— 系统停止 ——")

    def on_thread_finished(self):
        if self.thread is not None:
            try:
                self.thread.deleteLater()
            except Exception:
                pass
            self.thread = None

        if not self.stopping_manually:
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if "加载失败" in self.model_label.text_label.text() or "启动失败" in self.camera_label.text_label.text():
                self.system_badge.setText("启动异常")
                self.status_title.setText("当前状态")
                self.status_value.setText("等待修复")
                self.board_status_chip.setText("异常")
                self.board_footer.setText("状态说明：请检查模型文件与摄像头设备后重新启动。")
                self.status_progress.setValue(0)

        self.stopping_manually = False

    def reset_board_style(self, alert=False):
        self.status_board.setObjectName("statusBoardAlert" if alert else "statusBoardNormal")
        self.status_title.setObjectName("boardHintAlert" if alert else "boardHintNormal")
        self.status_value.setObjectName("boardStatusAlert" if alert else "boardStatusNormal")
        self.status_board.setStyleSheet("")
        self.status_title.setStyleSheet("")
        self.status_value.setStyleSheet("")
        self.style().unpolish(self.status_board)
        self.style().polish(self.status_board)
        self.style().unpolish(self.status_title)
        self.style().polish(self.status_title)
        self.style().unpolish(self.status_value)
        self.style().polish(self.status_value)
        self.status_board.update()
        self.status_title.update()
        self.status_value.update()

    def reset_ui_state(self, initial: bool = False):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.system_badge.setText("系统就绪" if initial else "系统已停止")
        self.video_tag_mode.setText("模式：实时监控")
        self.set_info_item_text(self.camera_label, "摄像头：未启动")
        self.set_info_item_text(self.model_label, "模型状态：待加载")
        self.set_info_item_text(self.alert_label, "告警状态：未触发")

        self.video_label.clear()
        self.video_label.setText("等待启动摄像头…" if initial else "监控已停止")
        self.video_label.setAlignment(Qt.AlignCenter)

        self.reset_board_style(alert=False)
        self.status_title.setText("当前状态")
        self.status_value.setText("待机监测" if initial else "系统就绪")
        self.board_status_chip.setText("系统待命" if initial else "已停止")
        self.board_prob_value.setText("0.00")
        self.board_mode_value.setText("待机模式" if initial else "实时监控")
        self.board_alert_value.setText("未触发")
        self.board_threshold_value.setText(f"{self.slider.value() / 100:.2f}")
        self.status_progress.setValue(0)
        self.board_footer.setText(
            "状态说明：系统待命，可启动摄像头进入实时监测。"
            if initial else "状态说明：系统已停止，点击启动按钮可重新进入监测。"
        )

        if initial:
            self.log_text.clear()
            self.log_text.append("—— 系统就绪，等待启动 ——")

    def change_sensitivity(self):
        value = self.slider.value()
        self.slider_title.setText(f"检测灵敏度：{value / 100:.2f}")
        self.sensitivity_progress.setValue(value)
        self.board_threshold_value.setText(f"{value / 100:.2f}")
        if self.thread:
            self.thread.update_sensitivity(value)
        self.append_log(f"灵敏度已调整为 {value / 100:.2f}")

    def clear_log(self):
        self.log_text.clear()
        self.log_text.append("—— 日志已清空 ——")

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def update_log(self, text):
        self.append_log(text)

    def update_runtime_info(self, camera_text, model_text, alert_text):
        self.set_info_item_text(self.camera_label, camera_text)
        self.set_info_item_text(self.model_label, model_text)
        self.set_info_item_text(self.alert_label, alert_text)
        self.board_alert_value.setText(alert_text.split("：")[-1])

    def update_status(self, status, prob):
        self.board_prob_value.setText(f"{prob:.2f}")
        self.status_progress.setValue(max(0, min(100, int(prob * 100))))

        if status == "FALL":
            self.system_badge.setText("告警触发中")
            self.video_tag_mode.setText("模式：告警联动")
            self.board_mode_value.setText("告警联动")
            self.board_alert_value.setText("已触发")
            self.board_status_chip.setText("高风险")
            self.reset_board_style(alert=True)
            self.status_title.setText("告警状态")
            self.status_value.setText("检测到跌倒")
            self.board_footer.setText("状态说明：系统已达到跌倒确认阈值，请立即查看现场情况。")
        elif status == "ERROR":
            self.system_badge.setText("启动异常")
            self.video_tag_mode.setText("模式：异常待处理")
            self.board_mode_value.setText("待处理")
            self.board_alert_value.setText("未触发")
            self.board_status_chip.setText("异常")
            self.reset_board_style(alert=True)
            self.status_title.setText("当前状态")
            self.status_value.setText("启动失败")
            self.board_footer.setText("状态说明：设备或模型加载异常，请检查运行环境。")
        else:
            self.system_badge.setText("系统运行中" if self.thread else "系统就绪")
            self.video_tag_mode.setText("模式：实时监控")
            self.board_mode_value.setText("实时监控")
            self.board_alert_value.setText("未触发")
            self.board_status_chip.setText("监控中")
            self.reset_board_style(alert=False)
            self.status_title.setText("当前状态")
            self.status_value.setText("正常监控")
            self.board_footer.setText("状态说明：系统正在持续分析当前画面中的人体姿态序列。")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled = qt_format.scaled(
            self.video_label.width() - 8,
            self.video_label.height() - 8,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        return QPixmap.fromImage(scaled)

    def closeEvent(self, event):
        self.stop_detection()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
