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
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSlider, QProgressBar,
    QFrame, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont, QPainter, QPen, QLinearGradient
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime


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
        return self.fc(out)


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
        self.model_loaded = False
        self.load_error = ''

        try:
            self.yolo_model = YOLO('yolov8n-pose.pt')
            self.lstm_model = FallLSTM().to(self.device)
            self.lstm_model.load_state_dict(torch.load('fall_detection_model.pth', map_location=self.device))
            self.lstm_model.eval()
            self.model_loaded = True
        except Exception as e:
            self.load_error = str(e)

    def run(self):
        if not self.model_loaded:
            self.update_log_signal.emit('❌ 模型加载失败，请检查 yolov8n-pose.pt 和 fall_detection_model.pth 文件。')
            if self.load_error:
                self.update_log_signal.emit(f'📌 详细原因：{self.load_error}')
            self.update_runtime_signal.emit('摄像头：未启动', '模型状态：加载失败', '告警状态：未触发')
            self.update_status_signal.emit('ERROR', 0.0)
            return

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            self.update_log_signal.emit('❌ 摄像头启动失败，请检查设备是否被占用。')
            self.update_runtime_signal.emit('摄像头：启动失败', '模型状态：加载成功', '告警状态：未触发')
            self.update_status_signal.emit('ERROR', 0.0)
            return

        sequence = deque(maxlen=30)
        fall_counter = 0
        trigger_frame = 3

        self.update_log_signal.emit('🎥 摄像头已启动，系统进入实时监测。')
        self.update_runtime_signal.emit('摄像头：已启动', '模型状态：加载成功', '告警状态：未触发')

        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                self.update_log_signal.emit('⚠️ 无法读取摄像头画面。')
                break

            results = self.yolo_model(frame, classes=0, verbose=False, conf=0.5)
            annotated_frame = results[0].plot()

            current_status = 'MONITORING'
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
                        current_status = 'FALL'
                        self.update_runtime_signal.emit('摄像头：已启动', '模型状态：加载成功', '告警状态：已触发')
                        if fall_counter == trigger_frame:
                            now = datetime.now().strftime('%H:%M:%S')
                            self.update_log_signal.emit(f'⚠️ [{now}] 检测到疑似跌倒，已达到确认阈值。')
                            self.update_log_signal.emit('📨 [模拟] 已发送提醒信息至监护端，请及时查看。')
                            if winsound is not None:
                                try:
                                    winsound.Beep(2000, 500)
                                except Exception:
                                    pass
                    else:
                        self.update_runtime_signal.emit('摄像头：已启动', '模型状态：加载成功', '告警状态：未触发')
            else:
                self.update_runtime_signal.emit('摄像头：已启动', '模型状态：加载成功', '告警状态：未触发')

            self.change_pixmap_signal.emit(annotated_frame)
            self.update_status_signal.emit(current_status, prob)

        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

    def update_sensitivity(self, value):
        self.sensitivity = value / 100.0


class SimpleSeal(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(88, 88)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#edf4ff'))
        painter.drawEllipse(1, 1, 86, 86)
        painter.setBrush(QColor('#2b5f99'))
        painter.drawEllipse(7, 7, 74, 74)
        painter.setBrush(QColor('white'))
        painter.drawEllipse(15, 15, 58, 58)
        painter.setBrush(QColor('#2b5f99'))
        painter.drawEllipse(23, 23, 42, 42)
        painter.setPen(QPen(QColor('white'), 2))
        painter.drawLine(44, 28, 44, 60)
        painter.drawLine(28, 44, 60, 44)
        painter.drawEllipse(36, 36, 16, 16)
        painter.end()


class Card(QFrame):
    def __init__(self, title='', subtitle=''):
        super().__init__()
        self.setObjectName('card')
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(18, 16, 18, 16)
        self.layout_main.setSpacing(10)

        if title:
            header = QVBoxLayout()
            header.setSpacing(2)
            lab1 = QLabel(title)
            lab1.setObjectName('cardTitle')
            header.addWidget(lab1)
            if subtitle:
                lab2 = QLabel(subtitle)
                lab2.setObjectName('cardSubTitle')
                header.addWidget(lab2)
            self.layout_main.addLayout(header)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(22)
        shadow.setOffset(0, 8)
        shadow.setColor(QColor(40, 78, 126, 28))
        self.setGraphicsEffect(shadow)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.thread = None
        self.setWindowTitle('行人跌倒检测系统')
        self.resize(1400, 860)
        self.setMinimumSize(1280, 760)

        self.setStyleSheet('''
            QMainWindow { background: #edf3fb; }
            QWidget {
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
                color: #173a63;
                font-size: 14px;
            }
            #root { background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #f8fbff, stop:1 #edf3fb); }
            #card {
                background: rgba(255,255,255,0.96);
                border: 1px solid #dbe6f3;
                border-radius: 22px;
            }
            #cardTitle { font-size: 18px; font-weight: 800; color: #183a63; }
            #cardSubTitle { font-size: 12px; color: #6e88a5; }
            #topTitleCn {
                font-size: 24px; font-weight: 900; color: #224d82;
            }
            #topTitleEn {
                font-size: 13px; font-weight: 600; color: #6d87a4;
            }
            #projectTitle {
                font-size: 20px; font-weight: 900; color: #1f4979;
            }
            #projectSub {
                font-size: 14px; font-weight: 700; color: #567395;
            }
            #projectMinor { font-size: 12px; color: #7d95ae; }
            #statusBadge {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #2e619a, stop:1 #6f9fca);
                color: white; border-radius: 16px; padding: 10px 18px;
                font-size: 14px; font-weight: 800;
            }
            #clockBadge {
                background: #ffffff; border: 1px solid #d9e6f3; border-radius: 14px;
                color: #365a84; padding: 8px 14px; font-size: 13px; font-weight: 700;
            }
            #ribbon {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #1f4f83, stop:1 #78a7d1);
                border-radius: 16px;
            }
            #ribbonText { color: white; font-size: 15px; font-weight: 800; }
            #videoWrap {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:1, stop:0 #081a35, stop:1 #122a4c);
                border: 1px solid #203e6a; border-radius: 24px;
            }
            #videoHeader {
                background: #1c3a67; border-radius: 14px; color: #eef5ff;
                padding: 10px 14px; font-size: 13px; font-weight: 700;
            }
            #tag {
                background: rgba(255,255,255,0.10);
                border: 1px solid rgba(205,223,245,0.20);
                border-radius: 12px; color: #e5eef9;
                padding: 7px 12px; font-size: 12px; font-weight: 700;
            }
            #videoLabel {
                background: rgba(11,31,60,0.55);
                border: 2px dashed rgba(173,194,220,0.25);
                border-radius: 18px; color: #dbe8f8;
                font-size: 18px; font-weight: 600;
            }
            #panelTitle {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #245767, stop:1 #41748f);
                color: white; border-radius: 12px; padding: 8px 14px;
                font-size: 13px; font-weight: 800;
            }
            #statusPanelNormal {
                background: #fbfdfa; border: 1px solid #d7eadf; border-radius: 18px;
            }
            #statusPanelAlert {
                background: #fff8f8; border: 1px solid #f3d4d4; border-radius: 18px;
            }
            #statusMain { font-size: 24px; font-weight: 900; color: #0a7a57; }
            #statusMainAlert { font-size: 24px; font-weight: 900; color: #cc3535; }
            #statusHint { font-size: 13px; color: #3f765f; font-weight: 700; }
            #statusHintAlert { font-size: 13px; color: #aa4a4a; font-weight: 700; }
            #chip {
                background: #eef4fb; border: 1px solid #d8e4f3; border-radius: 12px;
                color: #395d86; padding: 6px 12px; font-size: 12px; font-weight: 800;
            }
            #statBox {
                background: #ffffff; border: 1px solid #dce7f4; border-radius: 14px;
            }
            #statKey { font-size: 12px; color: #6f87a4; font-weight: 700; }
            #statVal { font-size: 15px; color: #1f4978; font-weight: 800; }
            #infoBox {
                background: #f8fbff; border: 1px solid #dbe6f4; border-radius: 14px;
                padding: 10px 14px; font-size: 14px; font-weight: 700; color: #29507f;
            }
            QPushButton {
                border: none; border-radius: 14px; min-height: 48px;
                font-size: 16px; font-weight: 800;
            }
            #startBtn { background: #44b56d; color: white; }
            #startBtn:hover { background: #39a660; }
            #startBtn:disabled { background: #9fdab3; }
            #stopBtn { background: #ef8e88; color: white; }
            #stopBtn:hover { background: #e47d77; }
            #stopBtn:disabled { background: #f4beb9; }
            #ghostBtn {
                background: #eef4fb; color: #365d8d; border: 1px solid #d7e4f3; min-height: 40px;
            }
            QSlider::groove:horizontal {
                height: 8px; border-radius: 4px; background: #d7e1ec;
            }
            QSlider::handle:horizontal {
                background: #4f7ec5; width: 18px; margin: -6px 0; border-radius: 9px;
            }
            QProgressBar {
                background: #e7eef6; border: none; border-radius: 5px;
                min-height: 10px;
            }
            QProgressBar::chunk { background: #4f7ec5; border-radius: 5px; }
            QTextEdit {
                background: #10284f; color: #eff6ff; border: none; border-radius: 16px;
                padding: 12px; font-family: Consolas, "Microsoft YaHei";
                font-size: 13px;
            }
            #bottomText { font-size: 13px; color: #345a88; font-weight: 700; }
        ''')

        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)

        self.root_layout = QVBoxLayout(root)
        self.root_layout.setContentsMargins(14, 14, 14, 14)
        self.root_layout.setSpacing(12)

        self.build_header()
        self.build_body()
        self.setup_clock()
        self.reset_ui_idle()

    def build_header(self):
        top = Card()
        top.layout_main.setContentsMargins(18, 14, 18, 14)
        top_row = QHBoxLayout()
        top_row.setSpacing(16)

        seal = SimpleSeal()
        top_row.addWidget(seal, 0, Qt.AlignTop)

        school_col = QVBoxLayout()
        school_col.setSpacing(4)
        cn = QLabel('安徽建筑大学')
        cn.setObjectName('topTitleCn')
        en = QLabel('ANHUI JIANZHU UNIVERSITY')
        en.setObjectName('topTitleEn')
        school_col.addStretch()
        school_col.addWidget(cn)
        school_col.addWidget(en)
        school_col.addStretch()
        top_row.addLayout(school_col)

        proj_col = QVBoxLayout()
        proj_col.setSpacing(4)
        t1 = QLabel('基于深度学习的行人跌倒检测系统')
        t1.setObjectName('projectTitle')
        t2 = QLabel('YOLOv8-Pose + LSTM 跌倒行为识别平台')
        t2.setObjectName('projectSub')
        t3 = QLabel('实时监控 · 智能判别 · 告警联动 · 可视化展示')
        t3.setObjectName('projectMinor')
        proj_col.addStretch()
        proj_col.addWidget(t1)
        proj_col.addWidget(t2)
        proj_col.addWidget(t3)
        proj_col.addStretch()
        top_row.addLayout(proj_col, 1)

        right_col = QVBoxLayout()
        right_col.setSpacing(8)
        self.header_status = QLabel('系统就绪')
        self.header_status.setObjectName('statusBadge')
        self.clock_label = QLabel('')
        self.clock_label.setObjectName('clockBadge')
        small = QLabel('校园安防 / 智慧监护 / 行人跌倒检测')
        small.setObjectName('projectMinor')
        small.setAlignment(Qt.AlignRight)
        right_col.addWidget(self.header_status, 0, Qt.AlignRight)
        right_col.addWidget(self.clock_label, 0, Qt.AlignRight)
        right_col.addWidget(small)
        top_row.addLayout(right_col)

        top.layout_main.addLayout(top_row)
        self.root_layout.addWidget(top)

        ribbon = QFrame()
        ribbon.setObjectName('ribbon')
        ribbon_layout = QHBoxLayout(ribbon)
        ribbon_layout.setContentsMargins(16, 8, 16, 8)
        rt = QLabel('智慧监护实时监控页面')
        rt.setObjectName('ribbonText')
        rb = QLabel('安徽建筑大学 · 校园安防演示界面')
        rb.setObjectName('ribbonText')
        rb.setStyleSheet('color: rgba(255,255,255,0.92); font-size: 13px; font-weight: 600;')
        ribbon_layout.addWidget(rt)
        ribbon_layout.addStretch()
        ribbon_layout.addWidget(rb)
        self.root_layout.addWidget(ribbon)

    def build_body(self):
        body = QHBoxLayout()
        body.setSpacing(14)
        self.root_layout.addLayout(body, 1)

        left = Card('实时监控画面', '主视图 / 姿态识别 / 跌倒判别')
        body.addWidget(left, 11)

        video_wrap = QFrame()
        video_wrap.setObjectName('videoWrap')
        video_layout = QVBoxLayout(video_wrap)
        video_layout.setContentsMargins(14, 14, 14, 14)
        video_layout.setSpacing(10)

        header = QLabel('实时监控画面 / Camera View')
        header.setObjectName('videoHeader')
        video_layout.addWidget(header)

        tags = QHBoxLayout()
        tags.setSpacing(10)
        self.tag_mode = QLabel('模式：实时监控')
        self.tag_mode.setObjectName('tag')
        self.tag_model = QLabel('模型：YOLOv8-Pose + LSTM')
        self.tag_model.setObjectName('tag')
        self.tag_target = QLabel('目标：person / pose')
        self.tag_target.setObjectName('tag')
        tags.addWidget(self.tag_mode)
        tags.addWidget(self.tag_model)
        tags.addWidget(self.tag_target)
        tags.addStretch()
        video_layout.addLayout(tags)

        self.video_label = QLabel('等待启动摄像头')
        self.video_label.setObjectName('videoLabel')
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(420)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label, 1)

        panel_title = QLabel('监测状态面板')
        panel_title.setObjectName('panelTitle')
        video_layout.addWidget(panel_title)

        self.status_panel = QFrame()
        self.status_panel.setObjectName('statusPanelNormal')
        panel_layout = QVBoxLayout(self.status_panel)
        panel_layout.setContentsMargins(14, 12, 14, 12)
        panel_layout.setSpacing(10)

        head_row = QHBoxLayout()
        head_col = QVBoxLayout()
        head_col.setSpacing(2)
        self.status_hint = QLabel('当前状态')
        self.status_hint.setObjectName('statusHint')
        self.status_value = QLabel('系统待命')
        self.status_value.setObjectName('statusMain')
        head_col.addWidget(self.status_hint)
        head_col.addWidget(self.status_value)
        head_row.addLayout(head_col)
        head_row.addStretch()
        self.status_chip = QLabel('系统待命')
        self.status_chip.setObjectName('chip')
        head_row.addWidget(self.status_chip)
        panel_layout.addLayout(head_row)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        self.stat_prob = self.make_stat('跌倒概率', '0.00')
        self.stat_threshold = self.make_stat('当前阈值', '0.75')
        self.stat_alert = self.make_stat('告警联动', '未触发')
        self.stat_mode = self.make_stat('识别模式', '实时监控')
        grid.addWidget(self.stat_prob['box'], 0, 0)
        grid.addWidget(self.stat_threshold['box'], 0, 1)
        grid.addWidget(self.stat_alert['box'], 1, 0)
        grid.addWidget(self.stat_mode['box'], 1, 1)
        panel_layout.addLayout(grid)

        self.prob_progress = QProgressBar()
        self.prob_progress.setRange(0, 100)
        self.prob_progress.setValue(0)
        self.prob_progress.setTextVisible(False)
        panel_layout.addWidget(self.prob_progress)

        footer = QHBoxLayout()
        self.status_desc = QLabel('状态说明：系统待命，可启动摄像头进入实时监测。')
        self.status_desc.setObjectName('bottomText')
        self.status_desc.setWordWrap(True)
        self.panel_time = QLabel('--:--:--')
        self.panel_time.setObjectName('chip')
        footer.addWidget(self.status_desc, 1)
        footer.addWidget(self.panel_time, 0, Qt.AlignRight)
        panel_layout.addLayout(footer)

        video_layout.addWidget(self.status_panel)
        left.layout_main.addWidget(video_wrap, 1)

        slogan = QLabel('崇德 · 笃学 · 励志 · 创新')
        slogan.setObjectName('bottomText')
        slogan.setAlignment(Qt.AlignCenter)
        left.layout_main.addWidget(slogan)

        right_col = QVBoxLayout()
        right_col.setSpacing(14)
        body.addLayout(right_col, 4)

        control = Card('系统控制', '启动 / 停止 / 灵敏度调整')
        right_col.addWidget(control)

        self.btn_start = QPushButton('启动系统')
        self.btn_start.setObjectName('startBtn')
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_stop = QPushButton('停止检测')
        self.btn_stop.setObjectName('stopBtn')
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_detection)
        control.layout_main.addWidget(self.btn_start)
        control.layout_main.addWidget(self.btn_stop)

        slider_row = QHBoxLayout()
        self.slider_label = QLabel('检测灵敏度：0.75')
        self.slider_label.setStyleSheet('font-size: 15px; font-weight: 800; color: #365b84;')
        self.slider_tip = QLabel('建议 0.70 - 0.85')
        self.slider_tip.setObjectName('chip')
        slider_row.addWidget(self.slider_label)
        slider_row.addStretch()
        slider_row.addWidget(self.slider_tip)
        control.layout_main.addLayout(slider_row)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(50, 95)
        self.slider.setValue(75)
        self.slider.valueChanged.connect(self.change_sensitivity)
        self.slider_progress = QProgressBar()
        self.slider_progress.setRange(50, 95)
        self.slider_progress.setValue(75)
        self.slider_progress.setTextVisible(False)
        self.clear_btn = QPushButton('清空日志')
        self.clear_btn.setObjectName('ghostBtn')
        self.clear_btn.clicked.connect(self.clear_log)

        control.layout_main.addWidget(self.slider)
        control.layout_main.addWidget(self.slider_progress)
        control.layout_main.addWidget(self.clear_btn)

        info = Card('运行信息', '设备状态 / 模型状态 / 告警状态')
        right_col.addWidget(info)
        self.camera_label = QLabel('摄像头：未启动')
        self.camera_label.setObjectName('infoBox')
        self.model_label = QLabel('模型状态：待加载')
        self.model_label.setObjectName('infoBox')
        self.alert_label = QLabel('告警状态：未触发')
        self.alert_label.setObjectName('infoBox')
        info.layout_main.addWidget(self.camera_label)
        info.layout_main.addWidget(self.model_label)
        info.layout_main.addWidget(self.alert_label)

        log = Card('检测日志', '事件记录 / 阈值变化 / 告警追踪')
        right_col.addWidget(log, 1)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(170)
        self.log_text.setPlaceholderText('系统日志将在这里显示…')
        log.layout_main.addWidget(self.log_text, 1)

    def make_stat(self, key, value):
        box = QFrame()
        box.setObjectName('statBox')
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(3)
        k = QLabel(key)
        k.setObjectName('statKey')
        v = QLabel(value)
        v.setObjectName('statVal')
        lay.addWidget(k)
        lay.addWidget(v)
        return {'box': box, 'key': k, 'value': v}

    def setup_clock(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_clock)
        self.timer.start(1000)
        self.update_clock()

    def update_clock(self):
        now = QDateTime.currentDateTime()
        text_full = now.toString('yyyy-MM-dd  hh:mm:ss')
        text_short = now.toString('hh:mm:ss')
        self.clock_label.setText(text_full)
        self.panel_time.setText(text_short)

    def append_log(self, text):
        now = datetime.now().strftime('%H:%M:%S')
        self.log_text.append(f'[{now}] {text}')

    def reset_ui_idle(self):
        self.header_status.setText('系统就绪')
        self.video_label.setText('等待启动摄像头')
        self.status_panel.setObjectName('statusPanelNormal')
        self.status_panel.style().unpolish(self.status_panel)
        self.status_panel.style().polish(self.status_panel)
        self.status_hint.setText('当前状态')
        self.status_hint.setObjectName('statusHint')
        self.status_value.setText('系统待命')
        self.status_value.setObjectName('statusMain')
        self.status_chip.setText('系统待命')
        self.status_desc.setText('状态说明：系统待命，可启动摄像头进入实时监测。')
        self.stat_prob['value'].setText('0.00')
        self.stat_threshold['value'].setText(f'{self.slider.value()/100:.2f}')
        self.stat_alert['value'].setText('未触发')
        self.stat_mode['value'].setText('实时监控')
        self.prob_progress.setValue(0)
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)
        self.status_hint.style().unpolish(self.status_hint)
        self.status_hint.style().polish(self.status_hint)

    def start_detection(self):
        if self.thread is not None:
            return
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.header_status.setText('系统运行中')
        self.camera_label.setText('摄像头：正在启动')
        self.model_label.setText('模型状态：正在加载')
        self.append_log('—— 系统启动 ——')

        self.thread = DetectionThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.update_log_signal.connect(self.append_log)
        self.thread.update_status_signal.connect(self.update_status)
        self.thread.update_runtime_signal.connect(self.update_runtime_info)
        self.thread.sensitivity = self.slider.value() / 100.0
        self.thread.finished.connect(self.on_thread_finished)
        self.thread.start()

    def stop_detection(self):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_label.setText('摄像头：未启动')
        self.model_label.setText('模型状态：待加载')
        self.alert_label.setText('告警状态：未触发')
        self.append_log('—— 系统停止 ——')
        self.reset_ui_idle()

    def on_thread_finished(self):
        if self.thread is not None:
            self.thread = None
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def clear_log(self):
        self.log_text.clear()
        self.append_log('—— 日志已清空 ——')

    def change_sensitivity(self):
        value = self.slider.value()
        display = value / 100.0
        self.slider_label.setText(f'检测灵敏度：{display:.2f}')
        self.slider_progress.setValue(value)
        self.stat_threshold['value'].setText(f'{display:.2f}')
        if self.thread is not None:
            self.thread.update_sensitivity(value)
        self.append_log(f'🔧 灵敏度调整为：{display:.2f}')

    def update_runtime_info(self, camera_text, model_text, alert_text):
        self.camera_label.setText(camera_text)
        self.model_label.setText(model_text)
        self.alert_label.setText(alert_text)

    def update_status(self, status, prob):
        self.stat_prob['value'].setText(f'{prob:.2f}')
        self.prob_progress.setValue(int(max(0.0, min(1.0, prob)) * 100))

        if status == 'FALL':
            self.header_status.setText('告警处理中')
            self.status_panel.setObjectName('statusPanelAlert')
            self.status_hint.setObjectName('statusHintAlert')
            self.status_value.setObjectName('statusMainAlert')
            self.status_value.setText('检测到跌倒')
            self.status_chip.setText('告警触发')
            self.status_desc.setText('状态说明：检测到跌倒风险，系统已触发模拟告警。')
            self.stat_alert['value'].setText('已触发')
        elif status == 'ERROR':
            self.header_status.setText('系统异常')
            self.status_panel.setObjectName('statusPanelAlert')
            self.status_hint.setObjectName('statusHintAlert')
            self.status_value.setObjectName('statusMainAlert')
            self.status_value.setText('运行异常')
            self.status_chip.setText('异常状态')
            self.status_desc.setText('状态说明：请检查模型文件或摄像头连接。')
            self.stat_alert['value'].setText('未触发')
        else:
            self.header_status.setText('系统运行中' if self.thread is not None else '系统就绪')
            self.status_panel.setObjectName('statusPanelNormal')
            self.status_hint.setObjectName('statusHint')
            self.status_value.setObjectName('statusMain')
            self.status_value.setText('正常监控')
            self.status_chip.setText('实时监控')
            self.status_desc.setText('状态说明：系统运行正常，正在持续分析人体姿态序列。')
            self.stat_alert['value'].setText('未触发')

        self.status_panel.style().unpolish(self.status_panel)
        self.status_panel.style().polish(self.status_panel)
        self.status_hint.style().unpolish(self.status_hint)
        self.status_hint.style().polish(self.status_hint)
        self.status_value.style().unpolish(self.status_value)
        self.status_value.style().polish(self.status_value)

    def update_image(self, cv_img):
        qt_img = self.convert_cv_qt(cv_img)
        self.video_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        available_w = max(640, self.video_label.width() - 12)
        available_h = max(360, self.video_label.height() - 12)
        scaled = image.scaled(available_w, available_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        return QPixmap.fromImage(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.video_label.pixmap() is None:
            return

    def closeEvent(self, event):
        if self.thread is not None:
            self.thread.stop()
            self.thread = None
        event.accept()


if __name__ == '__main__':
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont('Microsoft YaHei', 10))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
