import sys
import cv2
import numpy as np  
import math

from openCv_.sort_tracker.tracker import Sort
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QRadioButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QWidget, QGroupBox,
    QLineEdit, QButtonGroup, QTextEdit, QComboBox
)

class ABRAPanel(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ABRA - Hava Savunma Sistemi Paneli")
        self.setGeometry(100, 100, 1280, 720)
        self.setStyleSheet("background-color: #111; color: white;")
       

        self.tracker_red = Sort()
        self.tracker_blue = Sort()
        self.kernel = np.ones((3, 3), np.uint8)

       # HSV renk aralıkları  ortama göre güncellenir.
        self.lower_red = np.array([165, 60, 130], dtype=np.uint8)
        self.upper_red = np.array([180, 255, 255], dtype=np.uint8)
        self.lower_blue = np.array([90,  60, 90], dtype=np.uint8)
        self.upper_blue = np.array([105, 255, 255], dtype=np.uint8)



        self.cap = cv2.VideoCapture(0)  #kamer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_camera_view)
        self.timer.start(30)
        self.current_view_mode = "original"
        self.initUI()



    def update_camera_view(self):
         ret, frame = self.cap.read()
         if not ret:
             return

         frame = cv2.resize(frame, (640, 480))
         hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 🔴 KIRMIZI maske
         mask_red = cv2.inRange(hsv, self.lower_red, self.upper_red)
         mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, self.kernel)
         mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, self.kernel)
         mask_red = cv2.medianBlur(mask_red, 5)

    # 🔵 MAVİ maske
         mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
         mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, self.kernel)
         mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, self.kernel)
         mask_blue = cv2.medianBlur(mask_blue, 5)

    # 🔴 KIRMIZI balonlar
         detections_red = []
         contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         for cnt in contours_red:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                  continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            if 200 < area < 10000 and circularity > 0.65:
                 x, y, w, h = cv2.boundingRect(cnt)
                 detections_red.append([x, y, x + w, y + h, 1.0])
 
    # 🔵 MAVİ balonlar
         detections_blue = []
         contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
         for cnt in contours_blue:
             area = cv2.contourArea(cnt)
             perimeter = cv2.arcLength(cnt, True)
             if perimeter == 0:
                continue
             circularity = 4 * math.pi * area / (perimeter * perimeter)
             x, y, w, h = cv2.boundingRect(cnt)
             aspect_ratio = w / h
             if 200 < area < 10000 and circularity > 0.85 and 0.8 < aspect_ratio<1.2:  ##BURAYI DA ŞEKLE GÖRE GÜNCELLE ESNET
                  detections_blue.append([x, y, x + w, y + h, 1.0])

    # Takip sistemleri (SORT)
         tracks_red = self.tracker_red.update(np.array(detections_red))
         tracks_blue = self.tracker_blue.update(np.array(detections_blue))

    #  KIRMIZI kutuları çiz
         for track in tracks_red:
            x1, y1, x2, y2, track_id = map(int, track)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Id: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, "DUSMAN", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    #  MAVİ kutuları çiz
         for track in tracks_blue:
          x1, y1, x2, y2, track_id = map(int, track)
          cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Açık mavi kutu
          cv2.putText(frame, f"DOST ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, (255, 255, 0), 2)


      # QLabel'de göstermek için BGR → RGB dönüşüm
         rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
         h, w, ch = rgb_image.shape
         bytes_per_line = ch * w
         qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
         pixmap = QPixmap.fromImage(qt_image)

         

         self.camera_label.setPixmap(pixmap)
         




    def set_view_mode(self, mode):
        self.current_view_mode = mode
        self.append_log(f"Görüntü modu değiştirildi: {mode.upper()}")

    def append_log(self, message):
        self.log_panel.append(f"[LOG] {message}")

    

    def initUI(self):
        main_layout = QVBoxLayout()
        top_grid = QGridLayout()
        body_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        title = QLabel("ABRA - HAVA SAVUNMA SİSTEMİ PANELİ")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; letter-spacing: 2px;")
        title.setFixedHeight(36)

        logo_label = QLabel()
        logo_pixmap = QPixmap("ikonn/abra.png")
        if not logo_pixmap.isNull():
            logo_pixmap = logo_pixmap.scaled(130, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(logo_pixmap)
        else:
            logo_label.setText("LOGO YOK")
            logo_label.setStyleSheet("color:#bbb;")
        logo_label.setAlignment(Qt.AlignRight | Qt.AlignTop)

        spacer_left = QWidget()
        top_grid.addWidget(spacer_left, 0, 0)
        top_grid.addWidget(title, 0, 1)
        top_grid.addWidget(logo_label, 0, 2, alignment=Qt.AlignRight)
        top_grid.setColumnStretch(0, 1)
        top_grid.setColumnStretch(1, 4)
        top_grid.setColumnStretch(2, 1)

        control_row = QHBoxLayout()
        self.original_btn = QPushButton("Original Frame")
        self.mask_btn = QPushButton("Mask Frame")
        for btn in (self.original_btn, self.mask_btn):
            btn.setStyleSheet("border: 1px solid white; padding: 6px;")
        self.original_btn.clicked.connect(lambda: self.set_view_mode("original"))
        self.mask_btn.clicked.connect(lambda: self.set_view_mode("mask"))

        mod_label = QLabel("MOD")
        mod_label.setFont(QFont("Arial", 11, QFont.Bold))
        self.mod_otonom = QRadioButton("OTONOM")
        self.mod_manual = QRadioButton("MANUEL")
        self.mod_otonom.setChecked(True)
        control_row.addWidget(self.original_btn)
        control_row.addWidget(self.mask_btn)
        control_row.addSpacing(30)
        control_row.addWidget(mod_label)
        control_row.addWidget(self.mod_otonom)
        control_row.addWidget(self.mod_manual)
        control_row.addStretch()

        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: #222; border: none;")
        self.camera_label.setScaledContents(True)

        button_row = QHBoxLayout()
        actions = {
            "ARM": "Sistem aktif hâle getirildi.",
            "FIRE": "Ateş edildi!",
            "DISARM": "Sistem pasif duruma alındı."
        }
        for label, log_message in actions.items():
            btn = QPushButton(label)
            btn.setStyleSheet("QPushButton {border: 1px solid white; background-color: black; color: white; padding: 6px;} QPushButton:hover {background-color: #333;}")
            btn.clicked.connect(lambda _, msg=log_message: self.append_log(msg))
            button_row.addWidget(btn)

        arrows = QGridLayout()
        up = QPushButton();    up.setIcon(QIcon("ikonn/up.png"));    up.setIconSize(QSize(55, 55))
        down = QPushButton();  down.setIcon(QIcon("ikonn/down.png"));down.setIconSize(QSize(55, 55))
        leftb = QPushButton(); leftb.setIcon(QIcon("ikonn/left.png")); leftb.setIconSize(QSize(55, 55))
        rightb= QPushButton(); rightb.setIcon(QIcon("ikonn/right.png"));rightb.setIconSize(QSize(55, 55))
        center = QPushButton("ABRA")
        center.setStyleSheet("background-color: black; color: white; font-weight: bold; border: 1px solid white;")

        up.clicked.connect(lambda: self.append_log("Yukarı yönlendirildi."))
        down.clicked.connect(lambda: self.append_log("Aşağı yönlendirildi."))
        leftb.clicked.connect(lambda: self.append_log("Sola yönlendirildi."))
        rightb.clicked.connect(lambda: self.append_log("Sağa yönlendirildi."))
        center.clicked.connect(lambda: self.append_log("ABRA konumlandı."))

        arrows.addWidget(up,    0, 1)
        arrows.addWidget(leftb, 1, 0)
        arrows.addWidget(center,1, 1)
        arrows.addWidget(rightb,1, 2)
        arrows.addWidget(down,  2, 1)

        self.log_panel = QTextEdit()
        self.log_panel.setReadOnly(True)
        self.log_panel.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #00ffcc;
                font-family: Consolas;
                font-size: 12px;
                border: 1px solid #444;
            }
        """)
        log_box = QVBoxLayout()
        log_label = QLabel("LOG PANELİ")
        log_label.setStyleSheet("color: white; font-weight: bold;")
        log_box.addWidget(log_label)
        log_box.addWidget(self.log_panel)
        self.append_log("Sistem başlatıldı.")

        left_panel.addLayout(control_row)
        left_panel.addSpacing(5)
        left_panel.addWidget(self.camera_label)
        left_panel.addSpacing(10)
        left_panel.addLayout(button_row)
        left_panel.addSpacing(10)
        left_panel.addLayout(arrows)

        stage_group = QGroupBox("STAGE")
        stage_layout = QVBoxLayout()
        self.stage_buttons = QButtonGroup()
        for i in range(1, 4):
            rb = QRadioButton(f"STAGE-{i}")
            rb.setStyleSheet("color: white;")
            self.stage_buttons.addButton(rb)
            stage_layout.addWidget(rb)
        stage_group.setLayout(stage_layout)

        engage_group = QGroupBox("ENGAGEMENT PANEL")
        engage_layout = QVBoxLayout()

        self.combo_color = QComboBox()
        self.combo_color.addItems(["Kırmızı", "Mavi", "Yeşil"])
        self.combo_color.setStyleSheet("background-color: black; color: white;")

        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Daire", "Kare", "Üçgen"])
        self.combo_shape.setStyleSheet("background-color: black; color: white;")

        self.engage_input = QLineEdit()
        self.engage_input.setPlaceholderText("Engaged Target ID")
        self.engage_input.setStyleSheet("background-color: #1e1e1e; color: white; border: 1px solid white;")

        engage_btn = QPushButton("ENGAGE")
        disengage_btn = QPushButton("DISENGAGE")
        for b in (engage_btn, disengage_btn):
            b.setStyleSheet("QPushButton {background-color: black; color: white; border: 1px solid white;} QPushButton:hover {background-color: #333;}")

        engage_btn.clicked.connect(self.handle_engage)

        engage_layout.addWidget(self.combo_color)
        engage_layout.addWidget(self.combo_shape)
        engage_layout.addWidget(self.engage_input)
        engage_layout.addWidget(engage_btn)
        engage_layout.addWidget(disengage_btn)
        engage_group.setLayout(engage_layout)

        right_panel.addSpacing(50)
        right_panel.addWidget(stage_group)
        right_panel.addSpacing(20)
        right_panel.addWidget(engage_group)
        right_panel.addSpacing(20)
        right_panel.addLayout(log_box)
        right_panel.addStretch()

        body_layout.addLayout(left_panel, 2)
        body_layout.addSpacing(20)
        body_layout.addLayout(right_panel, 1)

        main_layout.addLayout(top_grid)
        main_layout.addSpacing(4)
        main_layout.addLayout(body_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def handle_engage(self):
        renk = self.combo_color.currentText()
        sekil = self.combo_shape.currentText()
        target_id = self.engage_input.text().strip()

        if target_id:
            self.append_log(f"Engaged Target ID: {target_id} | Renk: {renk}, Şekil: {sekil}")
        else:
            self.append_log(f"Engage denemesi yapıldı. ID girilmedi. Renk: {renk}, Şekil: {sekil}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ABRAPanel()
    window.show()
    sys.exit(app.exec_())
