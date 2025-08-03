import cv2
import numpy as np
import math
from openCv_.sort_tracker.tracker import Sort

# Kamera başlat
cap = cv2.VideoCapture(0)

# Sort takip sistemleri
tracker_red = Sort()
tracker_blue = Sort()

# 🔴 Kırmızı HSV aralığı (GENİŞLETİLMİŞ ve TAM KAPSAYICI)
lower_red = np.array([165, 60, 130], dtype=np.uint8)
upper_red = np.array([180, 255, 255], dtype=np.uint8)

# 🔵 Mavi HSV aralığı
lower_blue = np.array([85, 30, 80], dtype=np.uint8)
upper_blue = np.array([115, 255, 255], dtype=np.uint8)

# Gürültü temizliği için kernel
kernel = np.ones((3, 3), np.uint8)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- MASKE OLUŞTURMA ---
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- MASKE İYİLEŞTİRME ---
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_red = cv2.medianBlur(mask_red, 5)

    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_DILATE, kernel, iterations=1)
    mask_blue = cv2.medianBlur(mask_blue, 5)

    # --- DÜŞMAN BALON TESPİTİ ---
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

    # --- DOST BALON TESPİTİ ---
    detections_blue = []
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if 200 < area < 10000 and circularity > 0.65:
            x, y, w, h = cv2.boundingRect(cnt)
            detections_blue.append([x, y, x + w, y + h, 1.0])

    # --- TAKİP GÜNCELLE ---
    tracks_red = tracker_red.update(np.array(detections_red))
    tracks_blue = tracker_blue.update(np.array(detections_blue))

    # --- DÜŞMAN GÖSTERİM ---
    for track in tracks_red:
        x1, y1, x2, y2, track_id = map(int, track)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"Id: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, "dusman", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        print(f"dusman orta noktası ➤ ({cx}, {cy}) [ID={track_id}]")

    # --- DOST GÖSTERİM ---
    for track in tracks_blue:
        x1, y1, x2, y2, track_id = map(int, track)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(frame, f"Id: {track_id}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, "dost", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        print(f"dost orta noktası ➤ ({cx}, {cy}) [ID={track_id}]")

    # --- GÖSTERİM ---
    cv2.imshow("Balon Takibi - DOST & DUSMAN", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- TEMİZLİK ---
cap.release()
cv2.destroyAllWindows()
