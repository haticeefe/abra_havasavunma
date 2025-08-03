import cv2
import numpy as np

h_values = []
s_values = []
v_values = []

def mouse_callback(event, x, y, flags, param):
    global hsv, h_values, s_values, v_values
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y, x]
        h, s, v = pixel
        h_values.append(h)
        s_values.append(s)
        v_values.append(v)
        print(f"HSV: {pixel}")

        # HSV min/max otomatik hesaplama
        h_min, h_max = min(h_values), max(h_values)
        s_min, s_max = min(s_values), max(s_values)
        v_min, v_max = min(v_values), max(v_values)
        print(f"\n🎯 Güncel HSV Aralıkları:")
        print(f"Lower Bound: [{h_min}, {s_min}, {v_min}]")
        print(f"Upper Bound: [{h_max}, {s_max}, {v_max}]\n")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Kamera", frame)
    cv2.setMouseCallback("Kamera", mouse_callback)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
