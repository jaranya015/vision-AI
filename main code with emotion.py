import cv2
import time
import os
from datetime import datetime
from fer import FER
import tkinter as tk
from tkinter import Label
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

# ปิดการใช้งาน GPU
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow is using CPU.")

# โหลดโมเดล SSD + MobileNet สำหรับ Face Detection
modelFile = os.path.abspath("res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.abspath("deploy.prototxt")
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # ใช้ DEFAULT สำหรับ CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # ตั้งเป้าหมายเป็น CPU

# ตั้งค่า Emotion Detection
emotion_detector = FER(mtcnn=True)

# ตัวแปรเก็บข้อมูล
current_camera_index = 0
total_faces_detected = 0
unique_faces = set()
emotions_data = {}
faces_per_minute = {}  # เก็บจำนวนใบหน้าต่อนาที

# Dashboard
def update_dashboard():
    global total_faces_detected, emotions_data
    total_faces_label.config(text=f"Total Faces Detected: {total_faces_detected}")
    emotions_summary = "\n".join([f"{emotion}: {count}" for emotion, count in emotions_data.items()])
    emotions_label.config(text=f"Emotions Detected:\n{emotions_summary}")
    update_graph()

def update_graph():
    ax.clear()
    ax.bar(faces_per_minute.keys(), faces_per_minute.values(), color='blue')
    ax.set_title("Faces Detected Per Minute")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Number of Faces")
    graph_canvas.draw()

def create_dashboard():
    global total_faces_label, emotions_label, graph_canvas, ax
    root = tk.Tk()
    root.title("Face Detection Dashboard")
    total_faces_label = Label(root, text="Total Faces Detected: 0", font=("Arial", 16))
    total_faces_label.pack()
    emotions_label = Label(root, text="Emotions Detected:\nNone", font=("Arial", 14))
    emotions_label.pack()

    # กราฟจำนวนคนต่อนาที
    fig, ax = plt.subplots(figsize=(5, 3))
    graph_canvas = FigureCanvasTkAgg(fig, root)
    graph_canvas.get_tk_widget().pack()

    root.mainloop()

# สลับกล้อง
def switch_camera():
    global current_camera_index, cap
    current_camera_index = 1 - current_camera_index
    cap.release()
    cap = cv2.VideoCapture(current_camera_index)
    if not cap.isOpened():
        print(f"Camera with index {current_camera_index} is not available. Switching back.")
        current_camera_index = 1 - current_camera_index
        cap = cv2.VideoCapture(current_camera_index)

# เปิดกล้อง
cap = cv2.VideoCapture(current_camera_index)
if not cap.isOpened():
    print(f"Cannot open camera with index {current_camera_index}. Exiting...")
    exit()

# เริ่ม Dashboard
dashboard_thread = threading.Thread(target=create_dashboard, daemon=True)
dashboard_thread.start()

print("Press 'q' to quit the program.")
print("Press 'c' to switch cameras.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture video. Exiting...")
        break

    start_time = time.time()
    current_minute = datetime.now().strftime("%H:%M")

    # ตรวจจับใบหน้า
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # ติดตามใบหน้า
            unique_faces.add((x1, y1, x2, y2))

            # ตรวจจับอารมณ์
            cropped_face = frame[y1:y2, x1:x2]
            emotion, score = emotion_detector.top_emotion(cropped_face)
            if emotion:
                emotions_data[emotion] = emotions_data.get(emotion, 0) + 1
                cv2.putText(frame, f"{emotion}: {score:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # วาดกรอบใบหน้า
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # อัปเดตจำนวนใบหน้า
    total_faces_detected += face_count
    faces_per_minute[current_minute] = faces_per_minute.get(current_minute, 0) + face_count
    update_dashboard()

    # แสดงข้อมูล
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Face Detection with Emotion Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        switch_camera()

# ปิดโปรแกรม
cap.release()
cv2.destroyAllWindows()
