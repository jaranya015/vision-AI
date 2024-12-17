import cv2
import time
import os
from datetime import datetime

# โหลดโมเดล SSD + MobileNet สำหรับ Face Detection

# modelFile = r"C:\Users\TUF\Downloads\computer vision\Real Time Face Detection and Counting\res10_300x300_ssd_iter_140000.caffemodel"
# configFile = r"C:\Users\TUF\Downloads\computer vision\Real Time Face Detection and Counting\deploy.prototxt"

modelFile = os.path.abspath("res10_300x300_ssd_iter_140000.caffemodel")
configFile = os.path.abspath("deploy.prototxt")

net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# เปิดกล้องเว็บแคม
cap = cv2.VideoCapture(0)

# ตรวจสอบว่าเปิดกล้องสำเร็จหรือไม่
if not cap.isOpened():
    print("Cannot open camera. Exiting...")
    exit()

# ตั้งค่าสำหรับการบันทึกวิดีโอ
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_filename = "output_video.avi"
fps = 20.0
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

print("Press 'q' to quit the program.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Cannot capture video. Exiting...")
        break

    # เริ่มจับเวลาเพื่อวัด FPS
    start_time = time.time()

    # ปรับขนาดภาพให้เข้ากับโมเดล
    (h, w) = frame.shape[:2]
    resized_frame = cv2.resize(frame, (300, 300))

    # แปลงภาพเป็น Blob เพื่อส่งเข้าโมเดล
    blob = cv2.dnn.blobFromImage(resized_frame, scalefactor=1.0, size=(300, 300),
                                 mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # ส่งข้อมูลเข้าโมเดล
    net.setInput(blob)
    detections = net.forward()

    # วนลูปผ่านผลลัพธ์การตรวจจับใบหน้า
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]  # ความมั่นใจในการตรวจจับ
        if confidence > 0.5:  # กรองเฉพาะที่ความมั่นใจสูงกว่า 50%
            face_count += 1
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x1, y1, x2, y2) = box.astype("int")

            # วาดกรอบใบหน้า
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # แสดง Timestamp บนภาพ
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 2)

    # แสดงจำนวนใบหน้าที่ตรวจพบ
    cv2.putText(frame, f"Faces Detected: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # คำนวณ FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # แสดงภาพแบบเรียลไทม์
    cv2.imshow("Real-Time Face Detection with SSD + MobileNet", frame)

    # บันทึกวิดีโอ
    out.write(frame)

    # ตรวจสอบการกดปุ่ม 'q' เพื่อออกจากโปรแกรม
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting program...")
        break

# ปิดกล้องและการบันทึกวิดีโอ
cap.release()
out.release()
cv2.destroyAllWindows()
