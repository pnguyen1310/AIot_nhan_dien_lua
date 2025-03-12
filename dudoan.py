import cv2
import torch
import numpy as np
import threading
import time
import os

# 1️⃣ Tải mô hình YOLOv5
model_path = "best.pt"  # Thay đường dẫn nếu cần
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, verbose=False)

# 2️⃣ Địa chỉ IP Camera
ip_camera_url = "http://192.168.1.9:8080/video"

# 3️⃣ Lớp đọc video chạy song song
class VideoStream:
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(self.url)
        self.grabbed, self.frame = self.cap.read()
        self.running = True
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while self.running:
            grabbed, frame = self.cap.read()
            if grabbed:
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

# 4️⃣ Khởi động luồng video
video_stream = VideoStream(ip_camera_url)

# 5️⃣ Tạo thư mục lưu ảnh nếu chưa có
save_dir = "fire_images"
os.makedirs(save_dir, exist_ok=True)

# 6️⃣ Vòng lặp chính
skip_frames = 2  # Bỏ qua 2 frame để giảm lag
frame_count = 0

while True:
    frame = video_stream.read()
    if frame is None:
        print("❌ Không thể nhận dữ liệu từ camera. Kiểm tra URL.")
        break

    frame_count += 1
    if frame_count % skip_frames == 0:
        # Resize ảnh
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Chạy dự đoán với YOLOv5
        results = model(resized_frame)
        fire_detected = False

        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.3:  # Ngưỡng tin cậy
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(resized_frame, f"Fire: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                fire_detected = True

        # Lưu ảnh nếu phát hiện lửa (có bounding box)
        if fire_detected:
            timestamp = int(time.time())
            filename = os.path.join(save_dir, f"fire_detected_{timestamp}.jpg")
            cv2.imwrite(filename, resized_frame)  # Lưu ảnh với bounding box
            print(f"🔥 Ảnh chụp đã lưu: {filename}")

        # Hiển thị video
        cv2.imshow("🔥 Fire Detection - YOLOv5", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Dừng chương trình
video_stream.stop()
cv2.destroyAllWindows()
