import torch
import cv2
import numpy as np
import threading
import serial
import time
import os
import pygame  # Dùng pygame để phát âm thanh

# Khởi tạo pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("Alarm.wav")  # Thay bằng file âm thanh của bạn

# Kết nối Arduino qua Serial (THAY COM9 bằng cổng thực tế của bạn)
try:
    ser = serial.Serial('COM9', 9600, timeout=1)
    print("✅ Kết nối Arduino thành công!")
except serial.SerialException:
    print("❌ Không thể kết nối Arduino! Kiểm tra cổng COM.")
    ser = None

# Tải mô hình YOLOv5 đã huấn luyện
model_path = "best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Địa chỉ IP Camera
IP_CAM_URL = "http://192.168.1.9:8080/video"

# Tạo thư mục lưu ảnh nếu chưa có
save_dir = "fire_images"
os.makedirs(save_dir, exist_ok=True)

# Biến trạng thái
fire_detected = False  # Cờ báo cháy
mq2_triggered = False  # Cờ phát hiện khói/gas
alarm_playing = False  # Theo dõi trạng thái âm thanh
frame = None  # Lưu frame mới nhất
frame_lock = threading.Lock()  # Biến khóa luồng để tránh lỗi đọc frame

# Hàm phát âm thanh cảnh báo
def play_alarm():
    global alarm_playing
    if not pygame.mixer.get_busy():  # Kiểm tra nếu chưa có âm thanh nào đang phát
        alarm_playing = True
        alarm_sound.play()

# Hàm đọc tín hiệu từ cảm biến MQ-2
def read_mq2():
    global mq2_triggered
    while True:
        try:
            if ser and ser.in_waiting > 0:
                mq2_data = ser.readline().decode().strip()
                if mq2_data == "GAS_DETECTED":
                    mq2_triggered = True
                    print("🔥 MQ-2 phát hiện khói/gas! Bắt đầu nhận diện lửa...")
                else:
                    mq2_triggered = False
        except Exception as e:
            print(f"❌ Lỗi đọc MQ-2: {e}")
        time.sleep(0.1)  # Giảm tải CPU

# Chạy luồng đọc MQ-2
mq2_thread = threading.Thread(target=read_mq2, daemon=True)
mq2_thread.start()

# Hàm đọc video từ IP Camera (chạy trong luồng riêng)
def read_camera():
    global frame
    cap = cv2.VideoCapture(IP_CAM_URL)
    
    # Nếu không mở được camera, dừng chương trình
    if not cap.isOpened():
        print("❌ Không thể kết nối IP Camera! Kiểm tra địa chỉ.")
        return
    
    while True:
        ret, new_frame = cap.read()
        if ret:
            with frame_lock:  # Đảm bảo đọc/ghi frame an toàn giữa các luồng
                frame = cv2.resize(new_frame, (640, 480))  # Giảm kích thước ảnh để tăng tốc xử lý
        else:
            print("❌ Lỗi đọc camera! Kiểm tra kết nối.")
            break
        time.sleep(0.03)  # Tăng tốc xử lý

    cap.release()

# Chạy luồng đọc video
camera_thread = threading.Thread(target=read_camera, daemon=True)
camera_thread.start()

# Vòng lặp xử lý
while True:
    try:
        if not mq2_triggered:
            time.sleep(0.1)  # Tiết kiệm CPU khi MQ-2 chưa kích hoạt
            continue

        # Đảm bảo frame hợp lệ
        with frame_lock:
            if frame is None:
                time.sleep(0.05)
                continue
            frame_copy = frame.copy()  # Sao chép để YOLO xử lý không bị lỗi

        # Chạy mô hình YOLO để nhận diện lửa
        results = model(frame_copy)

        fire_found = False  # Mặc định không có lửa

        # Kiểm tra các bounding box từ YOLO
        for *xyxy, conf, cls in results.xyxy[0]:
            if conf > 0.3:  # Ngưỡng tin cậy
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_copy, f"Fire: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                fire_found = True  # Xác nhận có lửa

        # Nếu phát hiện lửa lần đầu tiên, phát cảnh báo và chụp ảnh
        if fire_found and not fire_detected:
            fire_detected = True
            if not alarm_playing:  # Chỉ phát âm thanh nếu chưa phát
                play_alarm()
            
            # Chụp ảnh và lưu
            timestamp = int(time.time())
            filename = os.path.join(save_dir, f"fire_detected_{timestamp}.jpg")
            cv2.imwrite(filename, frame_copy)  # Lưu ảnh với bounding box
            print(f"📸 Ảnh chụp đã lưu: {filename}")

        # Nếu không còn lửa, chỉ reset biến `fire_detected`, không reset `alarm_playing`
        elif not fire_found:
            fire_detected = False

        # Kiểm tra nếu âm thanh đã kết thúc, reset `alarm_playing`
        if not pygame.mixer.get_busy():
            alarm_playing = False

        # Hiển thị video
        cv2.imshow("Fire Detection", frame_copy)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        break

cv2.destroyAllWindows()
if ser:
    ser.close()
