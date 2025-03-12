# Hướng Dẫn Train và Chạy Mô Hình Phát Hiện Cháy

## 1. Cài Đặt Môi Trường

Trước khi chạy model, bạn cần cài đặt các thư viện cần thiết. Sử dụng lệnh sau để cài đặt:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python numpy pygame ultralytics
```

## 2. Huấn Luyện Mô Hình

Tập tin `fire-train.ipynb` chứa code huấn luyện mô hình YOLOv5. Để train model, bạn cần thực hiện các bước sau:

1. Mở file `fire-train.ipynb` bằng Jupyter Notebook hoặc Google Colab.
2. Chạy từng cell theo thứ tự để:
   - Tải dữ liệu và tiền xử lý.
   - Cấu hình và train mô hình.
   - Lưu mô hình đã train (`best.pt`).
3. Sau khi train xong, file `best.pt` sẽ được lưu lại để sử dụng trong dự đoán.

## 3. Chạy Dự Đoán

Bạn có thể sử dụng một trong hai file để chạy dự đoán:

### Cách 1: Chạy `final.py` (Có kết nối Arduino và Cảnh báo âm thanh)

```bash
python final.py
```

- Tập tin này sẽ:
  - Kết nối với Arduino qua Serial để đọc dữ liệu từ cảm biến MQ-2.
  - Lấy video từ IP Camera để phát hiện lửa.
  - Phát âm thanh cảnh báo khi phát hiện cháy.
  - Lưu ảnh có lửa vào thư mục `fire_images`.

**Lưu ý:** Cần thay đổi địa chỉ IP Camera trong `final.py` nếu cần.

### Cách 2: Chạy `dudoan.py` (Chỉ nhận diện lửa, không cần Arduino)

```bash
python dudoan.py
```

- File này chỉ sử dụng camera để phát hiện lửa, không có cảnh báo âm thanh hoặc giao tiếp với Arduino.
- Ảnh có lửa sẽ được lưu vào thư mục `fire_images`.
- Bấm phím `q` để dừng chương trình.

## 4. Các Lưu Ý Quan Trọng

- Đảm bảo `best.pt` có trong thư mục chạy.
- Nếu dùng Arduino, cần kiểm tra cổng Serial (`COMx` trên Windows hoặc `/dev/ttyUSBx` trên Linux).
- Kiểm tra kết nối IP Camera trước khi chạy dự đoán.
- Nếu có lỗi về `torch.hub.load`, thử chạy `pip install ultralytics --upgrade`.
