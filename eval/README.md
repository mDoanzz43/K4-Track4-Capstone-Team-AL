# Evaluation and demo evidence

Thư mục này gom bằng chứng có thể tái lập cho buổi demo. Không dùng dữ liệu thật
của bệnh nhân hoặc sinh viên.

## Chạy lại

```powershell
.\.venv\Scripts\python.exe eval\run_eval.py
.\.venv\Scripts\python.exe -m pytest -q -rs tests
```

Kết quả benchmark được ghi vào `results/metrics.json` và `results/REPORT.md`.
Ảnh trong `results/vision_pass/` thể hiện lần lượt input, ảnh nhị phân và nét được
trích. `results/failed_cases/` chứa input tái lập cho ca lỗi.

## Phạm vi metric

- **Ink coverage:** tỷ lệ pixel mực nằm trong khoảng nửa độ dày nét + 3 pixel so
  với đường tâm đã trích. Metric này đo khả năng giữ hình dạng, không phải OCR.
- **Vision latency:** thời gian xử lý một ảnh tổng hợp trên máy chạy eval; không bao
  gồm webcam, mạng hoặc thời gian vẽ robot.
- **Tracking RMS/max:** khoảng cách trên mặt giấy giữa đầu bút thực tế trong
  PyBullet và target khi bút hạ. Đây không phải độ chính xác robot vật lý.

Không được suy rộng các số liệu này thành “độ chính xác AI” trên người dùng thật.
Webcam gesture accuracy vẫn cần một protocol và người tham gia có đồng ý riêng.

## Artifact

- `video_demo.mkv`: video dự phòng.
- `results/REPORT.md`: bảng kết quả tóm tắt.
- `results/metrics.json`: số liệu từng case, metadata môi trường và commit.
- `results/test-results.txt`: output pytest gần nhất, gồm cả lý do test bị skip.
- `results/vision_pass/*.png`: bằng chứng input → binary → extracted strokes.
- `results/failed_cases/*`: đầu vào làm lỗi xuất hiện lại.
- `failed_cases.md`: expected/actual/cause/mitigation và giới hạn sử dụng.
