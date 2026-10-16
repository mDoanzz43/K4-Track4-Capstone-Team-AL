# Failed cases

## F01 — Chữ có độ tương phản quá thấp

- Input: `results/failed_cases/F01_low_contrast_input.png`.
- Expected theo góc nhìn người dùng: trích được chữ A mờ.
- Actual: pipeline trả `No ink found in the image`.
- Nguyên nhân: mực không tối hơn nền cục bộ đủ mức cần thiết và bị bộ lọc chống
  nhiễu loại bỏ.
- Cách giảm lỗi: chiếu sáng đều, dùng mực tối hơn, tăng contrast ảnh hoặc nhập bằng
  canvas.
- Giới hạn kết luận: đây là ảnh tổng hợp cố định để tái lập, chưa đại diện cho mọi
  camera và điều kiện ánh sáng.

## V01 — Ảnh trắng (expected rejection)

- Input: `results/failed_cases/V01_blank_expected_rejection.png`.
- Expected và actual: hệ thống từ chối do không có nét.
- Đây là kiểm tra validation, **không trình bày như lỗi mô hình**.

## F02 — Cử chỉ webcam khi tay bị che/ngược sáng

- Expected: chụm ngón cái–ngón trỏ làm bút hạ ổn định.
- Known behavior: keypoint có thể nhảy hoặc mất; bút nhấc/hạ sai.
- Nguyên nhân dự kiến: landmark 2D không ổn định khi ngón tay bị che hoặc ảnh thiếu
  sáng. Ngưỡng chụm là heuristic 0,32/0,48.
- Mitigation khi demo: ánh sáng phía trước, để trọn bàn tay trong khung, dùng phím
  `Space`, canvas hoặc video dự phòng.
- **Evidence còn thiếu:** chưa có benchmark người thật được đồng ý tham gia. Không
  đưa ra tỷ lệ chính xác cho ca này nếu chưa thu thập theo protocol hợp lệ.

