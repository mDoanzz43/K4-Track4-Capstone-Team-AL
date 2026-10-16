# K4-Track4-Capstone-Team-AL — Robot Vẽ Chữ

OpenCV → quỹ đạo → cánh tay KUKA iiwa vẽ lại ký tự trong PyBullet.

Web app: người dùng **vẽ ký tự trên canvas** hoặc **viết trên không trung bằng tay trước webcam**.
Ở chế độ webcam, hệ thống nhận diện keypoint bàn tay; chụm ngón cái + ngón trỏ để hạ bút.
OpenCV tách ký tự và trích nét, hệ thống sinh quỹ đạo bút, rồi cánh tay **KUKA LBR iiwa 7 bậc tự do**
vẽ lại ký tự trong mô phỏng **PyBullet**. Khung hình mô phỏng được stream trực tiếp lên trình duyệt.
Chỉ cần CPU, không cần GPU.

## Cài đặt & chạy

Yêu cầu: Python 3.10+ (đã thử với 3.11), Linux/macOS/Windows.

```bash
git clone https://github.com/mDoanzz43/K4-Track4-Capstone-Team-AL.git
cd K4-Track4-Capstone-Team-AL
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # fastapi, uvicorn, numpy, opencv-python, pybullet
./run.sh                            # hoặc: python -m uvicorn backend.app:app --port 8000
```

Mở **http://localhost:8000**. Webcam chỉ hoạt động trên `localhost` hoặc HTTPS, đó là quy định của trình duyệt.
Nếu truy cập từ máy khác thì phải chạy sau HTTPS.
Mô hình nhận diện tay (MediaPipe) đã nằm sẵn trong `frontend/vendor/mediapipe`, nên app chạy được hoàn toàn offline.
Muốn mở cho máy khác trong mạng LAN: `HOST=0.0.0.0 ./run.sh`.

Chạy test:

```bash
python -m pytest -q tests           # 43 test: vision, sketch, trajectory, IK/sim, HTTP + WebSocket, trình duyệt
```

Test trình duyệt (`tests/test_e2e_browser.py`) cần Playwright + Chromium (`pip install playwright && playwright install chromium`).
Nếu không có thì test này tự bỏ qua.

```bash
```

## Cách dùng

1. **Nhập ký tự**
   - *Vẽ trên canvas*: vẽ bằng chuột/cảm ứng; có hoàn tác, xoá, chỉnh độ dày nét.
   - *Vẽ bằng tay (webcam)*: bấm **Bật webcam** rồi giơ một bàn tay lên. Lần đầu sẽ tải mô hình mất khoảng 1 giây.
     **Chụm ngón cái và ngón trỏ** để hạ bút, di chuyển tay để viết, **mở ra** để nhấc bút.
     Con trỏ tròn trên video chuyển sang đỏ khi bút đang hạ.
     Nếu chụm tay khó nhận, giữ phím **Space** để hạ bút. Có nút hoàn tác và xoá.
   - *Tải ảnh*: kéo-thả, bấm để chọn file, hoặc dán ảnh (Ctrl+V). Chọn loại ảnh:
     - **字 Chữ / nét mực**: chữ Hán, thư pháp, chữ viết tay, logo, tranh nét. Mọi nét mực đều được giữ; nền tối chữ sáng cũng được.
     - **👤 Chân dung / ảnh chụp**: ảnh được chuyển thành tranh phác hoạ (đường nét + nét gạch đổ bóng).
       Tuỳ chọn: mức chi tiết, cắt theo khuôn mặt, tách người khỏi nền, đổ bóng bằng nét gạch.
2. Bấm **Trích nét →**: xem ảnh nhị phân, xương nét và quỹ đạo (màu = thứ tự nét, chấm = điểm bắt đầu),
   kèm thời gian robot vẽ ước tính.
3. Bấm **▶ Robot vẽ**: xem robot vẽ; chọn góc camera, tốc độ (0.5×–10×), bật/tắt bóng đổ và quỹ đạo kế hoạch.
   Khung bên phải vẽ lại **mực thực tế** lấy từ vị trí đầu bút mô phỏng. Khi xong sẽ hiện sai số bám quỹ đạo.

## Kiến trúc

```
Trình duyệt (HTML/JS thuần)                 Server Python (FastAPI, 1 tiến trình)
┌──────────────────────────┐  POST /api/extract   ┌─────────────────────────────┐
│ canvas / tay qua webcam  │ ───────────────────▶ │ vision.py (OpenCV)          │
│ MediaPipe Hands (WASM,   │ ◀── strokes + ảnh ── │  nhị phân → tách ký tự →    │
│ CPU) → keypoint → nét    │                      │  skeleton → đồ thị → nét    │
│ <img> khung hình robot   │  WebSocket /ws/draw  ├─────────────────────────────┤
│ canvas mực thực tế       │ ◀── JPEG + JSON ──── │ trajectory.py: quỹ đạo 3D   │
│ nút start/stop/camera    │ ───── lệnh JSON ───▶ │ robot_sim.py: PyBullet      │
└──────────────────────────┘                      │  DIRECT + TinyRenderer (CPU)│
                                                  └─────────────────────────────┘
```

| File | Vai trò |
|---|---|
| `backend/vision.py` | Pipeline OpenCV: giải mã ảnh, nhị phân hoá, tách ký tự, làm mảnh, chuyển xương nét thành đồ thị, nối thành nét bút |
| `backend/trajectory.py` | Ánh xạ nét 2D lên tờ giấy, sinh quỹ đạo đầu bút 240 Hz (hạ/nhấc bút, tốc độ vẽ không đổi) |
| `backend/robot_sim.py` | Cảnh PyBullet (bàn, giấy, iiwa, bút), IK, điều khiển motor, phát hiện mực, render + chồng mực |
| `backend/sketch.py` | Ảnh chụp → tranh phác hoạ: nhận diện mặt (YuNet), tách người (GrabCut), đường nét, nét gạch đổ bóng |
| `backend/models/` | `face_detection_yunet_2023mar.onnx` (230 kB, OpenCV Zoo) cho `cv2.FaceDetectorYN` |
| `backend/app.py` | FastAPI: REST trích nét, WebSocket stream mô phỏng, phục vụ frontend |
| `frontend/air.js` | Vẽ bằng tay: webcam + MediaPipe Hand Landmarker, cử chỉ chụm → hạ/nhấc bút, lọc One-Euro, raster hoá nét |
| `frontend/vendor/mediapipe/` | `@mediapipe/tasks-vision` 0.10.21 (Apache-2.0) + model `hand_landmarker.task`, đóng gói cục bộ |
| `frontend/` | Giao diện một trang (không framework, không bước build) |
| `tests/` | pytest cho từng tầng, end-to-end qua API/WebSocket và qua trình duyệt (Playwright) |

## Quyết định thiết kế chính

**Vẽ bằng tay qua webcam (keypoint)**
- Dùng **MediaPipe Hand Landmarker** (21 keypoint/bàn tay), chạy ngay trong trình duyệt bằng WASM với delegate CPU.
  Không cần gửi video về server và không cần GPU: đo được khoảng 25–35 ms/khung.
  Thư viện (0.10.21, bản ổn định) và model được đóng gói cục bộ, không phụ thuộc CDN.
- Vị trí bút là **trung điểm đầu ngón cái (4) và đầu ngón trỏ (8)**. Bút **hạ khi chụm**:
  khoảng cách 4–8 chia cho cỡ bàn tay (cổ tay 0 → khớp ngón giữa 9) phải nhỏ hơn 0.32, và nhấc khi vượt 0.48.
  Có trễ (hysteresis) và debounce 2 khung để bút không bị nhấp nháy. Cách này không phụ thuộc khoảng cách tay–camera.
  Mất tay khỏi khung hình thì tự nhấc bút.
- Vị trí được lọc bằng **One-Euro filter**: tay đứng yên thì ít rung, tay di chuyển nhanh thì ít trễ.
  Ảnh được lật như gương để người dùng viết tự nhiên.
- Nét viết trên không trung được vẽ thành ảnh trắng–đen, rồi đi qua **cùng pipeline OpenCV** như canvas.
  Nhờ vậy nét run tay được làm mảnh, làm trơn và đơn giản hoá trước khi đưa cho robot.

**Tải ảnh: chữ Hán / nét mực (`mode: "ink"`)**
- Dùng chung pipeline xương nét với canvas, nhưng giữ **mọi** thành phần mực (chữ Hán có nhiều bộ tách rời,
  nhiều chữ một lúc), và làm việc ở độ phân giải cao hơn (320 px thay vì 200 px) để giữ nét nhỏ.
- Tự nhận cực tính: nếu “mực” theo Otsu chiếm quá nửa ảnh thì ảnh bị đảo màu (chữ trắng trên nền đen).

**Tải ảnh: chân dung / ảnh chụp (`mode: "sketch"`, `backend/sketch.py`)**
- **Nhận diện khuôn mặt** bằng YuNet (`cv2.FaceDetectorYN`, chạy CPU vài ms) rồi cắt khung đầu–vai quanh mặt lớn nhất.
  OpenCV 5 đã bỏ Haar cascade nên dùng YuNet.
- **Tách người khỏi nền** bằng GrabCut, khởi tạo từ hộp mặt: elip mặt là tiền cảnh chắc chắn; đầu và thân hình thang
  là “có thể tiền cảnh”; mép trái/phải/trên là nền. Nét và đổ bóng chỉ vẽ bên trong người, thêm đường bao dáng người.
  Nếu không thấy mặt thì vẽ cả ảnh (phù hợp ảnh đồ vật, phong cảnh).
- **Đường nét**: CLAHE → bilateral filter (giữ cạnh) → adaptive threshold (trung bình cục bộ) cho các đường tối mảnh
  (mắt, mày, miệng, tóc, nếp áo). Bỏ đốm nhỏ, rồi đi qua cùng pipeline skeleton → đồ thị → nét.
- **Đổ bóng bằng nét gạch** (hatching): vùng tối nhất 30% của ảnh được gạch 45°, vùng tối nhất 12% gạch chéo thêm −45°.
  Dùng phân vị của chính ảnh, nên tự thích nghi độ sáng. Nét gạch được sinh bằng cách xoay mặt nạ cho hướng gạch nằm ngang,
  quét từng hàng lấy các đoạn liên tục, rồi xoay ngược lại. Các hàng xen kẽ đảo chiều để bút đi zig-zag.
- **Mức chi tiết** (0–1) điều khiển cùng lúc độ phân giải làm việc, ngưỡng đường nét, độ dài nét tối thiểu và khoảng cách gạch.
- Với ảnh có hàng trăm nét: quỹ đạo **nhấc bút thấp** (6 mm) khi nét kế tiếp ở gần thay vì luôn nhấc 4 cm.
  Với chân dung mẫu, thời gian vẽ giảm từ 340 s xuống 110 s. Lớp mực khi render được **vẽ tăng dần**
  (chỉ rasterise điểm mới), nên số nét không làm chậm khung hình.

**Thị giác (OpenCV)**
- *Canvas / nét vẽ bằng tay*: ngưỡng cố định (nền trắng, nét đen đã biết trước).
  *Ảnh chụp* (chỉ còn qua API `POST /api/extract` với `mode: "photo"`, giao diện không dùng nữa): chia ảnh xám cho ước lượng nền (closing với kernel lớn) để khử ánh sáng không đều, sau đó Otsu.
  Nếu ngưỡng Otsu quá cao (trang gần như trống) thì dùng ngưỡng cố định để không “bắt” nhiễu giấy.
- Chọn vùng mực bằng connected components. *Canvas*: giữ **mọi nét đã vẽ** (nhiều chữ, dấu mũ, dấu chấm), chỉ bỏ
  đốm 1–2 pixel. *Ảnh chụp*: bỏ đốm nhỏ và vùng lớn chạm biên (mép bàn/giấy, bóng), chọn thành phần chính, rồi
  **nới dần cụm** sang các thành phần đủ lớn nằm cách cụm không quá nửa cỡ chữ (chữ kế bên, dấu tiếng Việt).
  Vết bẩn ở xa sẽ bị loại.
- Trích nét bằng **xương nét (skeleton)** thay cho đường viền, vì robot cầm bút vẽ một đường đơn chứ không tô viền.
  Dùng Zhang-Suen (vector hoá bằng numpy), thêm một lượt xoá điểm đơn giản bằng bảng tra (LUT) để xương đúng 1 pixel,
  giữ nguyên topology. Không cần `opencv-contrib` hay scikit-image.
- Xương được chuyển thành **đồ thị** (đầu mút, cụm giao điểm, cạnh). Cắt nhánh thừa ngắn (spur).
  Các cạnh được nối thành nét theo heuristic đường Euler: bắt đầu ở đầu mút tự do, ưu tiên phía trên-trái,
  tại giao điểm thì đi theo hướng thẳng nhất. Vì vậy “B”, “8”, “O” chỉ cần 1 nét, “A” cần 2 nét.
- Đầu nét bị co lại khoảng nửa độ dày nét do làm mảnh. Hệ thống kéo dài các đầu mút tự do theo tiếp tuyến,
  dựa vào distance transform.
- Nét được làm trơn (trung bình trượt), đơn giản hoá bằng Douglas-Peucker (`approxPolyDP`), sắp thứ tự
  bằng nearest neighbour (cho phép đảo chiều) để giảm quãng nhấc bút.
- Có thêm chế độ **outline** (`findContours`) để so sánh.

**Quỹ đạo**
- Ký tự được đặt vừa vào ô 20 × 20 cm trên giấy, đúng chiều đọc khi nhìn từ sau lưng robot.
- Lúc vẽ, bút đi với tốc độ không đổi 10 cm/s nhờ lấy mẫu lại theo độ dài cung. Lúc nhấc, hạ và di chuyển,
  vận tốc theo profile cosin (mượt). Bút nhấc cao tối đa 4 cm khi chuyển nét, thấp dần tới 6 mm nếu nét kế tiếp ở gần.

**Robot & mô phỏng**
- KUKA iiwa (có sẵn trong `pybullet_data`, không cần tải thêm) gắn trên bàn. Bút là khối hình học đi theo khâu cuối.
- IK: dùng `calculateInverseKinematics` của PyBullet với null-space (giới hạn khớp, tư thế nghỉ “đang viết”)
  để tư thế tự nhiên và liên tục. Riêng IK này còn sai lệch vài mm, nên mỗi bước được **tinh chỉnh bằng FK**
  trên một robot “bóng” ở physics client riêng. Kết quả IK chính xác dưới 0.1 mm, bút luôn thẳng đứng.
- Khớp được điều khiển bằng **motor POSITION_CONTROL** (có giới hạn lực), mô phỏng động lực học ở 240 Hz.
  Robot không được đặt thẳng vào vị trí. **Mực được sinh từ vị trí đầu bút thực tế**, nên nét vẽ phản ánh
  đúng sai số bám (thường RMS ≈ 0.4 mm).
- Render bằng **TinyRenderer (CPU)** ở chế độ DIRECT. Mực không phải vật thể 3D: các điểm mực 3D được chiếu
  bằng cùng ma trận camera rồi chồng lên ảnh. Phần chồng dùng **segmentation mask** nên chỉ pixel thuộc tờ giấy
  nhận mực, và cánh tay che mực đúng như thật. Cách này nhanh hơn hẳn việc tạo hàng nghìn vật thể nhỏ.
- **Stream thời gian thực**: mô phỏng chạy trong thread riêng, đẩy JPEG + JSON (tiến độ, vị trí bút, mực mới)
  qua WebSocket. Hàng đợi có giới hạn để tạo back-pressure. Lịch render bám theo đồng hồ thực: nếu CPU không
  kịp thì **bỏ bớt khung hình chứ không làm chậm chuyển động**. Người dùng có thể dừng giữa chừng hoặc đổi camera lúc đang vẽ.

## Hạn chế còn lại

- **Không nhận dạng ký tự (OCR)**: hệ thống chỉ phát hiện và trích *hình dạng nét* chứ không biết đó là chữ gì.
  Thứ tự và chiều nét là heuristic, không phải thứ tự viết chuẩn (ví dụ “X” có thể thành 3 nét).
- **Vẽ bằng tay**:
  - Chỉ theo dõi 1 bàn tay, và chỉ có một cử chỉ là chụm.
  - Ngưỡng chụm (0.32 / 0.48) được chọn theo kinh nghiệm. Khi bàn tay nghiêng mạnh hoặc ngón tay bị che, chụm có thể
    nhận sai; lúc đó dùng phím Space.
  - Ánh sáng yếu hoặc ngược sáng làm giảm độ ổn định của keypoint.
  - Đã kiểm thử bằng webcam giả lập trong Chromium: ảnh tay thật từ bộ test của MediaPipe trượt theo hình chữ "V",
    nhận diện chạy khoảng 30 ms/khung trên CPU. Cử chỉ chụm được kiểm thử bằng chuỗi keypoint giả lập.
    **Chưa thử với webcam và bàn tay thật.**
  - Trình duyệt cần hỗ trợ WebAssembly và WebGL (máy không có GPU vẫn có WebGL phần mềm).
- Có thể vẽ nhiều ký tự cùng lúc, nhưng cả cụm sẽ bị co lại cho vừa ô 20 × 20 cm, nên chữ dài sẽ nhỏ.
- **Ảnh chữ (ink)**: ảnh phải chỉ có chữ trên nền tương đối đồng đều. Ảnh chụp có vật khác, bóng tay, đường kẻ vở…
  sẽ bị vẽ theo. Nét chữ Hán viết bằng bút lông (đậm nhạt, bay đầu) bị làm mảnh thành đường đều,
  và thứ tự nét không theo quy tắc bút thuận.
- **Phác hoạ chân dung**: đây là phong cách phác hoạ theo ngưỡng, không phải mô hình học sâu tạo tranh. Kết quả phụ thuộc
  ánh sáng; mặt thiếu sáng hoặc ngược sáng sẽ bị gạch tối nhiều. GrabCut có thể cắt nhầm khi nền cùng màu với tóc hay áo
  (khi đó tắt "Tách người khỏi nền"). Chỉ xử lý **một** khuôn mặt (lớn nhất).
  Đã kiểm thử với một ảnh chân dung thật (ảnh chính thức thuộc phạm vi công cộng) và ảnh tổng hợp.
- Ảnh nhiều chi tiết có thể cần vài phút mô phỏng. Ở tốc độ 10×, giới hạn thực tế là CPU (IK khoảng 1 ms/bước 240 Hz),
  nên chân dung mẫu khoảng 110 s mô phỏng mất khoảng 17–40 s thực. Mực mô phỏng dày 2 mm trên tranh 20 cm,
  nên vùng gạch dày trông đậm hơn bản xem trước.
- Skeleton tạo méo nhỏ ở các giao điểm của nét rất dày (giao điểm chữ X tách thành 2 điểm, góc nhọn hơi bo).
- Tiếp xúc bút–giấy là **hình học** (ngưỡng độ cao đầu bút), không mô phỏng lực ép hay ma sát. Bút không có va chạm.
- TinyRenderer trên CPU tốn khoảng 60 ms/khung ở 480×360. Bật bóng đổ chậm gần gấp đôi, nên ~10–15 fps.
  Mỗi tab trình duyệt tạo một mô phỏng riêng, nên nhiều người dùng cùng lúc sẽ chia nhau CPU.
- Kích thước ký tự, vị trí giấy và tốc độ vẽ đang là hằng số trong `TrajectoryConfig`
  (`backend/trajectory.py`). Frontend có bản sao `CHAR_SIZE`/`PAPER` trong `frontend/app.js` để vẽ mực;
  nếu đổi cấu hình thì phải sửa cả hai chỗ.
