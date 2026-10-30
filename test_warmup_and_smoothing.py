import sys
import unittest

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

class LiveCameraStateSimulator:
    """
    Mo phong 100% logic State Machine cua Live Camera trong frontend/js/app.js
    bao gom: Session Guard, Camera Warm-up, Sliding Buffer, Temporal Smoothing.
    """
    def __init__(self, warmup_duration_ms=2500, required_consecutive_hazards=2):
        self.WARMUP_DURATION_MS = warmup_duration_ms
        self.REQUIRED_CONSECUTIVE_HAZARDS = required_consecutive_hazards
        self.LIVE_NUM_FRAMES = 16

        self.camera_active = False
        self.current_session_id = 0
        self.is_warming_up = False
        self.frame_buffer = []
        self.consecutive_hazard_count = 0
        self.last_smoothed_status = "Safe"
        self.displayed_status = None
        self.displayed_subtitle = None
        self.is_live_analyzing = False
        self.session_logs = []

    def log(self, msg):
        self.session_logs.append(msg)

    def start_camera(self):
        if self.camera_active:
            return
        self.current_session_id += 1
        this_session = self.current_session_id

        self.camera_active = True
        self.is_warming_up = True
        self.consecutive_hazard_count = 0
        self.last_smoothed_status = "Safe"
        self.frame_buffer = []
        self.displayed_status = None # renderLiveEmpty()
        self.log(f"Camera started (Session #{this_session}). Warming up for {self.WARMUP_DURATION_MS/1000}s.")

    def finish_warmup(self, session_id):
        if session_id != self.current_session_id or not self.camera_active:
            return
        self.is_warming_up = False
        self.log(f"Warm-up complete (Session #{session_id}). Starting frame sampling.")

    def sample_frame(self, session_id, frame_blob="valid_frame"):
        if not self.camera_active or session_id != self.current_session_id or self.is_warming_up:
            return None # Frame bi discard trong warm-up hoac khi camera tat

        self.frame_buffer.append(frame_blob)
        if len(self.frame_buffer) >= self.LIVE_NUM_FRAMES:
            snapshot = self.frame_buffer[:self.LIVE_NUM_FRAMES]
            self.frame_buffer = self.frame_buffer[8:] # Sliding window trượt 8 frames
            return snapshot
        return None

    def on_inference_response(self, session_id, response_status_code, data=None, error_msg=None):
        if session_id != self.current_session_id or not self.camera_active:
            self.log(f"Discarding response from outdated session: {session_id}")
            return "DISCARDED"

        if response_status_code == 429:
            self.log(f"HTTP 429. Consecutive Hazard count preserved at: {self.consecutive_hazard_count}")
            return "BUSY_PRESERVED"

        if response_status_code != 200:
            self.log(f"HTTP Error {response_status_code}. Consecutive Hazard count preserved at: {self.consecutive_hazard_count}")
            return "ERROR_PRESERVED"

        # 200 OK -> Xử lý Temporal Smoothing
        raw_status = data.get("status")
        confidence = data.get("hazard_confidence", 0.0)

        if raw_status == "Hazard":
            self.consecutive_hazard_count += 1
        else:
            self.consecutive_hazard_count = 0

        if self.consecutive_hazard_count >= self.REQUIRED_CONSECUTIVE_HAZARDS:
            self.displayed_status = "Hazard"
        else:
            self.displayed_status = "Safe"

        self.last_smoothed_status = self.displayed_status
        self.log(f"Inference result: raw={raw_status}, consecutive={self.consecutive_hazard_count}, displayed={self.displayed_status}")
        return self.displayed_status

    def stop_camera(self):
        self.log(f"Stopping camera session #{self.current_session_id}")
        self.camera_active = False
        self.current_session_id += 1 # Vo hieu hoa pending requests
        self.is_warming_up = False
        self.frame_buffer = []
        self.is_live_analyzing = False
        self.consecutive_hazard_count = 0
        self.last_smoothed_status = "Safe"
        self.displayed_status = None # renderLiveEmpty()

    def switch_tab(self, target_tab):
        if target_tab == "video":
            if self.camera_active:
                self.stop_camera()


class TestLiveCameraImprovements(unittest.TestCase):

    def setUp(self):
        self.cam = LiveCameraStateSimulator()

    def test_case_1_normal_startup(self):
        """1. Camera khởi động bình thường: Bật camera -> warm-up -> bắt đầu lấy mẫu."""
        self.cam.start_camera()
        self.assertTrue(self.cam.camera_active)
        self.assertTrue(self.cam.is_warming_up)
        self.assertEqual(len(self.cam.frame_buffer), 0)

        # Trong thời gian warm-up, frame bị bỏ qua
        snapshot = self.cam.sample_frame(self.cam.current_session_id, "frame1")
        self.assertIsNone(snapshot)
        self.assertEqual(len(self.cam.frame_buffer), 0)

        # Hết warm-up
        self.cam.finish_warmup(self.cam.current_session_id)
        self.assertFalse(self.cam.is_warming_up)

        # Lấy mẫu bình thường
        for i in range(16):
            snapshot = self.cam.sample_frame(self.cam.current_session_id, f"frame_{i}")
        self.assertIsNotNone(snapshot)
        self.assertEqual(len(snapshot), 16)
        print("PASS: Trường hợp 1 - Camera khởi động bình thường.")

    def test_case_2_startup_dark_frames_filtered(self):
        """2. Camera khởi động nhưng hình ảnh ban đầu tối: Không lọt vào buffer nhờ warm-up."""
        self.cam.start_camera()
        sid = self.cam.current_session_id

        # Giả lập 8 frames tối trong 2.5s đầu
        for _ in range(8):
            self.cam.sample_frame(sid, "dark_glitch_frame")
        self.assertEqual(len(self.cam.frame_buffer), 0, "Buffer phải rỗng trong lúc warm-up!")

        # Kết thúc warm-up, chỉ nhận frames chuẩn sau đó
        self.cam.finish_warmup(sid)
        for i in range(16):
            self.cam.sample_frame(sid, f"stable_frame_{i}")
        self.assertEqual(len(self.cam.frame_buffer), 8) # 16 - 8 trượt = còn 8 frames
        print("PASS: Trường hợp 2 - Hình ảnh tối ban đầu bị loại bỏ hoàn toàn.")

    def test_case_3_single_hazard_then_safe(self):
        """3. Một cửa sổ dự đoán Hazard, cửa sổ tiếp theo Safe: Không kích hoạt cảnh báo đỏ."""
        self.cam.start_camera()
        sid = self.cam.current_session_id
        self.cam.finish_warmup(sid)

        # Cửa sổ 1: Raw Hazard (lần 1)
        res1 = self.cam.on_inference_response(sid, 200, {"status": "Hazard", "hazard_confidence": 0.58})
        self.assertEqual(self.cam.consecutive_hazard_count, 1)
        self.assertEqual(res1, "Safe", "Lần 1 dự đoán Hazard chưa đủ điều kiện cảnh báo, giao diện phải hiển thị Safe!")
        self.assertEqual(self.cam.displayed_status, "Safe")

        # Cửa sổ 2: Safe
        res2 = self.cam.on_inference_response(sid, 200, {"status": "Safe", "hazard_confidence": 0.12})
        self.assertEqual(self.cam.consecutive_hazard_count, 0, "Kết quả Safe phải reset bộ đếm về 0!")
        self.assertEqual(res2, "Safe")
        self.assertEqual(self.cam.displayed_status, "Safe")
        print("PASS: Trường hợp 3 - Một cửa sổ Hazard đơn lẻ không làm đổi giao diện, cửa sổ Safe tiếp theo reset bộ đếm.")

    def test_case_4_two_consecutive_hazards(self):
        """4. Hai cửa sổ liên tiếp dự đoán Hazard: Kích hoạt cảnh báo đỏ."""
        self.cam.start_camera()
        sid = self.cam.current_session_id
        self.cam.finish_warmup(sid)

        # Cửa sổ 1: Hazard
        self.cam.on_inference_response(sid, 200, {"status": "Hazard", "hazard_confidence": 0.55})
        self.assertEqual(self.cam.displayed_status, "Safe")

        # Cửa sổ 2: Hazard liên tiếp
        res2 = self.cam.on_inference_response(sid, 200, {"status": "Hazard", "hazard_confidence": 0.62})
        self.assertEqual(self.cam.consecutive_hazard_count, 2)
        self.assertEqual(res2, "Hazard")
        self.assertEqual(self.cam.displayed_status, "Hazard", "Đủ 2 lần liên tiếp phải kích hoạt cảnh báo Hazard!")

        # Cửa sổ 3: Trở lại Safe
        res3 = self.cam.on_inference_response(sid, 200, {"status": "Safe", "hazard_confidence": 0.20})
        self.assertEqual(self.cam.consecutive_hazard_count, 0)
        self.assertEqual(res3, "Safe", "Khi có Safe mới, giao diện lập tức chuyển lại Safe!")
        print("PASS: Trường hợp 4 - Hai cửa sổ liên tiếp dự đoán Hazard kích hoạt cảnh báo thành công và hồi phục khi Safe.")

    def test_case_5_api_429_preserves_count(self):
        """5. API trả HTTP 429: Không tính là Safe hoặc Hazard, bảo toàn bộ đếm."""
        self.cam.start_camera()
        sid = self.cam.current_session_id
        self.cam.finish_warmup(sid)

        # Giả sử đã có 1 Hazard
        self.cam.on_inference_response(sid, 200, {"status": "Hazard", "hazard_confidence": 0.52})
        self.assertEqual(self.cam.consecutive_hazard_count, 1)

        # Request tiếp theo gặp 429
        res = self.cam.on_inference_response(sid, 429)
        self.assertEqual(res, "BUSY_PRESERVED")
        self.assertEqual(self.cam.consecutive_hazard_count, 1, "HTTP 429 không được làm reset hoặc tăng bộ đếm!")
        print("PASS: Trường hợp 5 - HTTP 429 bảo toàn bộ đếm smoothing.")

    def test_case_6_api_500_or_network_error(self):
        """6. API trả HTTP 500 hoặc mất kết nối: Bảo toàn bộ đếm smoothing."""
        self.cam.start_camera()
        sid = self.cam.current_session_id
        self.cam.finish_warmup(sid)

        # Giả sử đã có 1 Hazard
        self.cam.on_inference_response(sid, 200, {"status": "Hazard", "hazard_confidence": 0.53})
        self.assertEqual(self.cam.consecutive_hazard_count, 1)

        # Gặp lỗi 500
        res = self.cam.on_inference_response(sid, 500, error_msg="Server crash")
        self.assertEqual(res, "ERROR_PRESERVED")
        self.assertEqual(self.cam.consecutive_hazard_count, 1, "HTTP Error không làm thay đổi bộ đếm!")
        print("PASS: Trường hợp 6 - Lỗi mạng / HTTP 500 bảo toàn bộ đếm smoothing.")

    def test_case_7_stop_camera_during_inflight_request(self):
        """7. Người dùng dừng camera khi request đang xử lý: Response bị discard, không update UI."""
        self.cam.start_camera()
        old_sid = self.cam.current_session_id
        self.cam.finish_warmup(old_sid)

        # Request đang bay với session cũ, người dùng bấm Stop
        self.cam.stop_camera()
        self.assertFalse(self.cam.camera_active)

        # Response bay về sau khi đã Stop
        res = self.cam.on_inference_response(old_sid, 200, {"status": "Hazard", "hazard_confidence": 0.99})
        self.assertEqual(res, "DISCARDED")
        self.assertIsNone(self.cam.displayed_status, "UI phải ở trạng thái empty, không được nhận kết quả từ session cũ!")
        print("PASS: Trường hợp 7 - Dừng camera khi request đang xử lý được loại bỏ sạch sẽ.")

    def test_case_8_switch_tab_resets_state(self):
        """8. Người dùng chuyển tab: stopCamera được gọi, reset toàn bộ state."""
        self.cam.start_camera()
        sid = self.cam.current_session_id
        self.cam.finish_warmup(sid)
        # Giả sử đang có 2 Hazard (đang báo động đỏ)
        self.cam.on_inference_response(sid, 200, {"status": "Hazard"})
        self.cam.on_inference_response(sid, 200, {"status": "Hazard"})
        self.assertEqual(self.cam.displayed_status, "Hazard")

        # Chuyển tab sang Video
        self.cam.switch_tab("video")
        self.assertFalse(self.cam.camera_active)
        self.assertEqual(self.cam.consecutive_hazard_count, 0)
        self.assertIsNone(self.cam.displayed_status, "Trạng thái Hazard cũ không được tồn tại sau khi chuyển tab!")
        print("PASS: Trường hợp 8 - Chuyển tab reset sạch sẽ mọi cảnh báo cũ.")

    def test_case_9_restart_camera_new_session(self):
        """9. Khởi động lại camera sau khi đã dừng: Tạo session mới, reset lịch sử."""
        self.cam.start_camera()
        sid1 = self.cam.current_session_id
        self.cam.finish_warmup(sid1)
        self.cam.on_inference_response(sid1, 200, {"status": "Hazard"})
        self.cam.stop_camera()

        # Khởi động lại
        self.cam.start_camera()
        sid2 = self.cam.current_session_id
        self.assertNotEqual(sid1, sid2)
        self.assertTrue(self.cam.is_warming_up)
        self.assertEqual(self.cam.consecutive_hazard_count, 0)
        self.assertEqual(self.cam.frame_buffer, [])
        print("PASS: Trường hợp 9 - Khởi động lại camera tạo session mới và reset toàn bộ lịch sử.")


if __name__ == "__main__":
    unittest.main()
