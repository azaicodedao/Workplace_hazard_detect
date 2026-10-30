import os
import sys
import json
import cv2
import numpy as np
import torch

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from app.inference.predictor import VideoPredictor
from app.inference.preprocessing import extract_frames, preprocess_video, preprocess_frames
from app.api.routes import _decode_image_to_rgb

VIDEO_PATH = "test_data/input_video.mp4"
FRAMES_DIR = "test_data/normal_classroom_frames"
os.makedirs(FRAMES_DIR, exist_ok=True)

def run_investigation():
    report = {}

    print("==================================================================")
    print("BƯỚC 1: KIỂM TRA VIDEO GỐC")
    print("==================================================================")
    if not os.path.exists(VIDEO_PATH):
        raise FileNotFoundError(f"Không tìm thấy video: {VIDEO_PATH}")

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("OpenCV không mở được video!")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    cap.release()

    video_info = {
        "file": VIDEO_PATH,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
    }
    report["video_info"] = video_info
    print(json.dumps(video_info, indent=2))

    print("\nKhởi tạo VideoPredictor...")
    predictor = VideoPredictor()

    # Forward pass trực tiếp để lấy cả logits
    pixel_values_orig = preprocess_video(VIDEO_PATH, predictor.image_processor).to(predictor.device)
    with torch.no_grad():
        h_logits_orig, a_logits_orig = predictor.model(pixel_values_orig)
        h_probs_orig = torch.softmax(h_logits_orig, dim=1)[0].tolist()

    pred_result_orig = predictor._infer(pixel_values_orig)
    report["step1_predict"] = {
        "status": pred_result_orig["status"],
        "hazard_confidence": pred_result_orig["hazard_confidence"],
        "hazard_logits": [round(x, 4) for x in h_logits_orig[0].tolist()],
        "hazard_softmax_probs": {"Safe_prob": round(h_probs_orig[0], 4), "Hazard_prob": round(h_probs_orig[1], 4)},
        "detected_actions": pred_result_orig["detected_actions"]
    }
    print("Kết quả Video Analysis (gốc):")
    print(json.dumps(report["step1_predict"], indent=2))

    print("\n==================================================================")
    print("BƯỚC 2: TẠO BỘ 16 FRAMES ĐỐI CHỨNG")
    print("==================================================================")
    raw_16_frames_rgb = extract_frames(VIDEO_PATH, num_frames=16)
    saved_frames_info = []

    for i, frame_rgb in enumerate(raw_16_frames_rgb):
        frame_filename = f"frame_{i:02d}.png"
        frame_path = os.path.join(FRAMES_DIR, frame_filename)
        # Chuyển RGB sang BGR để cv2.imwrite ghi màu chuẩn
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

        saved_frames_info.append({
            "index": i,
            "filename": frame_filename,
            "shape": list(frame_rgb.shape), # [H, W, 3]
            "dtype": str(frame_rgb.dtype)
        })

    report["step2_frames"] = {
        "folder": FRAMES_DIR,
        "count": len(saved_frames_info),
        "resolution": f"{raw_16_frames_rgb[0].shape[1]}x{raw_16_frames_rgb[0].shape[0]}",
        "frames": saved_frames_info
    }
    print(f"Đã lưu thành công 16 frames vào {FRAMES_DIR}. Độ phân giải: {report['step2_frames']['resolution']}")

    print("\n==================================================================")
    print("BƯỚC 3: SO SÁNH HAI ĐƯỜNG INFERENCE BẰNG CÙNG 16 FRAMES")
    print("==================================================================")
    # Thử nghiệm A: Video upload
    tensor_a = preprocess_video(VIDEO_PATH, predictor.image_processor).to(predictor.device)
    with torch.no_grad():
        h_logits_a, a_logits_a = predictor.model(tensor_a)
        h_probs_a = torch.softmax(h_logits_a, dim=1)[0].tolist()
    res_a = predictor._infer(tensor_a)

    # Thử nghiệm B: Đọc lại 16 ảnh từ đĩa
    loaded_frames_rgb = []
    for i in range(16):
        fpath = os.path.join(FRAMES_DIR, f"frame_{i:02d}.png")
        bgr = cv2.imread(fpath, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        loaded_frames_rgb.append(rgb)

    tensor_b = preprocess_frames(loaded_frames_rgb, predictor.image_processor).to(predictor.device)
    with torch.no_grad():
        h_logits_b, a_logits_b = predictor.model(tensor_b)
        h_probs_b = torch.softmax(h_logits_b, dim=1)[0].tolist()
    res_b = predictor._infer(tensor_b)

    # So sánh tensor
    diff = torch.abs(tensor_a - tensor_b)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()

    comparison_step3 = {
        "tensor_a_shape": list(tensor_a.shape),
        "tensor_b_shape": list(tensor_b.shape),
        "tensor_a_stats": {
            "min": round(tensor_a.min().item(), 4),
            "max": round(tensor_a.max().item(), 4),
            "mean": round(tensor_a.mean().item(), 4),
        },
        "tensor_b_stats": {
            "min": round(tensor_b.min().item(), 4),
            "max": round(tensor_b.max().item(), 4),
            "mean": round(tensor_b.mean().item(), 4),
        },
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "experiment_A_video_upload": {
            "hazard_logits": [round(x, 4) for x in h_logits_a[0].tolist()],
            "hazard_probs": {"Safe": round(h_probs_a[0], 4), "Hazard": round(h_probs_a[1], 4)},
            "hazard_confidence": res_a["hazard_confidence"],
            "status": res_a["status"],
            "detected_actions": res_a["detected_actions"]
        },
        "experiment_B_live_frames": {
            "hazard_logits": [round(x, 4) for x in h_logits_b[0].tolist()],
            "hazard_probs": {"Safe": round(h_probs_b[0], 4), "Hazard": round(h_probs_b[1], 4)},
            "hazard_confidence": res_b["hazard_confidence"],
            "status": res_b["status"],
            "detected_actions": res_b["detected_actions"]
        }
    }
    report["step3_comparison"] = comparison_step3
    print(json.dumps(comparison_step3, indent=2))

    print("\n==================================================================")
    print("BƯỚC 4: KIỂM TRA RIÊNG ĐƯỜNG TRUYỀN LIVE CAMERA (JPEG & RESIZE)")
    print("==================================================================")
    # 4.1: Encode 16 frame gốc sang JPEG (q=85), decode lại qua _decode_image_to_rgb (giữ nguyên độ phân giải)
    jpeg_bytes_list_native = []
    decoded_native = []
    for i, frame in enumerate(loaded_frames_rgb):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        _, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        jpeg_bytes = enc.tobytes()
        jpeg_bytes_list_native.append(jpeg_bytes)
        decoded_native.append(_decode_image_to_rgb(jpeg_bytes, i))

    res_4_1 = predictor.predict_from_frames(decoded_native)

    # 4.2: Mô phỏng canvas Live Cam CŨ: Resize ép dẹp 320x240, JPEG q=85
    decoded_squashed_320x240 = []
    for i, frame in enumerate(loaded_frames_rgb):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_resized = cv2.resize(bgr, (320, 240), interpolation=cv2.INTER_LINEAR)
        _, enc = cv2.imencode(".jpg", bgr_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        decoded_squashed_320x240.append(_decode_image_to_rgb(enc.tobytes(), i))

    res_4_2 = predictor.predict_from_frames(decoded_squashed_320x240)

    # 4.3: Gọi FastAPI TestClient POST /predict/live/ với 16 ảnh JPEG native
    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    files_payload = [
        ("frames", (f"frame_{i}.jpg", jpeg_bytes_list_native[i], "image/jpeg"))
        for i in range(16)
    ]
    resp = client.post("/predict/live/", files=files_payload)
    live_api_res = resp.json() if resp.status_code == 200 else {"error": resp.text, "status_code": resp.status_code}

    step4_results = {
        "4_1_native_res_jpeg85": res_4_1,
        "4_2_squashed_320x240_jpeg85": res_4_2,
        "4_3_live_api_post": live_api_res
    }
    report["step4_live_pipeline"] = step4_results
    print(json.dumps(step4_results, indent=2))

    print("\n==================================================================")
    print("BƯỚC 5: KIỂM TRA KHÁC BIỆT VỀ CÁCH LẤY MẪU (TEMPORAL SAMPLING)")
    print("==================================================================")
    # Đọc toàn bộ frames từ video gốc
    cap = cv2.VideoCapture(VIDEO_PATH)
    all_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    step_333ms = max(1, int(round(fps * 0.333)))
    print(f"Tổng số frame: {len(all_frames)}, FPS: {fps:.2f}, Bước nhảy cho 333ms: {step_333ms} frames")

    # 5.1: 16 frames uniform sampling (đã có ở bước 1 & 3: res_a)
    uniform_res = res_a

    # 5.2: Các cửa sổ sliding window liên tiếp với chu kỳ 333ms (16 frame, trượt 8 frame)
    # Cửa sổ 1: 0, step, 2*step, ..., 15*step (span = 15*step ~ 5 giây)
    # Trượt 8 frame: bắt đầu tại 8*step
    sliding_results = []
    window_idx = 0
    start_frame = 0

    while True:
        frame_indices = [start_frame + i * step_333ms for i in range(16)]
        if frame_indices[-1] >= len(all_frames):
            break

        window_frames = [all_frames[idx] for idx in frame_indices]
        t_start = start_frame / fps
        t_end = frame_indices[-1] / fps

        pred = predictor.predict_from_frames(window_frames)
        sliding_results.append({
            "window_index": window_idx,
            "start_sec": round(t_start, 2),
            "end_sec": round(t_end, 2),
            "frame_indices": [frame_indices[0], frame_indices[-1]],
            "hazard_confidence": pred["hazard_confidence"],
            "status": pred["status"],
            "detected_actions": pred["detected_actions"]
        })

        # Trượt 8 frame lấy mẫu (8 * step_333ms ~ 2.6s)
        start_frame += 8 * step_333ms
        window_idx += 1

    # Thêm trường hợp lấy 16 frame LIÊN TIẾP ở tốc độ gốc (16 frame consecutive @ 30fps = 0.53s)
    consecutive_results = []
    for c_start in range(0, min(len(all_frames) - 16, int(fps * 10)), int(fps * 2)): # mỗi 2 giây thử 1 clip 0.5s
        c_frames = [all_frames[c_start + i] for i in range(16)]
        c_pred = predictor.predict_from_frames(c_frames)
        consecutive_results.append({
            "start_sec": round(c_start / fps, 2),
            "end_sec": round((c_start + 15) / fps, 2),
            "hazard_confidence": c_pred["hazard_confidence"],
            "status": c_pred["status"],
            "detected_actions": c_pred["detected_actions"]
        })

    step5_results = {
        "uniform_sampling_16_frames": {
            "hazard_confidence": uniform_res["hazard_confidence"],
            "status": uniform_res["status"],
            "detected_actions": uniform_res["detected_actions"]
        },
        "step_333ms_frames": step_333ms,
        "sliding_windows_333ms": sliding_results,
        "consecutive_16_frames_native_fps": consecutive_results
    }
    report["step5_sampling"] = step5_results
    print(json.dumps(step5_results, indent=2))

    # Ghi toàn bộ report ra file json
    with open("test_data/investigation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n==================================================================")
    print("HOÀN TẤT ĐIỀU TRA! Đã lưu kết quả tại test_data/investigation_report.json")
    print("==================================================================")

if __name__ == "__main__":
    run_investigation()
