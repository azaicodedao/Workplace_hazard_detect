import torch
import os
import numpy as np
from transformers import VideoMAEImageProcessor
from app.models.model_loader import load_model
from app.inference.preprocessing import preprocess_video, preprocess_frames
from typing import List


class VideoPredictor:
    def __init__(self, hazard_threshold: float = 0.5, action_threshold: float = 0.5):
        print("Dang khoi tao Video Predictor...")
        self.hazard_threshold = hazard_threshold
        self.action_threshold = action_threshold

        # Load model, tu dien hanh vi (vocab), va thiet bi (GPU/CPU)
        self.model, self.idx2action, self.device = load_model()

        # Khoi tao Image Processor de chuan hoa khung hinh
        self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        print("Da tai xong toan bo he thong nhan dien!")

    # ------------------------------------------------------------------
    # Private helper: forward pass + decode output
    # Giu nguyen logic softmax/sigmoid va nguong phan loai.
    # ------------------------------------------------------------------
    def _infer(self, pixel_values: torch.Tensor) -> dict:
        """
        Chay forward pass va giai ma ket qua.
        Nhan pixel_values shape [1, 16, 3, 224, 224] da duoc chuyen len device.
        Tra ve dict {status, hazard_confidence, detected_actions}.
        Ham nay KHONG duoc thay doi nguong phan loai.
        """
        # Forward pass (no_grad de tiet kiem RAM/VRAM)
        with torch.no_grad():
            hazard_logits, action_logits = self.model(pixel_values)

        # --- Phan tich Tai Nan (Hazard) ---
        hazard_probs = torch.softmax(hazard_logits, dim=1)
        # Xac suat lop 1 (Hazard) nam o index 1 -- KHONG thay doi
        hazard_prob = hazard_probs[0, 1].item()

        is_hazard = hazard_prob >= self.hazard_threshold
        status = "Hazard" if is_hazard else "Safe"

        # --- Phan tich Hanh Vi (Multi-label Actions) ---
        action_probs = torch.sigmoid(action_logits)[0]
        detected_actions = []

        for idx, prob in enumerate(action_probs):
            prob_val = prob.item()
            if prob_val >= self.action_threshold:
                # Ep kieu idx thanh chuoi (str) vi JSON Dictionary luu key duoi dang chuoi
                action_name = self.idx2action.get(str(idx), self.idx2action.get(idx, f"Action_{idx}"))
                detected_actions.append({
                    "action": action_name,
                    "confidence": round(prob_val, 4),
                })

        return {
            "status": status,
            "hazard_confidence": round(hazard_prob, 4),
            "detected_actions": detected_actions,
        }

    # ------------------------------------------------------------------
    # Public API 1: nhan duong dan video (dung cho POST /predict/)
    # KHONG sua doi ham nay -- giu nguyen contract.
    # ------------------------------------------------------------------
    def predict(self, video_path: str) -> dict:
        """
        Nhan vao duong dan video, phan tich va tra ve ket qua dinh dang JSON-friendly.
        Ham nay giu nguyen 100% hanh vi so voi phien ban goc.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Khong tim thay file video de phan tich: {video_path}")

        # 1. Tien xu ly (Cat frame va chuan hoa Tensor)
        pixel_values = preprocess_video(video_path, self.image_processor)

        # Chuyen Tensor len GPU/CPU
        pixel_values = pixel_values.to(self.device)

        # 2. Suy luan va tra ket qua
        return self._infer(pixel_values)

    # ------------------------------------------------------------------
    # Public API 2: nhan list frames (dung cho POST /predict/live/)
    # Su dung chung _infer() voi predict() -- khong tao model thu hai.
    # ------------------------------------------------------------------
    def predict_from_frames(self, frames: List[np.ndarray]) -> dict:
        """
        Nhan danh sach 16 numpy arrays (H, W, 3), RGB, theo thu tu thoi gian.
        Preprocess va chay inference tren cung model dang duoc load.
        Tra ve cung dinh dang dict nhu predict().

        Parameters
        ----------
        frames : List[np.ndarray]
            Dung 16 frames, shape (H, W, 3), dtype uint8, kenh mau RGB.

        Returns
        -------
        dict
            {"status": "Safe"|"Hazard", "hazard_confidence": float, "detected_actions": list}
        """
        if len(frames) != 16:
            raise ValueError(
                f"predict_from_frames yeu cau dung 16 frames, nhan duoc {len(frames)}."
            )

        # 1. Tien xu ly bang cung image_processor
        pixel_values = preprocess_frames(frames, self.image_processor)

        # Chuyen Tensor len GPU/CPU
        pixel_values = pixel_values.to(self.device)

        # 2. Suy luan va tra ket qua (dung chung _infer voi predict())
        return self._infer(pixel_values)
