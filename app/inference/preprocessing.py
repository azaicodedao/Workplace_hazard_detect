import os
import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor
from typing import List


def extract_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Doc video va trich xuat so luong frame co dinh cach deu nhau."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Khong tim thay video tai duong dan: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Khong the mo video: {video_path}")

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Chuyen BGR (OpenCV mac dinh) sang RGB (VideoMAE standard)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"Video khong chua frame nao hoac bi loi: {video_path}")

    # Uniform sampling: lay deu num_frames tu video (chuan giong tap huan luyen)
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    sampled_frames = np.array([frames[i] for i in indices])

    return sampled_frames


def preprocess_video(video_path: str, image_processor: VideoMAEImageProcessor) -> torch.Tensor:
    """
    Tien xu ly video toan dien: cat 16 frames va chuan hoa qua Image Processor.
    Tra ve Tensor (pixel_values) san sang de dua vao model.
    Ham nay KHONG duoc sua doi -- giu nguyen hanh vi cho POST /predict/.
    """
    # 1. Trich xuat 16 frames
    raw_frames = extract_frames(video_path, num_frames=16)

    # 2. Xu ly qua bo processor cua VideoMAE (Resize, Normalize)
    inputs = image_processor(list(raw_frames), return_tensors="pt")

    # Tra ve tensor pixel_values shape [1, 16, 3, 224, 224]
    return inputs.pixel_values


def preprocess_frames(
    frames: List[np.ndarray],
    image_processor: VideoMAEImageProcessor,
) -> torch.Tensor:
    """
    Tien xu ly danh sach frames da co san (dung cho Live Camera).

    Parameters
    ----------
    frames : List[np.ndarray]
        Dung 16 numpy arrays, moi array shape (H, W, 3), dtype uint8, kenh mau RGB.
        Thu tu phai theo thu tu thoi gian tu som den muon.
    image_processor : VideoMAEImageProcessor
        Cung bo processor voi preprocess_video() de dam bao nhat quan.

    Returns
    -------
    torch.Tensor
        pixel_values shape [1, 16, 3, 224, 224] -- cung format voi preprocess_video().

    Notes
    -----
    Ham nay KHONG tu thuc hien sampling. Caller phai dam bao dung 16 frames
    da duoc chon dung khoang thoi gian phu hop truoc khi truyen vao.
    Xem LIVE_SAMPLE_INTERVAL_MS va LIVE_NUM_FRAMES trong routes.py de biet cau hinh.
    """
    if len(frames) != 16:
        raise ValueError(
            f"preprocess_frames yeu cau dung 16 frames, nhan duoc {len(frames)}."
        )

    # Dung cung image_processor voi preprocess_video de dam bao nhat quan
    inputs = image_processor(frames, return_tensors="pt")
    return inputs.pixel_values
