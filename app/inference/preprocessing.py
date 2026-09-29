import os
import cv2
import numpy as np
import torch
from transformers import VideoMAEImageProcessor

def extract_frames(video_path: str, num_frames: int = 16) -> np.ndarray:
    """Đọc video và trích xuất số lượng frame cố định cách đều nhau."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy video tại đường dẫn: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")
        
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Chuyển BGR (OpenCV mặc định) sang RGB (VideoMAE standard)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"Video không chứa frame nào hoặc bị lỗi: {video_path}")
        
    # Uniform sampling: Lấy đều num_frames từ video (chuẩn giống tập huấn luyện)
    indices = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
    sampled_frames = np.array([frames[i] for i in indices])
    
    return sampled_frames

def preprocess_video(video_path: str, image_processor: VideoMAEImageProcessor) -> torch.Tensor:
    """
    Tiền xử lý video toàn diện: cắt 16 frames và chuẩn hóa qua Image Processor.
    Trả về Tensor (pixel_values) sẵn sàng để đưa vào model.
    """
    # 1. Trích xuất 16 frames
    raw_frames = extract_frames(video_path, num_frames=16)
    
    # 2. Xử lý qua bộ processor của VideoMAE (Resize, Normalize)
    inputs = image_processor(list(raw_frames), return_tensors="pt")
    
    # Trả về tensor pixel_values
    return inputs.pixel_values
