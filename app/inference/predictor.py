import torch
import os
from transformers import VideoMAEImageProcessor
from app.models.model_loader import load_model
from app.inference.preprocessing import preprocess_video

class VideoPredictor:
    def __init__(self, hazard_threshold: float = 0.5, action_threshold: float = 0.5):
        print("Đang khởi tạo Video Predictor...")
        self.hazard_threshold = hazard_threshold
        self.action_threshold = action_threshold
        
        # Load model, từ điển hành vi (vocab), và thiết bị (GPU/CPU)
        self.model, self.idx2action, self.device = load_model()
        
        # Khởi tạo Image Processor để chuẩn hóa khung hình
        self.image_processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        print("Đã tải xong toàn bộ hệ thống nhận diện!")

    def predict(self, video_path: str):
        """
        Nhận vào đường dẫn video, phân tích và trả về kết quả định dạng JSON-friendly.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Không tìm thấy file video để phân tích: {video_path}")
            
        # 1. Tiền xử lý (Cắt frame và chuẩn hóa Tensor)
        pixel_values = preprocess_video(video_path, self.image_processor)
        
        # Thêm chiều Batch (unsqueeze) để tạo thành kích thước [1, 16, 3, 224, 224] và chuyển lên GPU
        pixel_values = pixel_values.unsqueeze(0).to(self.device)
        
        # 2. Suy luận (Chạy Forward Pass với no_grad để tiết kiệm RAM)
        with torch.no_grad():
            hazard_logits, action_logits = self.model(pixel_values)
            
        # 3. Phân tích Tai Nạn (Hazard)
        hazard_probs = torch.softmax(hazard_logits, dim=1)
        # Xác suất lớp 1 (Tai nạn/Hazard) nằm ở index 1
        hazard_prob = hazard_probs[0, 1].item()
        
        is_hazard = hazard_prob >= self.hazard_threshold
        status = "Hazard" if is_hazard else "Safe"
        
        # 4. Phân tích Hành Vi (Multi-label Actions)
        action_probs = torch.sigmoid(action_logits)[0]
        detected_actions = []
        
        # Lặp qua các hành vi, nếu xác suất cao hơn ngưỡng thì lưu lại
        for idx, prob in enumerate(action_probs):
            prob_val = prob.item()
            if prob_val >= self.action_threshold:
                # Ép kiểu idx thành chuỗi (str) vì JSON Dictionary thường lưu key dưới dạng chuỗi
                action_name = self.idx2action.get(str(idx), self.idx2action.get(idx, f"Action_{idx}"))
                detected_actions.append({
                    "action": action_name,
                    "confidence": round(prob_val, 4)
                })
                
        # 5. Đóng gói kết quả đầu ra
        result = {
            "status": status,
            "hazard_confidence": round(hazard_prob, 4),
            "detected_actions": detected_actions
        }
        
        return result
