import torch
import json
import os
from app.models.videomae_model import MultiTaskVideoMAE

def load_action_vocab(vocab_path: str):
    """
    Load file action_vocab.json để lấy từ điển các hành vi.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("action2idx", {}), data.get("idx2action", {})

def load_model(
    model_ckpt: str = "MCG-NJU/videomae-base", 
    weights_path: str = "models/best_multitask_videomae.pth", 
    vocab_path: str = "models/action_vocab.json"
):
    """
    Khởi tạo mô hình, nạp trọng số (.pth) và trả về model ở chế độ eval().
    """
    # 1. Đọc từ điển hành vi
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Không tìm thấy file từ điển hành vi: {vocab_path}. Vui lòng copy file action_vocab.json vào thư mục models/.")
    
    _, idx2action = load_action_vocab(vocab_path)
    num_actions = len(idx2action)

    if num_actions == 0:
        raise ValueError(f"File {vocab_path} không hợp lệ hoặc không chứa hành vi nào.")

    # 2. Khởi tạo kiến trúc mô hình (bản vẽ)
    print(f"Khởi tạo kiến trúc mô hình với {num_actions} hành vi...")
    model = MultiTaskVideoMAE(model_ckpt=model_ckpt, num_actions=num_actions)

    # 3. Load file trọng số (.pth)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}. Vui lòng copy file .pth vào đúng đường dẫn.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang nạp trọng số từ {weights_path} lên {device}...")
    
    # Nạp weights vào model (map_location đảm bảo an toàn nếu bạn train trên GPU nhưng test trên CPU)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    
    # 4. Đưa model về chế độ dự đoán (Evaluation Mode)
    model.to(device)
    model.eval()
    
    print("Mô hình đã sẵn sàng cho inference!")
    return model, idx2action, device
