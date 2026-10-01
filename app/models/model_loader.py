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


def remap_videomae_attention_biases(state_dict):
    """
    Map older Hugging Face VideoMAE Q/V bias names onto current Linear.bias names.

    q_bias -> query.bias, v_bias -> value.bias.
    Key projection stays unbiased; no key.bias is created.
    """
    remapped = {}
    q_mapped = 0
    v_mapped = 0

    for key, value in state_dict.items():
        if key.endswith("attention.attention.q_bias"):
            remapped[key[: -len("q_bias")] + "query.bias"] = value
            q_mapped += 1
        elif key.endswith("attention.attention.v_bias"):
            remapped[key[: -len("v_bias")] + "value.bias"] = value
            v_mapped += 1
        else:
            remapped[key] = value

    if q_mapped or v_mapped:
        print(
            f"Applied VideoMAE q_bias/v_bias compatibility mapping: "
            f"{q_mapped} q_bias -> query.bias, {v_mapped} v_bias -> value.bias "
            f"(no key.bias created)."
        )

    return remapped

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

    checkpoint = torch.load(weights_path, map_location=device)
    state_dict = remap_videomae_attention_biases(checkpoint)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())
    missing_keys = sorted(model_keys - ckpt_keys)
    unexpected_keys = sorted(ckpt_keys - model_keys)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            "Checkpoint keys do not match the model after q_bias/v_bias remapping. "
            f"Missing ({len(missing_keys)}): {missing_keys[:20]}. "
            f"Unexpected ({len(unexpected_keys)}): {unexpected_keys[:20]}."
        )

    model.load_state_dict(state_dict, strict=True)
    
    # 4. Đưa model về chế độ dự đoán (Evaluation Mode)
    model.to(device)
    model.eval()
    
    print("Mô hình đã sẵn sàng cho inference!")
    return model, idx2action, device
