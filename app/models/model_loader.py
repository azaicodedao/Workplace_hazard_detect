import torch
import json
import os
from app.models.videomae_model import MultiTaskVideoMAE

def load_action_vocab(vocab_path: str):
    """
    Load file action_vocab.json de lay tu dien cac hanh vi.
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
    Khoi tao mo hinh, nap trong so (.pth) va tra ve model o che do eval().
    """
    # 1. Doc tu dien hanh vi
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Khong tim thay file tu dien hanh vi: {vocab_path}. Vui long copy file action_vocab.json vao thu muc models/.")
    
    _, idx2action = load_action_vocab(vocab_path)
    num_actions = len(idx2action)

    if num_actions == 0:
        raise ValueError(f"File {vocab_path} khong hop le hoac khong chua hanh vi nao.")

    # 2. Khoi tao kien truc mo hinh
    print(f"Khoi tao kien truc mo hinh voi {num_actions} hanh vi...")
    model = MultiTaskVideoMAE(model_ckpt=model_ckpt, num_actions=num_actions)

    # 3. Load file trong so (.pth)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Khong tim thay file trong so: {weights_path}. Vui long copy file .pth vao dung duong dan.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dang nap trong so tu {weights_path} len {device}...")

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
    
    # 4. Dua model ve che do du doan (Evaluation Mode)
    model.to(device)
    model.eval()
    
    print("Mo hinh da san sang cho inference!")
    return model, idx2action, device
