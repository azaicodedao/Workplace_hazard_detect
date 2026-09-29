import torch
import torch.nn as nn
from transformers import VideoMAEModel

class MultiTaskVideoMAE(nn.Module):
    def __init__(self, model_ckpt: str = "MCG-NJU/videomae-base", num_actions: int = 1):
        super().__init__()
        # 1. Khởi tạo Backbone
        self.videomae = VideoMAEModel.from_pretrained(model_ckpt)
        hidden_size = self.videomae.config.hidden_size

        # 2. Lớp Attention Pooling để gom thông tin từ các frame
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # 3. Nhánh phân loại nguy hiểm (Hazard - 2 classes: Safe vs Hazard)
        self.hazard_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

        # 4. Nhánh phân loại hành vi (Action - Multi-label)
        self.action_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_actions)
        )

    def forward(self, pixel_values):
        # Trích xuất đặc trưng qua VideoMAE
        outputs = self.videomae(pixel_values=pixel_values)
        sequence_output = outputs.last_hidden_state 

        # Áp dụng attention pooling
        attn_weights = torch.softmax(self.attention(sequence_output), dim=1)
        pooled_output = torch.sum(sequence_output * attn_weights, dim=1)

        # Trả về kết quả từ cả 2 nhánh (Hazard và Action)
        return self.hazard_head(pooled_output), self.action_head(pooled_output)
