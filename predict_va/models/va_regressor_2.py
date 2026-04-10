import torch
import torch.nn as nn
from transformers import AutoModel

class VARegressor(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.regressor = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0]
        va = self.regressor(cls)
        return va[:, 0], va[:, 1]
