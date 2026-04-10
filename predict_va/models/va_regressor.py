# models/va_regressor.py

import torch
import torch.nn as nn
from transformers import AutoModel
import torch.nn.functional as F

class VARegressor(nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )

        hidden = self.encoder.config.hidden_size

        self.valence_head = nn.Linear(hidden, 1)
        self.arousal_head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls = outputs.last_hidden_state[:, 0, :]

        valence = torch.tanh(self.valence_head(cls))      # [-1, 1]
        arousal = torch.sigmoid(self.arousal_head(cls))  # [0, 1]

        return valence.squeeze(-1), arousal.squeeze(-1)
