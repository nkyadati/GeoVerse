import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", out_dim: int = 256):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size
        self.proj = nn.Linear(self.hidden_dim, out_dim)
        self._tok = AutoTokenizer.from_pretrained(model_name)
    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return self.proj(pooled)
