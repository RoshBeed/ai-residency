import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class Encoder(nn.Module):
    def __init__(self, heads, d_model, head_dim, ff_hidden_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.multi_head_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        attn_output = self.multi_head_attention(x)
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)
        residual = x
        ff_output = self.feed_forward(x)
        x = residual + self.dropout2(ff_output)
        x = self.norm2(x)
        return x 