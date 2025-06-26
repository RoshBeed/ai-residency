import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class Decoder(nn.Module):
    def __init__(self, heads, d_model, head_dim, ff_hidden_dim, dropout=0.1):
        super(Decoder, self).__init__()
        # Self-attention (masked)
        self.self_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Cross-attention (encoder-decoder attention)
        self.cross_attention = MultiHeadAttention(heads, d_model, head_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None):
        # Self-attention with residual connection
        residual = x
        attn_output = self.self_attention(x, mask=tgt_mask)
        x = residual + self.dropout1(attn_output)
        x = self.norm1(x)

        # Cross-attention with residual connection
        residual = x
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, mask=src_mask)
        x = residual + self.dropout2(cross_attn_output)
        x = self.norm2(x)

        # Feed forward with residual connection
        residual = x
        ff_output = self.feed_forward(x)
        x = residual + self.dropout3(ff_output)
        x = self.norm3(x)

        return x 