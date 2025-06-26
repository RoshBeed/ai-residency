import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, head_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.head_dim = head_dim
        self.output_dim = heads * head_dim

        assert d_model % heads == 0, f"d_model ({d_model}) must be divisible by heads ({heads})"
        self.head_dim = d_model // heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key=None, value=None, mask=None):
        if key is None and value is None:
            key = query
            value = query

        batch_size, q_len, _ = query.size()
        k_len = key.size(1)

        Q = self.W_q(query).view(batch_size, q_len, self.heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)

        # Compute attention scores with numerical stability
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Clamp attention scores to prevent extreme values
        attn_scores = torch.clamp(attn_scores, min=-100, max=100)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax with numerical stability
        attn_probs = torch.softmax(attn_scores, dim=-1)
        
        # Ensure probabilities are valid
        attn_probs = torch.clamp(attn_probs, min=1e-8, max=1.0)
        attn_probs = attn_probs / attn_probs.sum(dim=-1, keepdim=True)
        
        attn_probs = self.dropout(attn_probs)
        out = torch.matmul(attn_probs, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, q_len, self.d_model)
        out = self.W_o(out)
        return out 