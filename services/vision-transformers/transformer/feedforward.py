import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x) 