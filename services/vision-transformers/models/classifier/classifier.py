import torch
import torch.nn as nn
from transformer.patch_embedding import PatchEmbedding
from transformer.positional import PositionalEncoding
from transformer.encoder import Encoder

class VisionTransformer(nn.Module):
    def __init__(self, config):
        super(VisionTransformer, self).__init__()
        
        # Extract parameters from config
        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.num_classes = config["num_classes"]
        self.d_model = config["d_model"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.heads = config["heads"]
        self.head_dim = config["head_dim"]
        self.ff_hidden_dim = config["ff_hidden_dim"]
        self.dropout = config["dropout"]
        
        self.n_patches = (self.img_size // self.patch_size) ** 2
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.d_model
        )
        
        # Positional Encoding for encoder (CLS token + patches)
        self.encoder_pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_length=self.n_patches + 1
        )
        
        # Encoder Layers
        self.encoder_layers = nn.ModuleList([
            Encoder(
                heads=self.heads,
                d_model=self.d_model,
                head_dim=self.head_dim,
                ff_hidden_dim=self.ff_hidden_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_encoder_layers)
        ])
        
        # Final layer norm
        self.layer_norm = nn.LayerNorm(self.d_model)
        
        # Classification head with dropout
        self.dropout = nn.Dropout(self.dropout)
        self.classifier = nn.Linear(self.d_model, self.num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.patch_embedding.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embedding.proj.bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(self, x):
        # Input validation
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor, got {x.dim()}D")
        
        expected_size = (self.img_size, self.img_size)
        if x.size(-2) != expected_size[0] or x.size(-1) != expected_size[1]:
            raise ValueError(f"Expected input size {expected_size}, got {x.size(-2)}x{x.size(-1)}")
        
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {x.size(1)}")
        
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Prepend [CLS] token to patches
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.encoder_pos_encoding(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Final layer norm
        x = self.layer_norm(x)
        
        # Extract CLS token for classification
        cls_output = x[:, 0]
        
        # Apply dropout before classification for better regularization
        cls_output = self.dropout(cls_output)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits 