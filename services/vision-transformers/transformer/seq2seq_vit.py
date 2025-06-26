import torch
import torch.nn as nn
from .patch_embedding import PatchEmbedding
from .positional import PositionalEncoding, LearnablePositionalEncoding
from .encoder import Encoder
from .decoder import Decoder

class Seq2SeqVisionTransformer(nn.Module):
    def __init__(self, config):
        super(Seq2SeqVisionTransformer, self).__init__()
        
        # Extract parameters from config
        self.img_size = config["img_size"]
        self.patch_size = config["patch_size"]
        self.in_channels = config["in_channels"]
        self.d_model = config["d_model"]
        self.num_encoder_layers = config["num_encoder_layers"]
        self.num_decoder_layers = config["num_decoder_layers"]
        self.heads = config["heads"]
        self.head_dim = config["head_dim"]
        self.ff_hidden_dim = config["ff_hidden_dim"]
        self.dropout = config["dropout"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_length = config["max_seq_length"]
        
        self.n_patches = (self.img_size // self.patch_size) ** 2
        
        # Patch embedding for encoder
        self.patch_embedding = PatchEmbedding(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.d_model
        )
        
        # Positional Encoding for encoder (patches only)
        self.encoder_pos_encoding = PositionalEncoding(
            d_model=self.d_model,
            max_seq_length=self.n_patches
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
        
        # Final encoder layer norm
        self.encoder_norm = nn.LayerNorm(self.d_model)
        
        # Token embedding for decoder
        self.token_embedding = nn.Embedding(self.vocab_size, self.d_model)
        
        # Positional Encoding for decoder
        self.decoder_pos_encoding = LearnablePositionalEncoding(
            d_model=self.d_model,
            max_seq_length=self.max_seq_length
        )
        
        # Decoder Layers
        self.decoder_layers = nn.ModuleList([
            Decoder(
                heads=self.heads,
                d_model=self.d_model,
                head_dim=self.head_dim,
                ff_hidden_dim=self.ff_hidden_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_decoder_layers)
        ])
        
        # Final decoder layer norm
        self.decoder_norm = nn.LayerNorm(self.d_model)
        
        # Output projection
        self.output_projection = nn.Linear(self.d_model, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights properly to prevent NaN issues"""
        # Initialize patch embedding
        nn.init.normal_(self.patch_embedding.proj.weight, std=0.02)
        nn.init.zeros_(self.patch_embedding.proj.bias)
        
        # Initialize token embedding
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize learnable positional encoding (already done in the class)
        
        # Initialize encoder layer norms (only norm1 and norm2)
        for layer in self.encoder_layers:
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0.0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0.0)
        
        # Initialize decoder layer norms (only norm1, norm2, and norm3)
        for layer in self.decoder_layers:
            nn.init.constant_(layer.norm1.weight, 1.0)
            nn.init.constant_(layer.norm1.bias, 0.0)
            nn.init.constant_(layer.norm2.weight, 1.0)
            nn.init.constant_(layer.norm2.bias, 0.0)
            nn.init.constant_(layer.norm3.weight, 1.0)
            nn.init.constant_(layer.norm3.bias, 0.0)
        
        # Initialize encoder and decoder norms
        nn.init.constant_(self.encoder_norm.weight, 1.0)
        nn.init.constant_(self.encoder_norm.bias, 0.0)
        nn.init.constant_(self.decoder_norm.weight, 1.0)
        nn.init.constant_(self.decoder_norm.bias, 0.0)
        
    def create_causal_mask(self, seq_length, device):
        """Create a lower-triangular causal mask for decoder self-attention"""
        # 1s in allowed positions, 0s in masked (future) positions
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device)).unsqueeze(0).unsqueeze(0)
        # Shape: (1, 1, seq_len, seq_len) for broadcasting
        return mask
        
    def encode(self, x):
        """Encode the input image"""
        batch_size = x.size(0)
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # Add positional encoding
        x = self.encoder_pos_encoding(x)
        
        # Pass through encoder layers
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        
        # Final layer norm
        x = self.encoder_norm(x)
        
        return x
        
    def decode(self, encoder_output, target_sequence, target_mask=None):
        """Decode the target sequence"""
        batch_size, seq_length = target_sequence.size()
        
        # Token embedding
        x = self.token_embedding(target_sequence)
        
        # Add positional encoding
        x = self.decoder_pos_encoding(x)
        
        # Create causal mask for self-attention
        causal_mask = self.create_causal_mask(seq_length, x.device)
        
        # Pass through decoder layers with correct mask
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, tgt_mask=causal_mask, src_mask=target_mask)
        
        # Final layer norm
        x = self.decoder_norm(x)
        
        # Output projection
        logits = self.output_projection(x)
        
        return logits
        
    def forward(self, images, target_sequences, target_mask=None):
        """
        Forward pass for training
        
        Args:
            images: (batch_size, channels, height, width)
            target_sequences: (batch_size, seq_length)
            target_mask: Optional mask for target sequences
        """
        # Encode images
        encoder_output = self.encode(images)
        
        # Decode target sequences
        logits = self.decode(encoder_output, target_sequences, target_mask)
        
        return logits
        
    def generate(self, images, start_token, max_length, temperature=1.0):
        """
        Generate sequence for inference
        
        Args:
            images: (batch_size, channels, height, width)
            start_token: Token to start generation with
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
        """
        batch_size = images.size(0)
        device = images.device
        
        # Encode images
        encoder_output = self.encode(images)
        
        # Initialize sequence with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        for _ in range(max_length - 1):
            # Decode current sequence
            logits = self.decode(encoder_output, generated)
            
            # Get next token probabilities
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax with numerical stability
            next_token_logits = torch.clamp(next_token_logits, min=-100, max=100)
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Ensure probabilities are valid
            probs = torch.clamp(probs, min=1e-8, max=1.0)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            
            # Sample next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS token is generated
            if (next_token == 1).any():  # Assuming EOS token is 1
                break
                
        return generated 