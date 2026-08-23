"""Encoder-decoder Vision Transformer that reads a multi-digit image as a sequence."""

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from config import Seq2SeqArch
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.patch_embedding import PatchEmbedding
from transformer.positional import (
    LearnablePositionalEncoding,
    PositionalEncoding,
)

__all__ = ["Seq2SeqVisionTransformer"]


class Seq2SeqVisionTransformer(nn.Module):
    """Encodes an 84x84 three-digit composite and decodes the digits one at a time.

    There is no [CLS] token here, unlike the single-digit classifier: the decoder
    cross-attends over the whole patch sequence, so there is nothing for a single
    pooled summary vector to do.
    """

    def __init__(self, arch: Seq2SeqArch) -> None:
        """Build the model.

        Args:
            arch: Architecture the weights are shaped by.
        """
        super().__init__()
        self.arch = arch
        self.vocab_size = arch.vocab_size
        self.start_token = arch.start_token
        self.eos_token = arch.eos_token
        self.n_patches = (arch.img_size // arch.patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            img_size=arch.img_size,
            patch_size=arch.patch_size,
            in_channels=arch.in_channels,
            embed_dim=arch.d_model,
        )
        self.encoder_pos_encoding = PositionalEncoding(
            d_model=arch.d_model,
            max_seq_length=self.n_patches,
        )
        self.encoder_layers = nn.ModuleList(
            [
                Encoder(
                    heads=arch.heads,
                    d_model=arch.d_model,
                    head_dim=arch.head_dim,
                    ff_hidden_dim=arch.ff_hidden_dim,
                    dropout=arch.dropout,
                )
                for _ in range(arch.num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(arch.d_model)

        self.token_embedding = nn.Embedding(arch.vocab_size, arch.d_model)
        self.decoder_pos_encoding = LearnablePositionalEncoding(
            d_model=arch.d_model,
            max_seq_length=arch.max_seq_length,
        )
        self.decoder_layers = nn.ModuleList(
            [
                Decoder(
                    heads=arch.heads,
                    d_model=arch.d_model,
                    head_dim=arch.head_dim,
                    ff_hidden_dim=arch.ff_hidden_dim,
                    dropout=arch.dropout,
                )
                for _ in range(arch.num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(arch.d_model)
        self.output_projection = nn.Linear(arch.d_model, arch.vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise parameters, and reset every LayerNorm to the identity.

        Small-std initialisation of the projections is what stopped this model
        producing NaN losses in its first few steps.
        """
        nn.init.normal_(self.patch_embedding.proj.weight, std=0.02)
        # Conv2d's bias is optional by construction, so narrow before touching it.
        patch_bias = self.patch_embedding.proj.bias
        if patch_bias is not None:
            nn.init.zeros_(patch_bias)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def create_causal_mask(self, seq_length: int, device: torch.device) -> torch.Tensor:
        """Build the decoder's causal mask.

        Args:
            seq_length: Length of the decoder sequence.
            device: Device to allocate the mask on.

        Returns:
            Boolean mask of shape `(seq_length, seq_length)` where True means a
            position must not be attended to.
        """
        mask = torch.triu(torch.ones(seq_length, seq_length, device=device), diagonal=1)
        return mask.bool()

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images into patch memory.

        Args:
            images: Shape `(batch, in_channels, img_size, img_size)`.

        Returns:
            Shape `(batch, n_patches, d_model)`.
        """
        x = self.patch_embedding(images)
        x = self.encoder_pos_encoding(x)
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
        encoded: torch.Tensor = self.encoder_norm(x)
        return encoded

    def decode(
        self,
        encoder_output: torch.Tensor,
        target_sequence: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Decode token logits for a target sequence.

        Args:
            encoder_output: Encoder memory from `encode`.
            target_sequence: Token ids of shape `(batch, seq_len)`.
            target_mask: Optional mask over the encoder positions.

        Returns:
            Logits of shape `(batch, seq_len, vocab_size)`.
        """
        seq_length = target_sequence.size(1)

        x = self.token_embedding(target_sequence)
        x = self.decoder_pos_encoding(x)
        causal_mask = self.create_causal_mask(seq_length, x.device)

        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, encoder_output, tgt_mask=causal_mask, src_mask=target_mask)

        x = self.decoder_norm(x)
        logits: torch.Tensor = self.output_projection(x)
        return logits

    def forward(
        self,
        images: torch.Tensor,
        target_sequences: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Teacher-forced forward pass used during training.

        Args:
            images: Shape `(batch, in_channels, img_size, img_size)`.
            target_sequences: Token ids of shape `(batch, seq_len)`.
            target_mask: Optional mask over the encoder positions.

        Returns:
            Logits of shape `(batch, seq_len, vocab_size)`.

        Raises:
            ValueError: If the tensor ranks, batch sizes or token ids are invalid.
        """
        if images.dim() != 4:
            raise ValueError(f"Expected 4D image tensor, got {images.dim()}D")
        if target_sequences.dim() != 2:
            raise ValueError(f"Expected 2D target tensor, got {target_sequences.dim()}D")
        if images.size(0) != target_sequences.size(0):
            raise ValueError(
                f"Batch size mismatch: images {images.size(0)}, targets {target_sequences.size(0)}"
            )
        if int(target_sequences.max().item()) >= self.vocab_size:
            raise ValueError(
                f"Target contains invalid token: "
                f"{int(target_sequences.max().item())} >= {self.vocab_size}"
            )

        return self.decode(self.encode(images), target_sequences, target_mask)

    def generate(
        self,
        images: torch.Tensor,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively generate a token sequence per image.

        Args:
            images: Shape `(batch, in_channels, img_size, img_size)`.
            start_token: Token the sequence is primed with.
            max_length: Maximum sequence length including the start token.
            temperature: Sampling temperature.

        Returns:
            Token ids of shape `(batch, generated_len)`, where `generated_len` is at
            most `max_length`.
        """
        batch_size = images.size(0)
        device = images.device

        encoder_output = self.encode(images)
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)

        # Completion is tracked per sequence, not per batch. This loop used to break
        # the instant *any* sequence produced EOS, which truncated every other still
        # generating sequence in the batch: one fast-finishing row dragged the rest
        # down with it. That corrupted every batched evaluation while leaving the
        # single-image API path (batch_size 1, where "any" and "all" agree) correct,
        # so it hid in plain sight. `tests/test_week3.py` guards the fix.
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            logits = self.decode(encoder_output, generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Finished sequences keep emitting EOS rather than sampling and drifting.
            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.eos_token),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token)

            if bool(finished.all()):
                break

        return generated

    def generate_with_callback(
        self,
        images: torch.Tensor,
        start_token: int,
        max_length: int,
        temperature: float = 1.0,
        on_digit: Callable[[int, str, float], None] | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate, calling `on_digit(step, digit_str, prob)` per token.

        Same semantics as `generate` but fires a callback at each decoded digit so an
        SSE endpoint can stream the sequence live rather than returning it all at once.

        Args:
            images: Shape `(batch, in_channels, img_size, img_size)`.
            start_token: Token the sequence is primed with.
            max_length: Maximum sequence length including the start token.
            temperature: Sampling temperature.
            on_digit: Called per generated token as `on_digit(step, digit_str, prob)`
                with *step* zero-indexed. May be None.

        Returns:
            Token ids of shape `(batch, generated_len)`.
        """
        batch_size = images.size(0)
        device = images.device

        encoder_output = self.encode(images)
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            logits = self.decode(encoder_output, generated)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, self.eos_token),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)
            finished = finished | (next_token.squeeze(1) == self.eos_token)

            if on_digit is not None:
                for b in range(batch_size):
                    token_id: int = int(next_token[b].item())
                    prob: float = float(probs[b, token_id].item())
                    digit_str: str = (
                        "" if token_id in (start_token, self.eos_token) else str(token_id)
                    )
                    on_digit(step, digit_str, prob)

            if bool(finished.all()):
                break

        return generated
