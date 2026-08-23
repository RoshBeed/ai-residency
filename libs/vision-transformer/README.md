# `libs/vision-transformer` — from-scratch building blocks

Composing these by hand is the point: `nn.TransformerEncoderLayer` would have hidden the
attention maths these projects exist to demonstrate.

Two services are assembled entirely from what is here —
[`vit-classifier`](../../services/vit-classifier) uses the encoder, patch embedding and positional
encodings; [`vit-seq2seq`](../../services/vit-seq2seq) adds the decoder. They are the same blocks
wired differently, which is exactly why this is a library rather than a folder inside one of
them.

| Module | What it is |
| --- | --- |
| `attention.py` | `MultiHeadAttention` — scaled dot-product attention. One tensor in is self-attention, three is cross-attention, so the same class serves the encoder, the decoder's masked self-attention and its cross-attention. |
| `encoder.py` | `Encoder` — self-attention then feed-forward, each with a residual. Normalisation comes *after* the residual add (post-norm, matching the original Transformer paper rather than ViT's pre-norm). |
| `decoder.py` | `Decoder` — masked self-attention, cross-attention over the encoder memory, then feed-forward. |
| `feedforward.py` | `FeedForward` — the position-wise two-layer block. |
| `patch_embedding.py` | `PatchEmbedding` — splits an image into non-overlapping patches with a strided convolution, which is exactly a per-patch linear projection and is faster than reshaping. |
| `positional.py` | `PositionalEncoding` (fixed sinusoidal, used by both encoders) and `LearnablePositionalEncoding` (used by the seq2seq decoder). |

One naming note: `MultiHeadAttention`'s projections are `W_q`/`W_k`/`W_v`/`W_o` rather than
snake_case. Those names are the state-dict keys the checkpoints were saved with, so
renaming them would make every existing checkpoint unloadable.
