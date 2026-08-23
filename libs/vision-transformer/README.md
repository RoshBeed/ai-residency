# `libs/vision-transformer` — the transformer, from scratch

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

---

The rest of this file is the method these modules implement. It lives here rather than in either
service's write-up because both are built from exactly these blocks -- the same reason the code is
a library rather than a folder inside one of them.

## Patches as Tokens

An image doesn't come as a sequence, so the first job is to make it look like one. I cut a 28×28
grayscale MNIST digit into a grid of 4×4 patches (49 of them) and linearly project each flattened
patch to a 128-dimensional vector (`transformer/patch_embedding.py`).

I implemented that projection as a single strided `Conv2d(kernel=4, stride=4)` rather than a
literal flatten-then-matmul. The two are mathematically identical; the convolution kernel does the
sliding-window bookkeeping for free. A learnable `[CLS]` token gets prepended, sinusoidal
positional encodings are added, and the resulting 50×128 sequence goes through six transformer
encoder layers, the same setup as the original ViT [2]:

**Figure 4.** The original Vision Transformer architecture. Figure 1 from Dosovitskiy et al., *An
Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* [2] (arXiv:2010.11929) —
the paper this repo's classifier is built from.
![ViT architecture: an image is split into patches, linearly projected with a prepended learnable class embedding and position embeddings, fed through a Transformer Encoder, and read out through an MLP head into a class prediction; inset shows the Transformer Encoder's internal norm, multi-head attention, and MLP blocks with residual connections](assets/vit_architecture.png)

My classifier follows this diagram directly, at MNIST scale: a 28×28×1 digit is cut into 4×4
patches (`transformer/patch_embedding.py`, the strided `Conv2d(kernel=4, stride=4)` standing
in for "Linear Projection of Flattened Patches"), a learnable `[CLS]` token is prepended as shown,
sinusoidal position embeddings are added, and the result goes through 6 encoder layers
(`transformer/encoder.py`, matching the "Transformer Encoder" inset on the right: Norm →
Multi-Head Attention → residual → Norm → MLP → residual). I simplified only the paper's `MLP
Head`: this repo classifies from a single `Linear(128, 10)` rather than a multi-layer MLP, since
10 MNIST digit classes need far less capacity than ImageNet's class vocabulary.

## Multi-Head Self-Attention

Every encoder and decoder layer is built around the same scaled dot-product attention [1]:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

split across 8 heads (16-dim each for the classifier, 32-dim each for the seq2seq model; always
`heads × head_dim == d_model`). The actual implementation (`transformer/attention.py`):

```python
Q = self.W_q(query).view(batch_size, q_len, self.heads, self.head_dim).transpose(1, 2)
K = self.W_k(key).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)
V = self.W_v(value).view(batch_size, k_len, self.heads, self.head_dim).transpose(1, 2)

attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
attn_scores = torch.clamp(attn_scores, min=-100, max=100)  # numerical stability
attn_probs = torch.softmax(attn_scores, dim=-1)
out = torch.matmul(attn_probs, V)
```

The same `MultiHeadAttention` module is reused for three different jobs depending on what's passed
as `query`/`key`/`value`: encoder self-attention (all three are the same input), decoder masked
self-attention (all three are the decoder's own sequence, plus a causal mask), and decoder
cross-attention (query from the decoder, key/value from the encoder output).

## Transformer Encoder Block

I built each encoder layer post-norm, matching the original Transformer [1] rather than the
pre-norm variant common in newer models: attention (or feed-forward) output is added to the
residual stream *before* `LayerNorm` is applied, not after.

**Figure 5.** The encoder stack, cropped from the left half of Figure 1 in Vaswani et al.,
*Attention Is All You Need* [1] (arXiv:1706.03762) — the original Transformer paper this repo's
post-norm design follows, as opposed to ViT's own pre-norm variant shown inset in Figure 4 above.
![Encoder stack from the original Transformer paper: Input Embedding, plus sinusoidal Positional Encoding, feeds a stack of N encoder blocks, each a Multi-Head Attention sub-layer followed by Add & Norm, then a Feed Forward sub-layer followed by Add & Norm](assets/transformer_encoder_block.png)

`transformer/encoder.py` implements one of the `N×` blocks shown here: `Multi-Head Attention`
→ residual `Add` → `LayerNorm` → `Feed-Forward (Linear → ReLU → Linear)` → residual `Add` →
`LayerNorm`. The six stacked instances (`N=6`, matching the paper's own default) are the "encoder
layers" referenced throughout Section 3.1 above. This is deliberately the *post-norm* arrangement
the diagram shows (`LayerNorm` after each residual add, not before), which is why Figure 5 is
cropped from the original Transformer paper [1] rather than reused from ViT's own architecture
inset in Figure 4, which uses pre-norm instead.

