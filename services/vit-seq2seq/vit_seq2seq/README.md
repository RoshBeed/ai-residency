# `vit_seq2seq/` — the model

| Module | What it is |
| --- | --- |
| `seq2seq.py` | `Seq2SeqVisionTransformer` — encoder-decoder, an 84×84 three-digit composite in, a digit sequence out. No `[CLS]` token: the decoder cross-attends over the whole patch sequence, so a single pooled summary has nothing to do. |
| `trainer.py` | Training loop for the above: `python -m vit_seq2seq.trainer` |

Assembled from the blocks in [`libs/vision-transformer`](../../../libs/vision-transformer) and
shaped by the frozen `Seq2SeqArch` dataclass in [`../config/`](../config/), which also derives the
checkpoint's file name.

`generate` tracks completion per sequence within a batch. Stopping the whole batch at the first
`EOS` truncates every other sequence still decoding — including inside the trainer's own
evaluation loop, which makes the reported accuracy wrong rather than merely noisy.
