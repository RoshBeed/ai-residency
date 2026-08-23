# `models/` — the two models

Both are assembled from the blocks in [`../transformer/`](../transformer/) and shaped by the
frozen architecture dataclasses in [`../config.py`](../config.py).

| Module | What it is |
| --- | --- |
| `classifier/classifier.py` | `VisionTransformer` — encoder-only, 28×28 input, prepends a learned `[CLS]` token and classifies from its final hidden state. |
| `classifier/trainer.py` | Training loop for the above. `python -m models.classifier.trainer` |
| `seq2seq/seq2seq.py` | `Seq2SeqVisionTransformer` — encoder-decoder, 84×84 three-digit composite in, digit sequence out. No `[CLS]` token: the decoder cross-attends over the whole patch sequence, so a single pooled summary has nothing to do. |
| `seq2seq/trainer.py` | Training loop for the above. `python -m models.seq2seq.trainer` |

Both trainers follow the portfolio's canonical shape: seed, build loaders, compute a baseline,
start tracking, then an epoch loop that logs metrics and publishes a checkpoint whenever
validation improves. The baseline is not decoration — 10-class MNIST accuracy sounds
self-evidently good until you notice that always guessing the commonest class already scores
around 11%.
