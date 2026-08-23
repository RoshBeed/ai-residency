# `vit_classifier/` — the model

| Module | What it is |
| --- | --- |
| `classifier.py` | `VisionTransformer` — encoder-only, 28×28 input, prepends a learned `[CLS]` token and classifies from its final hidden state. |
| `trainer.py` | Training loop for the above: `python -m vit_classifier.trainer` |

Assembled from the blocks in [`libs/vision-transformer`](../../../libs/vision-transformer) and
shaped by the frozen `ClassifierArch` dataclass in [`../config/`](../config/), which also derives
the checkpoint's file name so the trainer and the API cannot disagree about it.

The trainer follows the portfolio's canonical shape: seed, build loaders, compute a baseline,
start tracking, then an epoch loop that logs metrics and publishes a checkpoint whenever
validation improves. The baseline is not decoration — 10-class MNIST accuracy sounds
self-evidently good until you notice that always guessing the commonest class already scores
around 11%.
