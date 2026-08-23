# `datasets/` — data loading

Not to be confused with the repo-root `data/` directory, which holds torchvision's raw MNIST
cache. This is code; that is data.

| Module | What it is |
| --- | --- |
| `mnist_dataset.py` | `MNISTDataset` — torchvision MNIST with a train/validation split carved out using its own seeded generator, so the split is identical between a training run and a later evaluation regardless of what else has consumed the global RNG. |
| `multi_digit_dataset.py` | `MultiDigitMNISTDataset` builds the splits; `MultiDigitDataset` composes each example on access. Nothing off the shelf produces "an image of the number 407", so this picks a real MNIST sample per digit, resizes, jitters and occasionally rotates it, then pastes the digits side by side onto one canvas. |

Composing per access rather than materialising a fixed set means every epoch sees different
handwriting for the same number, which is the cheap augmentation this task needs.

Every number in range appears in the training split, deliberately: the task is reading digits
off a canvas, not extrapolating to unseen numbers, so holding numbers out would measure
something the model was never asked to do. Validation and test cover a subset of the same
numbers with freshly composed handwriting.
