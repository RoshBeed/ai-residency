# An encoder–decoder Vision Transformer that reads multi-digit numbers

## TL;DR

- **What it is:** an encoder–decoder Vision Transformer written from scratch in raw PyTorch. The
  encoder reads an image holding one, two or three handwritten digits; the decoder generates the
  number one digit at a time, autoregressively, with a causal mask and cross-attention, and stops
  when it emits EOS.
- **Result:** **98.53% exact-sequence accuracy** on 3,000 held-out composites of one to three
  digits, against 3.33% for always answering the commonest number in the split and 0.10% for
  guessing uniformly over 0–999. EOS lands in the right place 100% of the time, and the number
  the API returns is the same number: the two paths answer identically on all 3,000.

Classifying one digit needs a single label. Reading a number needs an ordered sequence, of a length
the model decides on, which is a different problem: the model has to know when it has finished. That
is what the decoder adds — a start token, a causal mask so each position only sees what came before
it, cross-attention onto the encoded image, and an EOS token it emits when the number is done.

## Quickstart

```bash
# one venv for the whole workspace
make install

# FastAPI on :8000, interactive docs at /docs
make serve SERVICE=vit-seq2seq

# read Usage first
make train SERVICE=vit-seq2seq
```

Checkpoints are resolved locally first, then from the Hugging Face Hub, so serving works on a host
that has never trained anything.

The transformer building blocks this model is made of — patch embedding, multi-head
self-attention, the encoder block and the positional encodings — are not duplicated here. They live
in [`libs/vision-transformer`](../../libs/vision-transformer), shared with
[the single-digit classifier](../vit-classifier), because both models are the same blocks wired
differently. Sections 3.1–3.3 below describe what that library implements.

## 1. Introduction

Classifying one digit needs a single label, and the [classifier](../vit-classifier) does that with
an encoder and a pooled `[CLS]` token. Reading a *number* is a different problem: the answer is an
ordered sequence whose length the model decides, so it has to know when it has finished.

That is what a decoder adds. It is seeded with a start token, masked so each position sees only
what came before it, cross-attends onto the encoded image at every step, and emits an EOS token
when the number is done. The encoder underneath is the same one the classifier uses — the
difference between the two services is entirely in what reads its output.

**Figure 1.** Recognizing several digits — the task this service solves, on the right.
![The two tasks: a single encoder mapping a 28x28 digit image to the label 4, and an encoder-decoder mapping a 3-digit image through a decoder seeded with <start> to the token sequence 0 2 3 1](assets/task_overview.png)

## 2. System Architecture

The encoder is the same stack the classifier uses, from the shared library. What this service adds
is the decoder half: a causal mask, cross-attention onto the encoded image, and a generation loop
that terminates per sequence.

**Figure 2.** From an image of digits to the number, and where each piece lives.
```mermaid
flowchart TD
    subgraph lib["libs/vision-transformer (shared with vit-classifier)"]
        patch["PatchEmbedding"]
        pos["PositionalEncoding"]
        enc["Encoder"]
        dec["Decoder"]
    end

    subgraph svc["services/vit-seq2seq"]
        model["Seq2SeqVisionTransformer<br/>encoder-decoder, vocab 12"]
        trainer["vit_seq2seq/trainer.py"]
        data["datasets/multi_digit_dataset.py<br/>one to three digits composited per sample"]
        api["serving/api.py<br/>POST /predict, /predict/stream"]
    end

    patch --> model
    pos --> model
    enc --> model
    dec --> model
    data --> trainer --> model
    trainer -->|"on new best"| hub[("Hugging Face Hub")]
    hub -->|"at startup"| api
    model --> api
    api --> ui["Streamlit frontend"]
```

`/predict/stream` relays each digit as a Server-Sent Event the moment the decoder emits it, so the
frontend shows the number being read one token at a time rather than appearing at once.

## 3. Method

### 3.1 The transformer blocks

Patch embedding, multi-head self-attention and the encoder block are shared with the sibling
service and live in [`libs/vision-transformer`](../../libs/vision-transformer), which is also
where they are explained: how an image becomes a sequence of patch tokens, how attention is
computed, and why the block is post-norm like the original Transformer paper rather than
pre-norm like ViT.

What follows is what this service adds on top of them.

### 3.2 Sequence Decoding: Encoder–Decoder ViT

The multi-digit task can't use a fixed-size classification head, because the output is a variable
number of digits (0 up to 999). Instead, `vit_seq2seq/seq2seq.py` pairs the same
patch-embedding encoder with a transformer decoder over a 12-token vocabulary: digits 0–9, plus
learned `START` (10) and `EOS` (11) tokens.

**Figure 6.** Encoder–decoder ViT for multi-digit sequence generation.
![Encoder-decoder architecture for the sequence task: a 3-digit image goes through linear projection of flattened patches and encoder blocks to produce a 16x64 encoder output; in parallel, the target tokens <start> 7 7 6 9 are tokenized and embedded (5x11), pass through decoder blocks with cross-attention (Q/K/V formulas shown) to the encoder output, and are projected via nn.Linear(11,13) into next-token logits over the digit/start/finish/pad vocabulary, producing 7 7 6 9 <finish>](assets/seq2seq_slide_architecture.png)

`vit_seq2seq/seq2seq.py` implements this pipeline. On the left, a patch-embedding encoder
(`Conv2d(k=16,s=16)` → 49×256 for this repo's real 112×112 composites, larger than the
figure's illustrative 16×64). On the right, a tokenizer/embedding stage over a 12-token vocabulary
— digits 0–9 plus learned `START`/`EOS` tokens, the figure's `<start>`/`<finish>`/`<pad>` table.
The two feed masked self-attention and cross-attention decoder blocks
(`torch.matmul(Q, K.transpose(-2,-1))` per block, matching the figure's `Q`/`Kᵀ`/`A` derivation)
that project back out to next-token logits.

The figure's worked example — feeding in `<start> 7 7 6 9` and reading out `7 7 6 9 <finish>` — is
the same teacher-forced training setup `seq2seq/trainer.py` uses: the whole target sequence goes to
the decoder at once, with a causal mask so position *i* can't see positions after it
(`create_causal_mask`, matching the "Masked Multi-Head Attention" block in the original decoder
[1]). At inference, `generate()` runs the same decoder autoregressively instead — no target
sequence to see, so it starts from `[START]` and feeds each predicted token back in to produce the
next one, one forward pass per digit, until `EOS` (`<finish>`) is produced.

### 3.3 The data has to exercise the EOS token

An architecture that supports a variable-length answer only demonstrates one if the training data
asks for one. Labels are the number's own digits, so `7` is one token long and `407` is three, and
each length contributes the same number of training examples.

That balance is deliberate. There are 10 one-digit numbers, 90 two-digit and 900 three-digit, so
drawing uniformly from `range(1000)` would make 90% of examples three digits long and "stop after
the third digit" would be the right answer almost always. Stopping has to cost something to get
wrong before it measures anything.

Padding is EOS repeated rather than a new `PAD` id. A pad token would widen the decoder's output
head and make every earlier checkpoint unloadable, which would destroy the before/after
comparison. The trainer masks every position after the first EOS out of the loss, so the repeats
are never trained against.

Each digit is pasted at MNIST's native 28px and then warped: rotation up to ±15°, shear up to
±12°, scale 0.75–1.15, plus a sub-cell translation, applied to every digit rather than a fraction
of them. The whole group is offset within whatever slack a shorter number leaves, so a two-digit
number is not always flush left. Without that the three cells of an evenly divided canvas are
memorisable, and the encoder can learn each cell's appearance instead of finding a glyph.

The canvas is 112px, which is 7×16. The patch embedding is a strided convolution, so any canvas
that is not a multiple of the patch size silently drops the remainder: at 84px the 5×5 grid covers
80 pixels and the last four columns never reach the model at all. 112px keeps every pixel inside a
patch, at 49 patch tokens instead of 25.

## 4. Implementation

Mapping the architecture from [Section 2](#2-system-architecture) onto the actual file layout:

```
libs/vision-transformer/     Shared with vit-classifier: attention, encoder, decoder, patch
                             embedding, positional encoding, feed-forward
services/vit-seq2seq/
├── config/                  Config(BaseConfig) plus Seq2SeqArch, the frozen architecture
│                            dataclass that also derives the checkpoint file name
├── vit_seq2seq/
│   ├── seq2seq.py           Seq2SeqVisionTransformer: encoder over patches, causal decoder
│   │                        with cross-attention, and autoregressive generate()
│   └── trainer.py           Training loop with teacher forcing, warmup, label smoothing
├── datasets/
│   └── multi_digit_dataset.py  Composites one to three random MNIST digits onto a 112x112
│                            canvas per sample, each digit warped and jittered, on the fly
├── serving/api.py           FastAPI app: /predict, /predict/stream (per-digit SSE), /health
└── tests/                   Generation tests, including the regression test for batch-wide
                             EOS truncation, plus shape checks
```

## 5. Usage

Figure 3 ([Section 2](#2-system-architecture)) shows the end-to-end flow. Trained checkpoints
live on the Hugging Face Hub and are pulled on first use, so serving works without training
anything first and nothing large is committed here. A `Makefile` wraps every stage below with a consistent set of targets; the raw command each
target actually runs is kept underneath it, since that's what's really executing.

```bash
make install   # what this runs: uv sync   (installs from pyproject.toml + committed uv.lock)
make serve SERVICE=vit-seq2seq     # what this runs: uv run python -m serving.api   (FastAPI on :8000)
make -C services/vit-seq2seq evaluate   # what this runs: uv run python -m vit_seq2seq.evaluate
                                        # every number in Section 6, from the shipped checkpoint
make -C services/vit-seq2seq evaluate-sampled   # the same, decoding by sampling
                                        # what the greedy decode rule is worth, 21.5 points
make check     # lint + type check + tests
```

To retrain from scratch instead:

```bash
make train SERVICE=vit-seq2seq-classifier  # what this runs:
                        #   uv run python -m vit_classifier.trainer
                        # trains on MNIST, saves to checkpoints/
make train SERVICE=vit-seq2seq-seq2seq     # what this runs:
                        #   uv run python -m vit_seq2seq.trainer
                        # trains on composited multi-digit images, ~10min an epoch on M1 Max/MPS
```

`make train SERVICE=vit-seq2seq` is an alias for `make train SERVICE=vit-seq2seq-classifier` (the faster of the two).

A `notebooks/walkthrough.ipynb` runs the happy path end to end — pulling both checkpoints from
the Hub and running real inference — without retraining anything. Its first cell chdirs back
to the repo root so relative paths (checkpoints, MNIST cache) resolve the same way `make serve SERVICE=vit-seq2seq`'s
do; the package itself is installed, so nothing needs to be put on the import path.

**Docker Compose (Linux/CUDA servers):** one `Dockerfile.serve` and one `Dockerfile.train` at the
repository root, parameterised by `--build-arg SERVICE=`, so this service has no Dockerfile of its
own. The serving image builds `FROM roshb/ai-residency-base-serve` and the training image from the
CUDA base. `uv.lock` already pins
`torch`/`torchvision` to the CPU-only PyTorch wheel index, which otherwise resolves an unpinned
`torch` to a multi-GB CUDA build on `linux/arm64` — a real GPU host would need its own
CUDA-enabled lockfile rather than overriding this one at build time.

On Apple Silicon, that CPU-only pin isn't just about avoiding a CUDA download. Docker Desktop on
macOS runs containers inside a Linux VM, and that VM has no path to the host's Metal GPU at all
(no MPS passthrough exists, unlike Linux's native NVIDIA passthrough), so a container always
reports `"device": "cpu"` here no matter what the host itself can do. `make serve SERVICE=vit-seq2seq` /
`make train SERVICE=vit-seq2seq-classifier` / `make train SERVICE=vit-seq2seq-seq2seq` run natively and get real MPS acceleration; the
same targets under `make docker-*` are CPU-only on this hardware by construction, not a bug to
chase. Use the native `make` targets for real Apple Silicon speed, and Docker Compose for a
CPU-portable, Linux/CUDA-ready deployment story.

```bash
make -C deploy up               # what this runs: docker compose up serve
docker compose --profile train run --rm train-vit-seq2seq
make -C deploy up  # then: docker compose --profile train run --rm train-vit-seq2seq STAGE=seq2seq  # the slower seq2seq trainer
make -C deploy down             # what this runs: docker compose down
```

**API:**
- `GET /` and `GET /health` — the same liveness check, reporting which models are loaded and
  on which device. `/health` is the portfolio-wide convention the hub and the Helm probes use.
- `GET /models` — list available models
- `POST /predict` — predict with `model_type: "vit"` or `"seq2seq"`
- `POST /predict/vit`, `POST /predict/seq2seq` — model-specific shortcuts
- `POST /predict/both` — run both models on the same image for comparison

## 6. Experimental Results

Generation tracks completion per sequence within a batch, so shorter and longer targets in the
same batch each run to their own `EOS` independently. That is worth stating precisely because the
alternative, stopping the whole batch at the first `EOS`, silently truncates every other sequence
still decoding, including inside the trainer's own evaluation loop, which makes the reported
accuracy wrong rather than merely noisy.

`make evaluate` scores the shipped checkpoint on 3,000 held-out composites, balanced across one,
two and three digits, with freshly composed handwriting.

| metric | result |
|---|--:|
| **exact-sequence accuracy** | **98.53%** |
| commonest-number baseline | 3.33% |
| uniform guess over 0–999 | 0.10% |
| one digit | 99.20% |
| two digits | 98.50% |
| three digits | 97.90% |
| EOS in the right place | 100.00% |
| per-digit, aligned by position | 99.27% |

Exact-sequence is the strict metric: two correct digits out of three scores zero. Sequences are
compared digit by digit rather than read back as an integer, because an integer comparison cannot
tell a dropped zero from a correct answer. Composites are generated per access, so a second pass
scores a different draw of handwriting and the figure moves by about two tenths of a point.

Per-digit accuracy is 99.27% and flat across lengths (99.20% on one-digit targets, 99.25% on two,
99.30% on three), so a third digit is read as well as a lone one. That is the figure to read first:
it says the encoder is finding and reading glyphs rather than reading three known slots, with
rotation, shear, scale and translation applied to every digit. Exact-sequence falls with length
only because three independent reads have to all be right at once.

Every sequence stops in the right place, so what is left is a misread digit rather than a decoder
that cannot count.

### What the decode rule was worth

Both decode loops drew each token from the softmax with `torch.multinomial`. For a task with one
right answer that trades accuracy for variety nothing wants: an image of 206 depicts 206, and
sampling at temperature 1 picks a low-probability token every so often, losing the whole sequence
with it. Greedy decoding is the default now, and `sample=True` is opt-in.

| decode rule | exact-sequence | EOS correct | per-digit |
|---|--:|--:|--:|
| sampled, temperature 1 | 77.03% | 90.23% | 90.60% |
| **greedy** | **98.53%** | **100.00%** | **99.27%** |

Same checkpoint, same split, 21.5 points — `make evaluate-sampled` is the first row and
`make evaluate` the second, so both are reproducible from one command. The trainer's own
validation pass sampled as well, so the number it reported each epoch agreed with the number
`make evaluate` reported afterwards, and two agreeing numbers look like a verified one.

EOS is where sampling does most of its damage, and the length spread says how: decoding 1,000
one-digit targets, sampling emits nothing 11 times, two digits 71 times and three digits 3 times.
Each of those is one unlucky draw on a distribution the model got right.

### The number above is the number the service returns

Training builds a composite in memory; the API decodes a PNG someone posted. Those are two
preprocessing paths for one model, and a model is only as good as the worse one. Scored on the
same 3,000 composites, routing one copy through a real PNG encode and the service's own
`preprocess_image`:

| path | exact-sequence | correct length | per-digit |
|---|--:|--:|--:|
| the loader's tensors | 98.23% | 100.00% | 99.12% |
| re-encoded as PNG and preprocessed as the API does | 98.23% | 100.00% | 99.12% |

Not close — identical, with the same answer on all 3,000 and a largest per-pixel difference of
0.006, which is 8-bit PNG quantisation. The dataset normalises once, on the finished composite,
with the same mean and standard deviation the API applies, and
`test_a_composite_survives_the_round_trip_the_api_puts_it_through` fails if that stops being
true. (Both figures here are one draw of handwriting below the table above; composites are
generated per access.)

Verified through the published image rather than only in-process: one held-out composite per digit
count, posted to `/predict/seq2seq`, reads `0`, `55` and `206` with per-digit confidence around
0.92, and the same request five times running returns the same answer five times.

### Against the fixed-length predecessor

The previous checkpoint trained on labels built with `str(number).zfill(3)`, so every target was
three digits and EOS always landed at position 4. Scored on the variable-length task, greedy, the
same way:

| | exact-sequence | one digit | two digits | three digits | correct length |
|---|--:|--:|--:|--:|--:|
| fixed-length, 84px | 23.20% | 0.00% | 0.00% | 69.60% | 33.33% |
| **this checkpoint** | **98.53%** | **99.20%** | **98.50%** | **97.90%** | **100.00%** |

Zero on one- and two-digit numbers is not a perception failure. It emits three digits whatever it
sees, because that is the only thing its training data ever asked for.

Real transcript from the shipped checkpoint:

```
target   2 -> generated   2   MATCH
target  72 -> generated  72   MATCH
target 313 -> generated 313   MATCH
target 767 -> generated 769   miss
target  30 -> generated  90   miss
```

## 7. Related Work

- [the dual-encoder retrieval project](https://github.com/RoshBeed/ai-residency/tree/main/services/vector-search-retrieval) —
  the self-attention-free encoder, superseded here by a from-scratch transformer.
- [the image captioning project](https://github.com/RoshBeed/ai-residency/tree/main/services/multimodal-captioning) —
  the same encoder-decoder ViT pattern from [Section 3.2](#32-sequence-decoding-encoderdecoder-vit),
  applied to natural-language captions instead of digit sequences.

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., &
Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS.

[2] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T.,
Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2021). *An Image
is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.

[3] LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). *Gradient-Based Learning Applied to
Document Recognition.* Proceedings of the IEEE, 86(11). — the original MNIST dataset used
throughout this project.

## Acknowledgments

Built with Joao Esteves, Hikaru Tsujimura, and Felipe Lavratti. Figures 4 and 5 are reproduced directly
from Dosovitskiy et al. [2] and Vaswani et al. [1] respectively.

## Dependencies

PyTorch · torchvision · FastAPI · Streamlit · streamlit-drawable-canvas · OpenCV · Docker Compose ·
Terraform (GPU server provisioning)
