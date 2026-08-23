# Whisper Fine-Tuning

Container source for Whisper fine-tuning: fine-tuning OpenAI's Whisper-tiny model, in two parts — a
live demo that fine-tunes Whisper on a single `(audio, text)` pair in one gradient step, and a data
pipeline that prepares speech-emotion datasets (CREMA-D, EmoDB) toward teaching Whisper to tag *how*
something was said alongside the transcript. Explanatory write-up, diagrams and results live in
[`ai-residency-frontend`](https://github.com/RoshBeed/ai-residency-frontend) now, not here — this repo is
container source, not documentation. Full write-up:
[`docs/whisper-finetuning.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/whisper-finetuning.md).

## Prerequisites

- Python >=3.11, [`uv`](https://docs.astral.sh/uv/).
- **`ffmpeg`, the system binary** (`brew install ffmpeg` / `apt install ffmpeg`) — `openai-whisper`
  shells out to it directly for audio decoding. The pip package of the same name is unrelated and
  does not provide it; `uv sync` will not install it. Both Dockerfiles install it via `apt-get`.
- A `data/audio.mp3` clip for `make train`/`make demo`/`make serve` to fine-tune on — gitignored,
  user-supplied. No live microphone in most environments; a synthesized stand-in works fine:
  ```bash
  say -o clip.aiff "My name is Rosh"        # macOS
  ffmpeg -y -i clip.aiff data/audio.mp3
  ```

## Commands

```bash
make install         # uv sync
make download-data   # fetch EmoDB (direct zip); CREMA-D needs a manual git-lfs clone
make prepare-data     # Part 2: organize by emotion -> manifest -> emotion tokens -> EDA
make train            # Part 1: fine-tune Whisper-tiny one step on data/audio.mp3, write a checkpoint
make demo             # print base-vs-fine-tuned transcript comparison for data/audio.mp3
make serve            # FastAPI on :8000
make test             # pytest
make check            # lint + strict mypy + tests
```

## Docker

Two images, both `FROM roshbeed/ai-residency-base` (built by `ai-residency-repo-template`'s `Dockerfile.base` — build
that once locally with `make -C ../ai-residency-repo-template docker-build-base`, or pull it once published).
Neither needs a git credential: `ai-residency-common` is already installed in the base image.

```bash
make docker-build-train   # training image, FROM roshbeed/ai-residency-base
make docker-build-serve   # serving (inference) image, FROM roshbeed/ai-residency-base
make docker-up             # serve, in a container, on :8000
make docker-train          # train, in a container (profile-gated)
make docker-push-train     # push roshbeed/ai-residency-week5-train (needs docker login)
make docker-push-serve     # push roshbeed/ai-residency-week5-serve (needs docker login)
```

`docker-compose.yml` mounts `./checkpoints` (serve, train) and `./data` (train) from the host.
`Dockerfile.serve` is what Kubernetes runs; `Dockerfile.train` is what a RunPod/AWS GPU host pulls
and runs for training.

## Environment variables

Both optional — everything degrades rather than failing without them.

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local; serving falls back to whatever's on local disk |
| `WANDB_API_KEY` | Training run goes offline, sync later with `wandb sync` |

Any `Config` field (in `config/`) can be overridden with
`AI_RESIDENCY_<FIELD_NAME>`, e.g. `AI_RESIDENCY_NUM_EPOCHS=1 make train` for a quick smoke run.

A `.env` in this repo (or a shared one in a parent directory — the lookup walks up from the working
directory) covers both. Copy `.env.example` to `.env` and fill in.

## Standards

Shared behavior (device selection, logging, checkpoint format, seeding, Hub upload, W&B tracking)
comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff and mypy run
in strict mode with zero suppressions, matching every other repo in this portfolio — see
[`ai-residency-repo-template`](https://github.com/RoshBeed/ai-residency-repo-template) for the canonical shape this
repo conforms to.
