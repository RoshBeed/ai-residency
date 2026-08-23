# RLHF / PPO

Container source for RLHF / PPO: a modular, from-scratch implementation of Proximal Policy
Optimization (generation, KL penalty against a frozen reference model, reward scoring, GAE
advantage estimation, and the policy/value gradient updates), each step its own independently
testable module, orchestrated by `PPOTrainer.ppo_step()` over a LoRA-wrapped SmolLM2-135M. Reward
scoring uses a pretrained OpenAssistant reward model
(`OpenAssistant/reward-model-deberta-v3-large-v2`). There is no multi-epoch training loop and no
reward-model or SFT training script anywhere in this repo — that's architecturally absent, not
trimmed for the container, and it's documented honestly rather than hidden. The full write-up —
architecture diagrams, the PPO-step derivation, and the real bugs found while getting the plumbing
correct (an `-inf` loss traced to sampling-time logits, a silent padding/attention-mask mismatch
that made the PPO ratio 2.38 before a single gradient step) — lives in
[`ai-residency-frontend/docs/rlhf-ppo.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/rlhf-ppo.md),
not here. This repo is container source, not documentation.

## Prerequisites

- Python >= 3.11, [`uv`](https://docs.astral.sh/uv/)
- Docker, for the container targets

## Commands

```bash
make install   # uv sync
make train     # one PPO step end to end on two example prompts — this repo has no epoch loop
make serve     # FastAPI generation service on :8000 (POST /generate, GET /health)
make test      # pytest
make check     # lint + strict mypy + tests
```

`make train` is not shorthand for "start training" — it's the literal single PPO step this
codebase implements, run once, writing `checkpoints/policy_after_one_ppo_step.pth`. That file is
the product of one gradient step, so `make serve` deliberately does not load it: doing so would
imply a policy-quality improvement that never happened. `serve` instead loads a freshly
LoRA-initialized copy of SmolLM2-135M — an honest generation-only demo of the real pipeline
(tokenizer setup with `padding_side='left'`, float32 model loading, LoRA wrapping, sampling
`generate()`), not a trained result. Every `/generate` response says so explicitly in its `note`
field.

## Docker

Two single-purpose images, both `FROM roshbeed/ai-residency-base:local` (built by
`ai-residency-repo-template`'s `Dockerfile.base` — build that once first with `make docker-build-base` over
there, or pull it once it's published). Neither ever touches git or needs a credential:
`ai-residency-common` is already installed in the base image by the time their own `uv sync` runs.

- **`Dockerfile.serve`** — the real FastAPI generation service, what the portfolio frontend and
  Kubernetes talk to.
- **`Dockerfile.train`** — runs one PPO step end to end; what a RunPod/AWS GPU host pulls and
  runs.

```bash
make docker-build-train    # docker build -f Dockerfile.train
make docker-build-serve    # docker build -f Dockerfile.serve

make docker-up              # serve, via docker compose
make docker-train           # train, in a container (profile-gated)
make docker-down            # stop and remove all compose services

make docker-push-train      # push roshbeed/ai-residency-week6-train (needs `docker login`)
make docker-push-serve      # push roshbeed/ai-residency-week6-serve
```

`docker-compose.yml` has two services: `serve` (default) and `train` (profile-gated, so a plain
`docker compose up` never kicks off a real gradient-update step). `checkpoints/` is bind-mounted
for `train` at run time rather than baked into the image.

## Environment variables

Everything degrades rather than failing, so a fresh clone still runs without any of these:

| Variable | Absent |
|---|---|
| `HF_TOKEN` | Checkpoints stay local, warning logged |
| `WANDB_API_KEY` | Run goes offline, sync later with `wandb sync` |

Any field on `PPOConfig` (`config/ppo_config.py`) can be overridden with an
`AI_RESIDENCY_<FIELD_NAME>` environment variable — e.g. `AI_RESIDENCY_MAX_NEW_TOKENS=20 make train` for a longer
generation, or `AI_RESIDENCY_KL_COEF=0.2` to change the reference-drift penalty. A `.env` in this repo or in
the parent directory (checked first, nearest wins) is loaded automatically; see `.env.example`.

## Known limitation

No trained PPO checkpoint exists anywhere in this repo. The policy, value and "SFT" model copies
are only LoRA-initialised — there's no supervised fine-tuning stage, no reward-model training
script (the reward signal comes from a pretrained OpenAssistant model instead), and `make train` is
a single PPO step rather than an epoch loop — so no policy-improving PPO run has happened. The PPO
mechanics themselves (generation, KL, advantage estimation, the clipped policy loss) are verified
correct end to end against a real reward model; what they're optimising is an untrained policy. See
the full write-up linked above for the exact evidence and the reasoning behind treating a real
training run as a scoped-out follow-up rather than a bug to fix here.

## Standards

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload, W&B
tracking — comes from [`ai-residency-common`](https://github.com/RoshBeed/ai-residency-common), pinned by tag. Ruff
(format and lint) and mypy run in strict mode with zero suppressions; the `[tool.ruff]`/
`[tool.mypy]` blocks are copied verbatim from `ai-residency-common` so configuration cannot drift between
repos in this portfolio.

## Full write-up

Architecture diagrams, the method (LoRA, the PPO step module by module), and experimental results
— including the three real bugs found and fixed while verifying this pipeline — live in
[`ai-residency-frontend/docs/rlhf-ppo.md`](https://github.com/RoshBeed/ai-residency-frontend/blob/main/docs/rlhf-ppo.md).
