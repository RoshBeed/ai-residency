# PPO Language Model Fine-Tuning

## TL;DR

- **What it is:** a from-scratch, modular PPO implementation (generation, KL penalty, reward, advantage, policy/value updates) with LoRA, running end to end against a real pretrained reward model.
- **What it demonstrates:** every PPO mechanic — generation, KL penalty, GAE advantage, clipped policy and value losses — verified correct with real gradient steps and a full test suite.
- **Result:** 60 PPO steps on real TL;DR prompts move held-out mean reward by **-0.04 under greedy
  decoding and +0.09 under seeded sampling** — the two metrics disagree about the sign, because the
  policy barely moved: the LoRA adapter's effective update is 1.26e-04 of the weights it is added
  to. The loop runs end to end on real data; 60 steps at `learning_rate=1e-5` is not enough to
  change what the policy writes (§6).
- **What I'd do next:** not a bigger learning rate — scaling the update 1000x makes reward
  monotonically worse (§6), so that is ruled out rather than untried. The suspects left are in the
  signal: advantage estimates from 4-prompt batches, 10-token completions, and an SFT stage so the
  KL penalty measures drift from a supervised checkpoint rather than from the base model.

I built a modular, from-scratch implementation of Proximal Policy Optimization (PPO) [1] for fine-tuning a
language model against a learned reward model: the RLHF training loop [2, 3] used to align GPT-3
into InstructGPT [4], broken into one small, independently testable module per step: generation,
KL penalty, reward computation, advantage estimation, and the policy/value updates themselves.

## Quickstart

```bash
# one venv for the whole workspace
make install

# FastAPI on :8000, interactive docs at /docs
make serve SERVICE=rlhf-ppo

# read Usage first -- some stages run for hours
make train SERVICE=rlhf-ppo
```

Checkpoints are resolved locally first, then from the Hugging Face Hub, so serving works on a host that has never trained anything.

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Method](#3-method)
4. [Implementation](#4-implementation)
5. [Usage](#5-usage)
6. [Experimental Results](#6-experimental-results)
7. [Related Work](#7-related-work)
8. [References](#references)
9. [Acknowledgments](#acknowledgments)
10. [Dependencies](#dependencies)

## 1. Introduction

The brief was explicit about the shape of the pipeline: take a base model (Qwen3), do
supervised fine-tuning (SFT) on a summarization dataset, train a reward model on human preference
pairs, then train a policy model with PPO to maximize that reward. That's the same three-stage
recipe OpenAI used for their TL;DR summarization work [3]. I kept that framing here because this
repo's example prompts are literally in that dataset's format
(`SUBREDDIT: ... TITLE: ... POST: ... TL;DR:`).

**Figure 1.** The three-stage RLHF pipeline: collect human preference data, train a reward model
on it, then optimize a policy against that reward model with RL. Figure 2 from Stiennon et al.
2020 [3].
![Diagram of the human feedback, reward model training, and policy training procedure, from Stiennon et al. 2020](assets/fig1-rlhf-three-stage.png)

This repo's scope is stage 3: given a `PolicyModel`, a `RewardModel`, and a frozen reference
("SFT") model, run the actual PPO update loop: generate, score, penalize drift from the reference,
estimate advantages, update. Concretely, `PPOTrainer.ppo_step()` (Figure 4 below) is one iteration
of the diagram's third panel: a batch of prompts stands in for "a new post sampled from the
dataset," `PolicyModel.generate()` is "policy π generates a summary," `RewardComputer` is the
reward-model scoring arrow, and the value/policy gradient steps at the end are "the reward is used
to update the policy via PPO." Stages 1 and 2 in the diagram (collecting real preference labels,
training the reward model on them) are represented in the code by classes with the right names,
ready for a real preference dataset and training script to fill in.

## 2. System Architecture

Before the method — here's how the pieces actually fit together. Four independent model wrappers
(policy, reference, reward, value), one trainer orchestrating all of them, and a chain of small
modules doing generation, KL, advantage estimation, and the two training updates in between.

**Figure 2.** Package layout: four independent model wrappers sharing only code, not weights.
```mermaid
flowchart TD
    config["config/<br/>PPOConfig"] --> trainer["rlhf_ppo.ppo/<br/>PPOTrainer"]

    subgraph ModelsPkg["models/ — 4 independent LoRA-wrapped copies of the base model"]
        policym["PolicyModel"]
        sftm["SFTModel"]
        rewardm["RewardModel"]
        valuem["ValueModel"]
    end

    baseSetup["base_setup/<br/>tokenizer + LoRA + AutoModelForCausalLM"] --> ModelsPkg
    ModelsPkg --> trainer

    trainer --> generator["generator.py<br/>SequenceGenerator"]
    trainer --> rewardcomp["reward_computer.py"]
    trainer --> valueadd["value_adder.py"]
    trainer --> modeltrain["model_trainer.py"]

    generator --> attribution["token_attribution/<br/>state-action pairs"]
    attribution --> klmod["reward_kl/<br/>KLDivergence + RewardAdjuster"]
    klmod --> advmod["advantage/<br/>TDError + GAEAdvantage + ReturnCalculator"]
    advmod --> buffer[("rlhf_ppo.buffer")]
    advmod --> trainmod["training/<br/>PolicyLoss + ValueLoss + PolicyRatio"]
    trainmod --> modeltrain
```

`PolicyModel`, `SFTModel`, `RewardModel`, and `ValueModel` all subclass the same `BaseModel`, and
`BaseModel.__init__` calls `base_model_setup.create_base_model()`, which does a fresh
`AutoModelForCausalLM.from_pretrained(...)` plus `get_peft_model(...)` every time it's called. Each
of the four models is a **separate, independently LoRA-initialized copy** of Qwen3-0.6B-Base, not
four heads sharing one transformer body. I didn't share a backbone across the four roles here,
which costs roughly 4x the memory a shared design would need, in exchange for each role's forward
pass being completely independent of the others.

## 3. Method

### 3.1 Parameter-Efficient Fine-Tuning with LoRA

Four models need gradients or forward passes here (policy, reference, reward, value), each
wrapping a full copy of Qwen3-0.6B-Base. Fine-tuning every parameter of four separate 0.6B-param
copies is the kind of thing PEFT exists to avoid, so I used LoRA [5] instead: freeze the pretrained
weight matrix entirely and learn a low-rank update on top of it.

**Figure 3.** Weight updates in regular fine-tuning versus LoRA, Figure 1 from Hu et al. 2021 [5].
![Diagram comparing the weight update in regular fine-tuning (full delta-W matrix) versus LoRA (low-rank A/B decomposition)](assets/fig2-lora-weight-update.png)

`base_setup/lora_setup.py` applies this to the attention projections
(`q_proj`, `k_proj`, `v_proj`, `o_proj`) with rank `r=16`, via Hugging Face's `peft` library. One
line (`get_peft_model(base_model, lora_config)`) stands in for the `A`/`B` decomposition above.
The frozen gray `W` block on the right is what `get_peft_model` does under the hood: the pretrained
`q_proj`/`k_proj`/`v_proj`/`o_proj` weight matrices are left untouched and marked
`requires_grad=False`, while `peft` inserts the trainable low-rank `A`/`B` pair (`r=16` here, versus
the diagram's generic `r`) alongside them. Training only ever touches `A` and `B`, never `W` itself.

### 3.2 The PPO Step

`PPOTrainer.ppo_step()` is ten sequential function calls, each one module:

**Figure 4.** One PPO step, matching `PPOTrainer.ppo_step()` line for line.
```mermaid
flowchart TD
    prompts["Prompts"] --> gen

    subgraph gen["Generation and token attribution"]
        policy["Policy model<br/>generate() + logprobs"]
        sft["SFT model (frozen)<br/>reference logprobs"]
        policy --> sa["generate_sequences()<br/>-> state-action pairs + generated text"]
        sft --> sa
    end

    sa --> reward["Reward model<br/>compute_rewards()"]
    sa --> kl["KL divergence<br/>add_kl_to_sa_pairs()<br/>logprob_policy - logprob_ref"]

    reward --> adjust["Reward adjuster<br/>reward - kl_coef x KL"]
    kl --> adjust

    adjust --> values["Value model<br/>add_values_to_sa_pairs()"]

    values --> advantage["Advantage computation<br/>TD errors -> GAE -> reward-to-go"]

    advantage --> vtrain["Value trainer<br/>MSE loss"]
    advantage --> ptrain["Policy trainer<br/>PPO-clip loss"]

    vtrain --> updated["Updated policy + value models"]
    ptrain --> updated
    updated -.next PPO step.-> prompts

    style gen fill:#e1f5fe,stroke:#01579b
    style adjust fill:#fff3e0,stroke:#e65100
```

Generation produces a sequence token by token; each token becomes one (state, action) pair: state
is the prompt plus everything generated so far, action is the next token. Both the policy and the
frozen reference model score that same action, and the difference of their log probabilities is
the per-token KL divergence:

```python
# reward_kl/kl_divergence.py
pair["kl_div"] = logprob_policy - logprob_ref
```

That KL gets subtracted from the sequence-level reward
(`reward_kl/reward_adjuster.py`: `adjusted_reward = reward - kl_coef * kl`),
the mechanism that stops the policy from drifting arbitrarily far from the reference model just to
chase reward. From there, TD errors (`advantage/td_error.py`), GAE
(`advantage/gae_advantage.py`, the standard backwards recursive
`gae = td_error + gamma * lam * gae`), and reward-to-go all feed two separate trainers: the value
model minimizes MSE against reward-to-go, and the policy model maximizes the PPO-clipped
objective:

```python
# training/policy_loss.py
ratio = torch.exp(new_logprobs - old_logprobs)
unclipped = ratio * advantages
clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
loss = -torch.mean(torch.min(unclipped, clipped))
```

I recompute both `old_logprobs` (in `generator.py`) and `new_logprobs` (in `policy_ratio.py`) the
same way: a direct, correctly-masked forward pass over the real token sequence, not from
`generate()`'s own sampling-time scores. Getting that identical on both sides is what makes the PPO
ratio `exp(new_logprobs - old_logprobs)` meaningful at all — see [Section 6](#6-experimental-results)
for the invariant this guarantees.

## 4. Implementation

Mapping the architecture from [Section 2](#2-system-architecture) onto the actual file layout:
this repo is a flat set of modules at the repo root — no `src/`, no install step, no `sys.path`
manipulation anywhere. `make train SERVICE=rlhf-ppo` runs `python -m training.run_ppo` and `make serve SERVICE=rlhf-ppo` runs
`python -m serving.api`, directly.

Shared concerns that every repo in this portfolio needs identically — device selection, logging,
seeding, the base config fields — come from [`libs/common`](https://github.com/RoshBeed/ai-residency/tree/main/libs/common)
rather than being reimplemented here. `PPOConfig` inherits from its `BaseConfig`.

```

├── config/                PPOConfig(BaseConfig) — model name, LoRA rank, PPO hyperparameters
├── rlhf_ppo/
│   ├── base_setup/          Tokenizer, LoRA config, and base-model construction
│   ├── models/              PolicyModel, SFTModel, RewardModel, ValueModel (each: BaseModel + a head)
│   ├── ppo/
│   │   ├── trainer.py         PPOTrainer — orchestrates the 12-stage ppo_step()
│   │   ├── generator.py       SequenceGenerator — generation + per-token (state, action) extraction
│   │   ├── reward_computer.py RewardComputer — scores full (prompt + generation) text
│   │   ├── value_adder.py     ValueAdder — attaches value predictions to (s, a) pairs
│   │   └── model_trainer.py   ModelTrainer — value MSE loss + policy PPO-clip loss, two optimizers
│   ├── token_attribution/   Alternative (s, a) pair construction path (extractor + pair builder)
│   ├── reward_kl/           KLDivergence, RewardAdjuster
│   ├── advantage/           TDError, GAEAdvantage, ReturnCalculator
│   ├── training/
│   │   ├── policy_ratio.py    PolicyRatio — recomputes log pi_new(a|s) for the PPO ratio
│   │   ├── policy_loss.py     PolicyLoss — the PPO-clip surrogate objective
│   │   ├── value_loss.py      ValueLoss — MSE against reward-to-go
│   │   └── run_ppo.py         Entrypoint: one ppo_step() on two example prompts (`make train SERVICE=rlhf-ppo`)
│   ├── buffer.py            ExperienceBuffer — the experience one policy update reads
│   └── types.py             SAPair, the (state, action) dict every stage reads from and writes to
├── serving/api.py         FastAPI generation service (`make serve SERVICE=rlhf-ppo`), what the portfolio frontend calls
examples/basic_usage.py  Three consecutive PPO steps, for watching the losses move
notebooks/walkthrough.ipynb    Narrated, cell-by-cell run of the real ppo_step() call sequence
tests/test_ppo_components.py   Unit tests for every module above
```

## 5. Usage

The commands below are the primary, documented way to run this repo: a thin `Makefile` wrapping
the real entrypoints, plus a Docker path for running them without a local Python setup at all.

```bash
make install   # uv sync — installs from pyproject.toml/uv.lock
make train SERVICE=rlhf-ppo     # uv run python -m training.run_ppo — see note below
make serve SERVICE=rlhf-ppo     # uv run python -m serving.api — FastAPI generation service on :8000
make test SERVICE=rlhf-ppo      # uv run pytest -q
make check     # lint + strict mypy + tests, everything CI would run
```

**What the generation demo does:** the browser UI (the unified frontend in `services/frontend`,
calling this repo's `serving/api.py`) loads `PolicyModel` and generates a sampled completion for a
prompt you type in — the real generation pipeline end to end (tokenizer setup with
`padding_side="left"`, float32 model loading, LoRA wrapping, sampling `generate()`). This is the
one-model demo path; `PolicyModel` here is a freshly LoRA-initialized copy of Qwen3-0.6B-Base, and
`make train SERVICE=rlhf-ppo` below exercises the full four-model pipeline.

**What `make train SERVICE=rlhf-ppo` actually runs:** one PPO step end to end (generation, KL penalty, reward
scoring, GAE advantage, one value-model gradient step, one policy-model gradient step) on two
example prompts. `examples/basic_usage.py` runs three of them back to back to watch the losses move
between steps.

**What `make serve SERVICE=rlhf-ppo` actually runs:** the same FastAPI generation service (`serving/api.py`) the
portfolio frontend calls, matching every other project's HTTP contract.

**Device selection:** `PPOTrainer` calls `ai_residency_common.get_device()`, which picks `cuda` > `mps` >
`cpu` in that order, at construction time. Running this locally on Apple Silicon
now uses MPS automatically: four float32 model copies (SFT/policy/reward/value) benefit
meaningfully from it, and I confirmed empirically that LoRA/PEFT plus batched `generate()` produce
the same coherent output on MPS as on CPU, just faster. Inside Docker, this always falls back to
CPU. The container is Linux, and MPS is a macOS-only backend with no passthrough into a Linux VM,
so `make -C deploy up  # then: docker compose --profile train run --rm train-rlhf-ppo`/`make -C deploy up` never see `mps` regardless of the host.

**Figure 5.** Running `training/run_ppo.py` (what `make train SERVICE=rlhf-ppo` runs under the hood).
```mermaid
sequenceDiagram
    actor Dev as Developer
    participant Main as run_ppo.py
    participant Trainer as PPOTrainer
    participant Gen as SequenceGenerator
    participant RC as RewardComputer
    participant MT as ModelTrainer

    Dev->>+Main: python -m ...training.run_ppo
    Main->>+Trainer: ppo_step(prompts)
    Trainer->>+Gen: generate_sequences(prompts)
    Gen-->>-Trainer: sa_pairs, generated_texts
    Trainer->>Trainer: add_kl_to_sa_pairs()
    Trainer->>+RC: compute_rewards(prompts, generated_texts)
    RC-->>-Trainer: rewards
    Trainer->>Trainer: adjust_rewards_with_kl()
    Trainer->>Trainer: add_values, TD errors, GAE, reward-to-go
    Trainer->>+MT: train_value_model(sa_pairs)
    MT-->>-Trainer: value_loss
    Trainer->>+MT: train_policy_model(sa_pairs)
    MT-->>-Trainer: policy_loss
    Trainer-->>-Main: results dict
    Main-->>-Dev: log generated texts, rewards, losses
```

```bash
python -m training.run_ppo
pytest -q
```

### Docker

Two images, built from the repository's root `Dockerfile.serve` (the FastAPI generation service,
`FROM roshb/ai-residency-base-serve`) and `Dockerfile.train` (one PPO step on a GPU host,
`FROM` the CUDA base). Both are parameterised by `--build-arg SERVICE=rlhf-ppo`, so this service
carries no Dockerfile of its own. Neither needs a git
credential — `libs/common` is already installed in the base image.

```bash
# both images for this service, built from the repository root
make build SERVICE=rlhf-ppo
make push SERVICE=rlhf-ppo

# the whole stack, this service included
make -C deploy up

# train in a container
cd deploy && docker compose --profile train run --rm train-rlhf-ppo
```

`train` is profile-gated in `docker-compose.yml` (`profiles: [train]`) so a bare
`docker compose up` never runs a real gradient-update step by accident; `serve` has no such gate
since it's safe to run by default. A commented-out GPU reservation block sits in
`docker-compose.yml` for a future real GPU host, inert on this CPU-only environment but
structurally ready.

## 6. Experimental Results

`PPOTrainer.ppo_step()` runs end to end against a real pretrained reward model, with a fixed seed
(`libs/common`'s `set_seed`) for reproducible runs. One gradient step on two example
prompts produces:

```
Generated texts: [" I'm pretty sure we should; however NodeList was",
                   ' Do i use the more expensive alternative, and if']
Rewards: [1.3879, 1.3136]
Value loss: 57.1910
Policy loss: -6.9180
```

The PPO ratio — `exp(new_logprobs - old_logprobs)` — comes back at exactly 1.0000 across every
token before any gradient step, on a deliberately unequal-length prompt pair, and a regression
test asserts it. That's the invariant that matters most here: it confirms `old_logprobs` and
`new_logprobs` are genuinely comparable, both computed via a real, correctly-masked forward pass
rather than `generate()`'s own sampling-time scores (see [Section 3.2](#32-the-ppo-step)).

### A real loop, and what 60 steps actually buy

`make train-loop` runs PPO on real prompts from `CarperAI/openai_summarize_tldr` and reports mean
reward before and after on held-out prompts. It is built so it cannot flatter itself: the 16
evaluation prompts come from the front of the split and the training prompts strictly after it, so
the two sets cannot overlap; the baseline is taken before a single gradient step; and the trainer
starts from a freshly initialised adapter rather than resuming one.

| | greedy | sampled (4x, seeded) | policy drift |
|---|--:|--:|--:|
| pre-PPO, 16 held-out prompts | -3.4202 | -5.0505 | 0 |
| after 60 PPO steps | -3.4585 | -4.9640 | 1.26e-04 |
| change | **-0.0383** | **+0.0865** | |

**The two metrics disagree about the sign**, and that is the result. Greedy reward falls by 0.04 and
sampled reward rises by 0.09, which is what a measurement looks like when the thing being measured
has barely moved: `policy_drift` — the adapter's effective update as a fraction of the weights it is
added to — comes out at **1.26e-04**. The policy shifted by about a hundredth of a percent. Neither
sign is a finding at that magnitude, and reporting one of them alone would have invented a direction
the run does not support.

What the run does establish is that the loop works end to end on real data, and what it costs to
move a policy: 60 steps at `learning_rate=1e-5` is not enough to change what this policy writes.

That measurement exists because the alternative was ambiguous. Greedy decoding at iterations 10, 20,
30 and 40 returned **-3.4202 every time, to four decimal places** — greedy output only changes when
a change flips an argmax, and across 160 token decisions none flipped. Without a drift number,
"reward did not move" and "the metric cannot see that the policy moved" look identical. So the loop
reports both, plus a seeded-sampling reward that responds to a shift in the output distribution
rather than only to a change of its mode.

**A bigger learning rate is not the fix, and that was tested rather than assumed.** The obvious
response to "the policy barely moved" is to move it further. LoRA's update is `(alpha/r) * B @ A`,
so multiplying every B matrix by k multiplies the drift by k without retraining anything — which
makes the question cheap to answer:

| policy drift | greedy reward | sampled reward |
|--:|--:|--:|
| 1.26e-04 (what 60 steps produced) | -3.4585 | -4.9640 |
| 1.26e-03 | -3.7579 | -4.9146 |
| 1.26e-02 | -4.9454 | -5.6183 |
| 1.26e-01 | -5.6364 | -5.4824 |

Two things fall out. The metric is **not** blind — it responds sharply from 1e-03 upward, so the flat
reward at 1.26e-04 is genuinely a policy that did not move rather than an instrument that could not
see. And travelling further in the direction these 60 steps found makes reward monotonically
**worse**, by 2.2 under greedy decoding at 1000x. Turning up the learning rate would not have
produced a better number; it would have produced a worse one with more confidence.

This is a statement about *this* direction, not about larger steps in general: PPO with a bigger
learning rate would compute different advantages and find a different direction. What it rules out
is the cheap explanation. The remaining suspects are in the signal rather than the step size —
advantage estimates from 4-prompt batches, 10-token completions, and a KL reference that is the base
model rather than an SFT checkpoint.

Two bugs came out of chasing that, and both were invisible to every passing test:

**`_load_ckpt` had never loaded a checkpoint.** `load_checkpoint` returns the whole saved record —
`model_state_dict`, `optimizer_state_dict`, `epoch`, `metrics`, `extra` — and the loader passed that
straight to `load_state_dict(strict=False)`. None of those five keys is a parameter name, so nothing
matched and nothing loaded, under a log line reading `Loaded policy LoRA checkpoint (513 missing, 5
unexpected)`. The 5 was the whole story. It is 240 parameters applied, 0 unexpected now, and a
checkpoint matching nothing is an error naming the file.

**Then the loop had to opt out of it.** A working loader means the next run resumes the last run's
adapter and calls it "pre-PPO", so `run_loop` passes `load_adapters=False` and checks that baseline
drift is 0 before trusting its own baseline.

**Also worth knowing: MPS cannot finish a step here.** A PPO step holds activations for four model
roles across twelve stages, and on MPS that accumulates past the backend's high-watermark ceiling
and dies with `MPS backend out of memory (MPS allocated: 42.45 GiB, max allowed: 42.43 GiB)` — after
252-413s in the value update alone. The same step on CPU completes in **70s**, value update 17.3s
and policy update 22.0s, so `make train-loop` pins `AI_RESIDENCY_DEVICE=cpu`.

**What is and is not a real model here.** The reward model is real: `OpenAssistant/reward-model-deberta-v3-large-v2`,
preference-trained by someone else and used inference-only, so the reward signal is not something
this repo invented. The policy is SmolLM2-135M-Instruct with LoRA. `SFTModel` is the gap — a full
RLHF pipeline initialises the policy from a supervised fine-tuned checkpoint and penalises KL
against *that*, and no SFT stage exists here, so it is a second copy of the base model. The KL
penalty works exactly as written; it measures drift from the base model rather than from an SFT
model.

## 7. Related Work

- [the image captioning project](https://github.com/RoshBeed/ai-residency/tree/main/services/multimodal-captioning) —
  both projects load Qwen3 explicitly in float32, since a bf16-by-default checkpoint isn't
  compatible with the float32 heads either one adds on top.
- [the vision transformer project](https://github.com/RoshBeed/ai-residency/tree/main/services/vision-transformers) — the same
  "break it into small, independently-testable modules" approach applied here to a training loop
  (generation, KL, reward, advantage, policy/value updates) rather than a model's forward pass.

## References

[1] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy
Optimization Algorithms.* arXiv:1707.06347.

[2] Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). *Deep
Reinforcement Learning from Human Preferences.* NeurIPS.

[3] Stiennon, N., Ouyang, L., Wu, J., Ziegler, D. M., Lowe, R., Voss, C., Radford, A., Amodei, D.,
& Christiano, P. (2020). *Learning to Summarize from Human Feedback.* NeurIPS. The Reddit TL;DR
dataset and three-stage recipe (Figure 1) this repo's prompts and task brief are modeled on.

[4] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C.,
Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M.,
Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). *Training Language Models
to Follow Instructions with Human Feedback.* NeurIPS. (InstructGPT)

[5] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021).
*LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685.

## Acknowledgments

Built with Helen Zhou, Ewan Beattie, and Aparna Pillai; the from-scratch PPO implementation is my
own. Figures 1 and 3 reproduce Figure 2 from Stiennon et al. [3] and Figure 1 from Hu et al. [5]
directly (see References).

## Dependencies

PyTorch · Hugging Face `transformers` · `peft` (LoRA) · `datasets` · NumPy · tqdm · `accelerate` ·
FastAPI · Streamlit · [`libs/common`](https://github.com/RoshBeed/ai-residency/tree/main/libs/common) (shared device,
logging, config and tracking helpers) · ruff · mypy
