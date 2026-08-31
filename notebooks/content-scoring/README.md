# Experiment record

The write-up in [`services/content-scoring`](../../services/content-scoring/README.md) states
conclusions. This is where the measurements behind them live, including the ones that went nowhere
and the two occasions a conclusion had to be withdrawn.

Scripts expect to run from the service directory, where `Config`'s relative paths resolve:

```bash
cd services/content-scoring
uv run python ../../notebooks/content-scoring/encoder_probes.py mini
```

They sit outside the service deliberately: they hardcode an `mps` device, download pretrained models
the service does not depend on, and have no tests.

## The two objectives, measured against each other

Identical code, features, seeds and promotion logic; the only difference is the loss. Late fusion,
three seeds each, on the held-out test period.

| | `log_mse` | `poisson` (shipped) | actual |
|---|---|---|---|
| median abs error | **1.36** | 7.49 | — |
| MAE | **16.96** | 24.24 | — |
| mean predicted | 3.39 | **15.99** | 18.91 |
| max predicted | 80 | **1,286** | 3,158 |
| total upvotes | 18% | **85%** | 100% |
| Spearman | **0.367** ±0.0009 | 0.280 ±0.0202 | — |
| PR-AUC on `score ≥ 100` | 0.089 | **0.097** | — |

`log_mse` fits the conditional median, which is 3, and gets a median error of 1.36 upvotes while
never exceeding 80 on any of 69,275 posts. `poisson` fits the mean, which is 19. Predicting 8 for a
post that scores 3 is the price of being able to predict 400 for one that scores 500.

`log_mse` is also **twenty times more stable across seeds**. A loss dominated by a handful of
thousand-point posts depends on which batch those posts land in, and that instability is why the
early-versus-late comparison separates under one objective and dissolves under the other.

## Encoder probes

| representation | train rows | test ρ | viral ROC-AUC | PR-AUC |
|---|---|---|---|---|
| word2vec, mean-pooled (shipped) | 710k | 0.366 | 0.678 | 0.090 |
| MiniLM-22M, frozen | 710k | 0.364 | 0.669 | 0.079 |
| MiniLM-22M, fine-tuned | 710k | **0.383** | 0.721 | 0.107 |
| Qwen2.5-0.5B + LoRA | 300k | 0.379 | **0.736** | **0.117** |

Measured under `log_mse`, since these are questions about ranking.

**A frozen encoder buys almost nothing.** Testing only the frozen case is how the write-up briefly
concluded the text side was exhausted — a conclusion that survived about an hour.

**The comparison must be data-matched.** MiniLM on 300k against the service on 710k looked like a
tie; matched at 300k the service scores 0.3495 against MiniLM's 0.3700. The first version of that
comparison flattered the baseline.

**Scale is not the story.** Qwen-0.5B and MiniLM-22M land within 0.004 of each other on title-only
text. Representation quality beat parameter count by twenty times the margin parameter count beat
itself. Caveat: Qwen used LoRA rank 16 on attention projections only, so this says cheap adaptation
of a 0.5B model does not help — not what a full fine-tune of a larger model would do.

**Learning rate barely matters**: 1e-5 / 3e-5 / 5e-5 give 0.368 / 0.381 / 0.377.

## Objective and the tail

Three objectives, same architecture, three seeds:

| objective | test ρ | PR-AUC on `score ≥ 100` |
|---|---|---|
| `log_mse` | **0.366** | 0.090 |
| `poisson` | 0.283 | 0.096 |
| binary classifier on `score ≥ 100` | 0.207 | **0.106** |

The classifier finds front-page posts best and cannot rank at all. **No single output does both**,
because three quarters of Hacker News posts sit between 1 and 5 upvotes and nothing separates a 2
from a 4. Which to train is a product question.

## How much of this is luck

A quarter of all linked posts are reposts — 85,340 URLs submitted more than once.

| | |
|---|---|
| Spearman between the 1st and 2nd posting of the same URL | **0.260** |
| the service, on posts it has never seen (`log_mse`) | 0.366 |

The same content posted twice predicts itself worse than the model predicts an unseen post — the
model can beat that bound because it also knows *when*, *who* and *where*, which two submissions of
one URL do not hold fixed. Of URLs whose first submission scored 2–5, the second reached a maximum
of 2,645. Knowing identical content already hit the front page moves the odds from 5.6% to 11.3%.

## Capacity, three ways

None of these helped:

| | |
|---|---|
| depth, 1 → 4 layers | 0.368 → 0.343 |
| gradient-boosting leaves, 31 → 255 | 0.307 → 0.302 |
| boosted trees instead of the MLP | 0.307 against 0.317 on identical features |

Capacity over a fixed weak representation is not the constraint; the representation is. Stated more
broadly than that at first, which the encoder probes then contradicted.

**Caveat**: these ran while the momentum features were present. They have since been removed, and a
re-run without them is in `capacity2.log`.

## The momentum features

Trailing-window statistics over the corpus — a domain's mean score over the previous 30 days,
whether a title's words are spiking this week — on the theory that a post about Apple during a
launch week is a different bet, and that the corpus already records what the world is attending to
so no external knowledge is needed.

Three seeds each way: **ρ 0.3655 with them, 0.3655 without.** Identical on every metric.
`domain_recent_mean` correlates +0.169 with the target alone but adds nothing the domain embedding
and drift term were not already carrying; `token_burst` correlates −0.025. Removed.

## Author history

| prior front pages | P(≥100) |
|---|---|
| 0 | 3.9% |
| 1 | 5.7% |
| 2–3 | 5.8% |
| 11+ | 4.6% |

Raw, the effect saturates and decays — prolific posters have many hits and much more mediocrity.
Controlled for experience (authors with 5–50 prior posts): **3.9% → 6.7%**, a 1.7× lift. Real,
repeatable, and much smaller than it feels.

## Things that were wrong for a while

- The eval harness measured RMSE in log space, so nothing was watching whether the service could
  predict a count. It shipped predicting 19% of the upvotes that happen.
- Three seeds were trained and one was reported: every seed wrote the same checkpoint path.
- `viral_probability` was computed from the Poisson's analytic tail, which is ~1e-25 at a mean of
  15, so it returned 0.0% for every post. It is fitted on validation now.
- The sweep optimised rank correlation while the objective underneath had moved to counts.
- The slice table exponentiated actuals that were already raw counts, printing `inf`.
