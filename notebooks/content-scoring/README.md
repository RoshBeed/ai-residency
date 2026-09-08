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

## Fusion: where the title and the metadata should meet

Five scripts, added while turning "early versus late fusion" from two hand-built models
into something measurable.

**`shape_probes.py`** — does each input group prefer a different tower shape? Each group
alone, one neuron budget arranged four ways, three budgets so a deep-and-narrow preference
can be told apart from a narrow first layer. Answer: no. Every group, every budget, "not
separated". That matters because giving each input the shape it wants is late fusion's one
structural advantage over early fusion, and here there is nothing to give.

**`shape_specs.py`** — builds matched-neuron architectures for the comparison above, and
asserts the counts match. Rounding had put three "equal" models at 253, 256 and 288 units.

**`fusion_probes.py`** — the controlled synthetic experiments.

- `interaction` puts a knob on the thing Snoek names as late fusion's weakness. A target
  `g(title) + h(metadata) + α·interaction`, over real HN features, swept from additive to
  interaction-dominated. Crossover at α≈0.75; the gap reaches 0.118; eight times the width
  recovers nine percent of it, so it is a representational limit rather than a fitting one.
- The same sweep separates three ways of joining two towers: **adding their scores** can
  represent no interaction, a **dot product** can represent multiplicative ones and tracked
  full early fusion to within 0.0014 at every strength, and a **shared network** can
  represent anything. Early fusion and the shared-network form were indistinguishable
  across all 48 α×width cells — largest difference 0.0006.
- `robustness` damages one modality at inference. Late fusion degrades more gracefully as a
  modality disappears, but the advantage is a small-model effect: at 128 units it pays
  0.012 undamaged to insure 0.031, at 512 it pays 0.010 to insure 0.006.
- `threshold` adds a synthetic interaction to the *real* target, to bound the null. The
  real data's gap sits where an added interaction of 0.02–0.05 standard deviations does.

**`hn_interactions.py`** — the Hacker News question itself, rather than a synthetic one.
For each pairing it fits a model that cannot interact (a tower each, reduced to a score,
summed) against one that can, on the real upvote target. Eleven pairings: the title against
each metadata group, and every metadata pair against each other.

Ten of eleven come back zero or negative. The three that reach significance are all
*negative* — the joint model is worse — which is what a model spending capacity searching
for structure that is not there looks like. A good title is worth the same at 3am as at
3pm, on github.com as on a personal blog.

The script also carries the controls that a null needs: `--target` (additive in `log1p` is
multiplicative in raw counts, so the finding is a statement about a coordinate system until
checked in more than one), `--epochs` (interactions are harder to learn than main effects),
`--slice` (an interaction confined to Show HN posts averages away over 69,000 rows), and a
capacity-matched additive control (the joint model also has more parameters, so a positive
result could be either).

**`linkedin_figures.py`** — the two diagrams for the write-up's summary, drawn from the
recovered medians so the post and the README cannot disagree.

## Things that were wrong for a while

- The eval harness measured RMSE in log space, so nothing was watching whether the service could
  predict a count. It shipped predicting 19% of the upvotes that happen.
- Three seeds were trained and one was reported: every seed wrote the same checkpoint path.
- `viral_probability` was computed from the Poisson's analytic tail, which is ~1e-25 at a mean of
  15, so it returned 0.0% for every post. It is fitted on validation now.
- The sweep optimised rank correlation while the objective underneath had moved to counts.
- The slice table exponentiated actuals that were already raw counts, printing `inf`.
- **Late fusion was a generalised additive model.** With one hidden layer it had a bare
  linear head after the concat, so it was `Σ w_g·f_g(x_g)` and could not represent *any*
  interaction. The early-versus-late gap reported at the time was measuring that, not
  fusion timing.
- **Block widths were allocated in proportion to input columns**, giving the 300-d title
  233 of 256 hidden units and the author's three scalars 2. The title is 300-d because
  word2vec chose 300, not because it says more; measured, the 29 metadata columns
  out-predict the 300 title columns. The rule handed the weaker input eight times the
  capacity, and a fusion sweep on top of it measured the bottleneck.
- **A sweep on validation MAE found a 28% "improvement" that ranked at chance.** Skewed
  targets reward systematic under-prediction: the winner accounted for 15% of the upvotes
  that happened. Across the sweep, upvotes-accounted-for climbed from 0.15 to 1.05 as MAE
  got *worse*. The mirror of the Spearman mistake above; neither metric is sufficient alone.
- **A robustness smoke test overwrote 1,440 interaction runs** by taking the same default
  `--out`. Medians survived in the log; per-seed values did not.
- **`matrix[tokens] * mask` allocated 27 GB** to produce an 0.85 GB pooled result and
  kernel-panicked the machine. macOS compresses and swaps until the compressor is out of
  segments, then the watchdog panics; jetsam never gets a chance. Pool in chunks.
- **Claims were made from three-seed screenings and two of them were wrong.** The
  allocation ranking reversed at ten seeds — the arm reported as best was mid-table, the
  one reported as worse-than-nothing was second-best. The screening had flagged itself as
  unable to separate its arms, and the write-up went ahead anyway.
