I spent a week testing early vs late fusion on a Hacker News upvote predictor, and the
useful finding wasn't which one won. It was that "late fusion" names two different
architectures, and only one of them has the problem every paper warns about.

Here's the thing nobody says plainly: **score-level late fusion is an additive model.**

If each input goes down its own tower to a single score and you add the scores, your model
is f(a, b) = g(a) + h(b). That's a generalised additive model — the same object statistics
has argued about since the 1980s. It cannot represent anything that depends on a and b
*jointly*. Not badly. At all. No width fixes it.

Early fusion — concatenate first — has no such restriction.

So I built a knob. Same real Hacker News features, but a synthetic target where I control
how much of the signal needs both inputs at once, then swept it. 1,440 runs, three
architectures, ten seeds a point.

With no interaction, all three tie exactly. Turn it up and score-level fusion doesn't just
lose, it goes *backwards* — 0.118 Spearman behind by the end. And capacity doesn't help:
eight times the width recovered nine percent of the gap. It's a wall, not a hill.

Now the part that surprised me.

Most people saying "late fusion" mean towers into a shared network. That architecture is
**fully general** — it sees every interaction, just one layer later. Across all 48
conditions I tested, it was indistinguishable from early fusion. Largest difference:
0.0006.

So the classic warning is real, and it mostly doesn't apply to what people are building.

On my actual Hacker News data? The interaction between the title and the metadata is real
but tiny — worth about 0.006 Spearman, roughly 2% of the model's signal. Which means the
architecture choice was free, and I could have picked on operational grounds from the
start.

**And "add the scores" isn't the only way to join two towers.** I tested a third: take a
dot product instead of a sum. That turns out to sit exactly between the two — it can
represent multiplicative interactions and nothing else. On a target where the signal is
title × metadata, it tracked full early fusion to within 0.0014 at every strength I tried,
while summing scores fell from 0.71 to 0.41.

So the real hierarchy is three rungs, not two:

  add two scores  -> no interactions at all
  dot product     -> multiplicative interactions only
  shared network  -> any interaction

How you join the towers decides which functions your model can express. That is the whole
phenomenon, and it is a design decision people usually make by habit.

**The test, before you pick an architecture:**
Fit one model per modality. Combine their predictions linearly. Compare against a single
model over everything concatenated. If the joint model doesn't beat the combination, you
have no interactions worth modelling — and every fusion architecture will score the same.
Twenty lines, and it tells you whether the rest of the decision matters.

**Choosing, when it does matter:**

Early fusion — any interaction, one model, one training run. Needs every modality present
at train and inference, and one team owning the whole thing.

Late fusion, shared trunk — still sees interactions, and lets each input have its own
encoder. This is the default most people actually want.

Late fusion, score level — maximally modular. Separate teams, separate deployments,
per-modality attribution, and it degrades gracefully when an input goes missing. You pay
for that with every interaction in your data.

One caveat I'd flag from my own results: late fusion's robustness advantage shrank as
models got bigger. At 128 hidden units it bought real insurance against a missing input.
At 512 the cost stayed and the benefit mostly vanished — early fusion learned to survive a
missing input on its own. Test it at your scale, not someone else's.

Code and the full write-up are in the repo.
