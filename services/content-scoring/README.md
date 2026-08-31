# Hacker News Score Prediction

## TL;DR

- **What it is:** given a Hacker News post — its title, where it links, when it was submitted and
  who by — predict how many upvotes it will get.
- **The data:** 872,554 real posts, November 2023 to August 2026, split by time so the model is
  always predicting a future it has not seen.
- **Result:** the served model's median error is **7.5 upvotes**, less than half the 16.2 you get by
  guessing the average every time, and it accounts for 85% of the upvotes that actually happened.
- **The honest part:** Hacker News is substantially a lottery. The same link posted twice scores
  similarly only about a quarter of the time, and no model reads that off a title.

This is the second half of Week 1. The [word-embeddings
project](https://github.com/RoshBeed/ai-residency/tree/main/services/word-embeddings) trains the
word vectors from scratch; this one puts them to work on a task where the title is only part of the
story.

**Figure 1.** The task.

![A Hacker News post — "The Best Essay" from paulgraham.com, 207 points by tosh — with an arrow into a box labelled "Magic" and an arrow out to the number 265.](assets/word-embeddings/fig1-hn-score-task.png)

## Contents

1. [Results](#1-results)
2. [How it works](#2-how-it-works) — the four architectures, and where the title meets the metadata
3. [The data](#3-the-data) — 872k posts, and why the split is by time
4. [Extending the vocabulary](#4-extending-the-vocabulary) — text8 has never heard of `gpt-4`
5. [What I learned](#5-what-i-learned) — including the parts that did not work
6. [Usage](#6-usage) and [implementation](#7-implementation)

## 1. Results

Held-out June to August 2026: 69,275 posts carrying 1,309,690 upvotes. Every row is the mean of
three seeds.

| model | median error | mean predicted | highest prediction | share of upvotes |
|---|---|---|---|---|
| *what actually happened* | — | *18.9* | *3,158* | *100%* |
| guess the average every time | 16.2 | 19.2 | 19 | 101% |
| TF-IDF + Ridge on the title | 13.0 | 21.4 | 453 | 113% |
| title only | 15.1 | 19.5 | 265 | 103% |
| metadata only | 13.1 | 19.3 | 562 | 102% |
| early fusion | 10.0 | 16.6 | 651 | 88% |
| **late fusion** (served) | **7.5** | 16.0 | **1,286** | 85% |

Two things to read here. The median error more than halves against guessing the average, so the
model is doing real work. And its highest prediction is 1,286 upvotes — it can call a front-page
post, which matters because those are the posts anyone cares about.

It still under-accounts by 15%, which is worth stating plainly: the model is a little shy of the
biggest posts.

**Figure 2.** Every model against the baselines.

![Two horizontal bar panels comparing eight models. The four neural architectures lead the four baselines on both panels, with late fusion highest.](assets/fusion-comparison.png)

### What the model uses

Zero out one branch at a time and see what the prediction loses:

| branch removed | effect |
|---|---|
| **when it was posted** | **−0.12** |
| the title | −0.06 |
| the author's history | −0.04 |
| the title's shape | −0.02 |
| where it links | −0.01 |

**When you post matters more than what you say.** That is an uncomfortable result for a project
about titles, and it is why the metadata-only ablation exists — without it, "the title matters"
would have been an assumption rather than a measurement.

**Figure 3.** Leave one branch out.

![Horizontal bars showing the drop in rank correlation when each branch is zeroed: temporal largest, then text, author, title shape and domain.](assets/feature-ablation.png)

### Where it struggles

`Show HN` posts are much harder than ordinary ones — they compete on what was built rather than on
how it is described, and a title carries less of that. Titles full of words the vocabulary does not
know are harder again, which is what the vocabulary extension in §4 exists to narrow.

**Figure 4.** Accuracy by slice.

![Five bar panels of accuracy by slice: score quartile, domain, post prefix, out-of-vocabulary fraction and weekday.](assets/slice-performance.png)

## 2. How it works

A post arrives as four different kinds of thing: some text, a timestamp, a link, and an author. The
question this project is built around is **where those should meet**.

**Figure 5.** Early fusion versus late fusion.

![Two architectures side by side. Late fusion runs each input through its own network before combining them; early fusion concatenates everything first and uses a single network.](assets/word-embeddings/fig3-late-early-fusion.png)

**Late fusion** gives each kind of input its own small network, then joins the results at the end.
**Early fusion** concatenates everything up front and lets one network sort it out.

**Figure 6.** The late-fusion model this service serves.

![A Hacker News post feeding three feature extractors, each into its own small network, whose outputs are concatenated and passed through a final network to a single predicted score.](assets/word-embeddings/fig2-reward-model-architecture.png)

Both are built, and so are two ablations — **title only** and **metadata only** — because without
them "combining helps" cannot be tested. It does help, by a wide margin: either input alone lands
near a 13–15 upvote median error, and together they reach 7.5.

The choice between early and late fusion turns out to matter much less. The gap between them is
smaller than the difference between two training runs of the same model, so this write-up does not
claim a winner.

### Predicting a count, not a rank

The model is trained with a Poisson objective, the natural fit for counting things: it predicts an
expected number directly.

The first version used squared error on the log of the score, and was quietly useless as a
predictor — it never once predicted more than 80 upvotes across 69,275 posts, and accounted for 18%
of the upvotes that happened. It was excellent at *ordering* posts and could not say how many
upvotes any of them would get.

That is a different question rather than a bug. Squared error on logs chases the *typical* post,
which on Hacker News scores 3. Poisson chases the *average*, which is 19. Both are defensible; only
one answers "how many upvotes will this get". `AI_RESIDENCY_OBJECTIVE=log_mse` switches back.

### Two numbers, not one

`/predict` returns an expected count and a probability of reaching the front page:

```
 8.3 upvotes   P(100+) = 3.0%   Show HN: I built a database that runs in the browser
16.2 upvotes   P(100+) = 5.3%   Attention Is All You Need
 2.0 upvotes   P(100+) = 0.9%   10 JavaScript tricks you wont believe
```

A single expected value hides how wide the range is. Even among the posts this model ranks in its
top ten percent, the median outcome is 4 upvotes — that group's average is only high because a few
of them take off.

## 3. The data

872,554 Hacker News stories, November 2023 to August 2026, mirrored at
[`roshbeed/ai-residency-hn-posts`](https://huggingface.co/datasets/roshbeed/ai-residency-hn-posts).

The service used to train on 782 posts pulled from a database that is no longer reachable, so the
corpus could not be rebuilt. Now `make etl` fetches, cleans and splits the whole thing with no
credentials, and `make etl-verify` checks the mirror against the live Hacker News API.

**Split by time, not at random.** Train on 2023–2026, validate on spring 2026, test on summer 2026.
Scores drift and posts from the same day compete for the same front page, so a random split would
let the model see the future of its own test set.

**Figure 7.** The score distribution.

![Two panels over 710,722 training posts. Raw scores on log-log axes fall steeply from 180,000 posts scoring 1 down to single posts scoring thousands. The logged version shows a tall spike at the low end and a long thin tail.](assets/score-distribution.png)

The median post scores 3 and the maximum is 5,710. That shape is why the objective matters so much,
and why a model that predicts 3 for everything can look deceptively good.

**The comment count is never used.** It predicts the score well and is not known when a post is
submitted, so a model using it would score beautifully and answer a question nobody asked. The same
care applies to the author's history, which is built only from their strictly earlier posts.

## 4. Extending the vocabulary

text8 is Wikipedia from 2006. It has no `gpt-4`, no `kubernetes`, no `postgres` — and those are much
of what a Hacker News title is about.

So this service extends the vocabulary onto 710,000 Hacker News titles, keeping every existing word
exactly where it was and adding the new ones on top:

| | before | after |
|---|---|---|
| vocabulary | 71,290 | **86,438** |
| words per title the model cannot read | 11.3% | **3.2%** |

The pretrained vectors are held fixed while the new words are learned. Letting everything move fit
the Hacker News corpus better and **cost 0.15 of the word-similarity benchmark** the sibling project
publishes — 710k headlines is not enough signal to re-place 71,290 vectors, and the check that
caught it now runs as part of the pipeline.

## 5. What I learned

**When you post matters more than what you write.** The strongest single branch is the timestamp,
not the title.

**Hacker News is substantially a lottery.** A quarter of all linked posts are reposts — 85,340 URLs
submitted more than once — and the same URL's two scores agree only weakly. Of URLs whose first
submission scored 2–5, the second reached a maximum of 2,645. Knowing that identical content already
hit the front page roughly doubles the odds, from 5.6% to 11.3%, and no more than that.

**A better encoder beats a bigger model.** Averaging word vectors throws away word order — *Google
acquires OpenAI* and *OpenAI acquires Google* are identical to this model. Swapping in a fine-tuned
sentence transformer improves it; making the network deeper does not, and consistently makes it
worse. The measurements are in [`notebooks/content-scoring/`](../../notebooks/content-scoring/).

**What did not work.** Trailing-window features — how a domain has been doing lately, whether a
title's words are spiking — on the theory that a post about Apple during a launch week is a
different bet. Measured across three seeds: no effect at all, on any metric. The reasoning still
seems right to me; the signal is not there.

**What I would do next.** Replace the encoder, which is the one change the evidence clearly
supports. Then decide whether this is a "how many upvotes" product or a "which posts will take off"
product, because those want different models and the first currently wins by default.

## 6. Usage

```bash
make -C services/content-scoring etl            # fetch and split 872k posts
make -C services/content-scoring extend-vocab   # add Hacker News words to the vectors
make -C services/content-scoring train          # train one architecture
make -C services/content-scoring evaluate       # score every trained seed
make -C services/content-scoring serve          # the API on :8000
```

Every input is a public Hugging Face repo, so the whole chain runs on a clean machine with no
credentials.

`make train-all` trains four architectures at three seeds; `make capacity` runs the depth
comparison; `wandb sweep sweep.yaml` then `make sweep SWEEP=<id>` searches hyperparameters.

**Figure 8.** The Streamlit page, which runs all four architectures over a post you type.

![The content-scoring page showing a Hacker News title input and a predicted score.](assets/word-embeddings/streamlit_ui.png)

## 7. Implementation

```
content_scoring/
├── etl.py           Fetch, clean and split the corpus
├── vocab.py         Extend the text8 vocabulary onto the titles
├── features.py      The four feature groups, and the causal author history
├── data.py          Splits as tensors
├── model.py         The four architectures behind one build_model
├── train.py         The epoch loop; streams evaluation to W&B
├── evaluate.py      Metrics, baselines, slices, ablation, calibration
├── predict.py       Scoring one post through the same code as the corpus
└── report.py        The figures
```

Every number in this write-up comes from `make evaluate`, and the trainer streams the same
measurements to W&B, so a chart and a table here cannot disagree.

Word-vector scoring lives in [`libs/word2vec`](../../libs/word2vec), shared with the sibling
project.

## Dependencies

`torch`, `pandas`, `pyarrow`, `scikit-learn`, `scipy`, `fastapi`, `wandb`,
[`ai-residency-common`](../../libs/common), [`ai-residency-word2vec`](../../libs/word2vec).
