# Word Embeddings from Scratch (CBOW and Skip-gram, Negative Sampling)

## TL;DR

- **What it is:** word2vec trained from scratch — CBOW and skip-gram, negative sampling, no pretrained models — on the `text8` Wikipedia corpus.
- **Result today:** WordSim-353 ρ **0.73**, SimLex-999 ρ **0.30**, Google analogies **46.4%** — at or above published word2vec-on-text8 results on all three. See [Results](#1-results).
- **What I'd do next:** carry the vectors into the downstream [content-scoring project](https://github.com/RoshBeed/ai-residency/tree/main/services/content-scoring) and measure whether the better intrinsic scores move the extrinsic one — the sweep says window size and negative count are the knobs *not* to spend more time on.

This service trains the word vectors the rest of the portfolio builds on — word2vec with negative
sampling [1, 2], no pretrained models, published to the Hugging Face Hub where the content-scoring
project consumes them frozen, behind a FastAPI nearest-neighbour/analogy backend.

`text8` rather than a corpus of my own, because it is *the* word2vec benchmark corpus: published
scores are quoted against these exact 17M tokens, so a number here compares to something rather
than only to itself. Results first below; the method is behind them.

## Contents

1. [Results](#1-results) — the numbers, the curves, the embedding space
2. [How it works](#2-how-it-works) — distributional hypothesis to negative sampling
3. [Architecture](#3-architecture) — how the pieces fit together
4. [Implementation](#4-implementation) — the file map
5. [Usage](#5-usage) — train it, sweep it, deploy it
6. [Related work](#6-related-work), [references](#references), [dependencies](#dependencies)

## 1. Results

Trained from scratch on `text8`, scored against the three standard intrinsic benchmarks, next to
published word2vec results on the same corpus:

| benchmark | untrained | **skip-gram** | CBOW | published word2vec | coverage |
|---|---|---|---|---|---|
| WordSim-353 (Spearman ρ) | 0.028 | 0.726 | **0.730** | 0.68 | 99.4% |
| SimLex-999 (Spearman ρ) | 0.008 | **0.297** | 0.290 | 0.30 | 99.3% |
| Google analogies (accuracy) | 0.000 | **0.464** | 0.408 | 0.38 | 62.8% |

Both architectures beat the published references on WordSim-353 and the analogy set, and land
within 0.01 on SimLex-999. Each number was reproduced by re-running its configuration from scratch
after the sweep — skip-gram 0.4956 both times, CBOW 0.4758, to four decimals.

**CBOW wins WordSim-353**, 0.730 against 0.726 — the highest of all fifty runs — and loses the
analogy set by 5.6 points. Skip-gram is what the API serves, but "skip-gram is better" would be
too strong: it is better at analogies and marginally worse at relatedness. Both are published;
`AI_RESIDENCY_ARCHITECTURE=cbow` switches without a rebuild.

**The untrained column is not a formality.** It is what caught the state this service was in
before: its shipped checkpoint reported a loss of 10.02, which reads like a trained model right up
until you notice an untrained one reports 11.09.

### 1.1 What it actually learned

Nearest neighbours from the served checkpoint, which is the check no aggregate score can stand in
for:

| word | five nearest by cosine |
|---|---|
| `king` | kings, throne, fitzalan, ermengarde, son |
| `france` | french, paris, bienville, spain, nazaire |
| `computer` | computers, hardware, computing, software, datasheet |
| `three` | four, two, five, one, six |
| `river` | tributaries, rivers, tributary, headwaters, confluence |
| `war` | troops, allied, army, fought, fended |

`three → four, two, five, one, six` is the one to look at. Nothing in the corpus labels those as
numbers; they cluster because they appear in the same positions, which is the distributional
hypothesis from [2.1](#21-the-distributional-hypothesis) showing up directly in the output.

And four analogies, straight off the running `/analogy` endpoint, with the three query words
excluded:

| query | top 3 |
|---|---|
| `bigger - big + small` | **smaller**, larger, large |
| `walked - walk + run` | **ran**, running, runs |
| `paris - france + italy` | bologna, turin, **rome** |
| `king - man + woman` | throne, ermengarde, anjou |

Three of the four put the intended answer in the top three, two of them first. The fourth is the
example everyone quotes, and it misses — so it is worth saying exactly why, because the obvious reading is wrong.

**It is not that the gender direction was never learned.** Four of six family analogies land on the
right word outright:

| query | top 3 |
|---|---|
| `father : mother :: son : ?` | **daughter**, wife, granddaughter |
| `brother : sister :: grandson : ?` | **granddaughter**, daughter, grandchildren |
| `man : woman :: son : ?` | **daughter**, wife, mother |
| `boy : girl :: father : ?` | **mother**, wife, daughter |

The axis is there. What fails is `king` in particular, and the neighbours say why: `throne`, `anjou`,
`childless`, `marrying`. In 17M tokens of Wikipedia, `king` occurs overwhelmingly in regnal lists
and succession prose, so its vector encodes *monarchy and lineage* far more strongly than it
encodes *male*. Push it along the gender axis and you land on the nearest female role in that same
dynastic context. `queen` comes 15th; `daughter` is 3rd once candidates are restricted to the
30,000 most frequent words as the benchmark does — a coherent answer to the question the vectors
actually represent, and not the one the demo wants.

Corpus size is the rest of it. `queen` appears 1,940 times here and `woman` 1,753; the canonical
demo comes from Google News at 100 billion words, about 6,000 times more text. Category accuracy
matches that story — `gram6-nationality-adjective` 93.2%, `capital-common-countries` 86.2%,
`gram8-plural` 71.4%, against `family` at 39.9%, the weakest of the semantic categories.

Reporting the miss rather than dropping it is the point: at 46.4% overall, showing only the
queries that work would be choosing the sample to fit the claim.

### 1.2 The curves

`text8` is 17.0M tokens and 253,854 distinct words; at `min_count=5` the vocabulary is **71,290**,
keeping 98.3% of tokens — the standard figure for this corpus, and a useful check that the
tokenization matches everyone else's. Subsampling leaves about 5.6M training positions per epoch,
redrawn each epoch. Analogy accuracy splits into **60.2% semantic** and **39.5% syntactic**;
coverage there is 62.8% because answers are restricted to the 30,000 most frequent words, and the
accuracies are over the covered questions rather than diluted by the rest.

**Figure 1.** Benchmarks over training, both architectures, against the published references.

![Three panels — WordSim-353, SimLex-999 and Google analogies — plotted against epoch for CBOW and skip-gram, each with a dashed line marking the published word2vec-on-text8 result](assets/benchmarks-vs-epoch.png)

**Figure 2.** Training loss, each architecture divided by its own random-init baseline.

![Loss against optimizer step for CBOW and skip-gram, each normalised by its own random-init value, with a dashed line at 1.0 marking the untrained model](assets/training-loss.png)

Loss is plotted relative to each run's own starting point rather than in absolute terms. CBOW at 5
negatives starts at 4.16 and skip-gram at 41.59, because skip-gram scores $2w$ positives per row
where CBOW scores one — so an absolute plot would compare the number of terms in a sum, not the
quality of two models. The alternative, a second y-axis, can be made to show whatever you like.

**Read Figures 1 and 2 together.** CBOW drives its loss down to 0.15 of its starting value while
skip-gram only reaches 0.33 — and skip-gram produces the better vectors. The objective and the
thing you actually want are related but not the same, which is why the sweep ranks on benchmark
score and why a loss curve alone would have picked the wrong architecture.

### 1.3 What the sweep found

48 runs over 12 hours on one M1 Max, Bayesian search over architecture, embedding size, window,
negatives, subsampling threshold and learning rate, ranked on the composite score.

**Figure 3.** Composite score against each hyperparameter, one panel each, coloured by architecture.

![Six scatter panels showing composite score against architecture, embedding size, window, negatives, subsampling threshold and learning rate across 48 sweep runs](assets/sweep-results.png)

What actually moved the number, strongest first:

- **Subsampling threshold, by a distance.** $t = 10^{-3}$ scored 0.290 against 0.486 at
  $t = 10^{-5}$ — a wider spread than any other parameter produced. Discarding frequent words
  aggressively is the single highest-leverage setting here, which is consistent with it being the
  one the previous implementation did not do at all.
- **Architecture.** Skip-gram averaged 0.485 against CBOW's 0.395, and every one of the top eight
  runs is skip-gram. But the split is not as clean as the means suggest: **CBOW's one well-tuned
  run scored the highest WordSim-353 of the entire sweep, 0.730**, and lost on analogies (0.408 vs
  0.464), not on similarity. Skip-gram wins the composite because analogies are a third of it.
- **Learning rate**, a clear inverted-U peaking near $2 \times 10^{-3}$. Both tails cost real
  accuracy.
- **Window and negatives: essentially nothing.** Holding architecture, dimension and threshold at
  their best values, window 8 vs 10 is 0.4883 vs 0.4847, and 5 / 10 / 15 negatives are 0.4844 /
  0.4873 / 0.4862. Worth knowing that these are the knobs *not* to spend the next sweep on.

**Two caveats, because the sweep design creates them.** Bayesian search exploits: 45 of 48 runs are
skip-gram, 46 of 48 use $t = 10^{-5}$, and 43 of 48 use 300 dimensions. So the top-of-table ranking
is solid, but the per-hyperparameter marginals rest on one or two runs each at the losing settings
— "subsampling mattered most" is a direction, not a measured effect size. And the window/negatives
comparisons above are drawn from a slice with everything else held fixed, which is the only way to
read a marginal out of a search that was never designed to be balanced. A grid would answer these
properly; it would also have cost far more than twelve hours.

### 1.4 More epochs made it worse

**26 of the 48 runs peaked before their final epoch.** Extending the winning configuration from 10
epochs to 20 confirmed it directly: the score peaked at epoch 7 and then fell away, WordSim-353
from 0.722 to 0.684 and SimLex-999 from 0.295 to 0.244, while analogy accuracy held flat.

Part of that is ordinary over-training, and part is an interaction worth naming: the learning rate
decays linearly to zero across `num_epochs`, so lengthening a run also stretches the schedule. At
20 epochs the rate is still high where the score peaks, and the run never gets the annealing that
the 10-epoch schedule delivers exactly where it matters. The number of epochs is not an
independent knob you can raise for free — it is part of the schedule.

The trainer keeps the best epoch by score rather than the last one, so this costs quality but never
a checkpoint.

### 1.5 The embedding space

**Figure 4.** t-SNE projection of the 200 most frequent words. Nothing labelled any of these
groups; they fall out of co-occurrence alone.

![t-SNE projection of the learned skip-gram embedding space, the 200 most frequent words, with number words, nationalities and modal verbs each falling into their own region](assets/tsne_embeddings.png)

The legible clusters sit at the edges: the number words (`zero` through `nine`) in one tight group
at lower left, nationalities (`french`, `german`, `english`, `british`, `american`) just above
them, and the modals (`can`, `may`, `will`, `would`, `could`) together on the right. The centre is
function words, which are frequent, appear in every context, and are correspondingly hard to
separate — 200 words rather than 300 because past roughly that the centre packs into illegibility.

**Figure 5.** Country-capital offsets in the trained space — the same plot as Mikolov et al.'s
Figure 12, but drawn on these vectors rather than the paper's.

![PCA projection of ten country vectors each joined by an arrow to its capital city, most offsets running up and to the left at similar angles, with china-to-beijing a much longer outlier](assets/country-capital-pca.png)

Figure 12 further down is the paper's version. This is the same claim made about the
vectors this service actually trained, which is the one worth checking — and it is worth reporting
what it shows rather than what the paper's shows.

Nine of the ten offsets run up and to the left at broadly similar angles, which is the relation
being encoded as a direction. They are not the clean parallel bundle of the published figure: the
lengths vary noticeably, and `china → beijing` is a long outlier running almost horizontally. That
is consistent with the number in the table — 60.2% on semantic analogies means the relation is
real and roughly consistent, not that it is exact.

## Quickstart

```bash
# one venv for the whole workspace
make install

# FastAPI on :8000, interactive docs at /docs
make serve SERVICE=word-embeddings

# train on text8 -- the corpus is fetched from the Hub on first use
make train SERVICE=word-embeddings
```

Stages specific to this service live in its own Makefile (`make -C services/word-embeddings help`):

- `train` — Train word vectors on text8 (~75s/epoch on an M1 Max; 10 epochs by default)
- `train-sample` — Train on a tiny slice of the corpus — seconds, writes a separate checkpoint
- `time` — Measure throughput and project how long a full run takes, without training
- `sweep` — Run a W&B sweep agent: `make sweep SWEEP=<id> [HOURS=12] [COUNT=n]`
- `evaluate` — Score the trained checkpoint on WordSim-353, SimLex-999 and the Google analogies
- `visualise` — Score the checkpoint and write a t-SNE plot of the embedding space to assets/

Checkpoints are resolved locally first, then from the Hugging Face Hub, so serving works on a host that has never trained anything.

## 2. How it works

Turning free text into something a model can use is the problem this repo is about. A title
doesn't reduce to numbers the way a timestamp or comment count does, and unlike a bounded category
("city" → population, income, crime rate), there's no fixed decomposition to fall back on — the
space of possible titles is unbounded.

**Figure 6.** Ordinal, one-hot, and learned encodings of a single word (`master`, one of five
education levels).

![Four encodings of "master" side by side: ordinal (3), years-of-schooling (17), one-hot ([0,0,0,1,0]), and a 5-dimensional learned embedding — each traced through a small lookup diagram](assets/fig5-word-encodings.png)

Ordinal and one-hot encoding, the two standard categorical encodings, both fail on individual
words, for the same reason they fail on any large, unordered category. What actually works, and
what I built: a **learned embedding** — a trainable vector per word, positioned so similar words
land close together. `nn.Embedding` in `word_embeddings/model.py` is that "Learned Embeddings"
column, for real. Everything below is how that training works.

### 2.1 The Distributional Hypothesis

The training signal here comes from the distributional hypothesis [3]: words that show up in
similar contexts tend to mean similar things. Classic illustration — you've probably never seen
the word *bardiwac* before, but you can guess what it means just from context:

> *"He handed her a glass of **bardiwac**."*
> *"I dined off bread, cheese and this excellent **bardiwac**."*

Glasses, dinner, pairing with food — that's enough to tell you *bardiwac* is a drink, probably
wine. Word2vec [1] turns that same intuition into a training signal directly, with zero labeling
cost. Here's how, piece by piece.

### 2.2 Tokenization: why this service does not lemmatize

**Figure 7.** Stemming versus lemmatization — the usual way to fold `run`/`running`/`ran` onto one
vocabulary entry.

![Two rows of tokenizer boxes: stemming maps run/running/runs to id 72 but fails on "ran"; lemmatization maps run/running/runs/ran to 72 and "was" to "be" (id 22)](assets/fig6-stemming-lemmatization.png)

**This service does neither, deliberately.** `text8` arrives lowercased and stripped to `[a-z ]`,
so there is nothing left to normalize — but the real reason is that folding those forms together
would destroy over half the benchmark.

Nine of the fourteen Google analogy categories are syntactic: `gram3-comparative` (good:better),
`gram7-past-tense` (walk:walked), `gram8-plural` (car:cars). They are 10,675 of the 19,544
questions. Lemmatizing collapses `walk`, `walks` and `walked` onto one vector, which makes **55% of
the analogy set unanswerable by construction**.

An earlier version of this pipeline did lemmatize. That code still lives in
`word_embeddings/etl.py`, building the Hacker News corpus the sibling content-scoring service
consumes; nothing on this training path calls it.

**Figure 8.** Tokenization and embedding lookup. Ids are assigned in descending frequency order, so
"the 300 most frequent words" is the slice `[:300]`.

![A tokenizer mapping "hello", "polymorphism" (split into polym/orphism/circul/ation), and "code" to vocabulary indices 22172-775, with "orphism" (index 28611) highlighted feeding into an embedding matrix row [0.34, -0.5, 0.15, 0.19]](assets/fig7-tokenization-embedding-lookup.png)

### 2.3 CBOW and Skip-gram

Word2vec [1] turns the distributional hypothesis into a prediction task two ways: **CBOW** predicts
a centre word from its surrounding context; **skip-gram** predicts the context from the centre word.

**Figure 9.** The two architectures, Figure 1 from the original paper [1].

![Figure 1 from Mikolov et al. 2013: the CBOW architecture (four context words w(t-2)..w(t+2) summed into a projection layer predicting w(t)) beside the Skip-gram architecture (w(t) projected out to predict w(t-2)..w(t+2))](assets/fig8-cbow-skipgram-paper.png)

This service trains **both**, and the sweep in [Results](#1-results) picks between
them. They are one class in `libs/word2vec`, not two: the architecture decides which vectors get
combined, the objective decides how a score becomes a loss, and the two axes are independent.

**Figure 10.** The CBOW forward pass with real tensor shapes — context in, embeddings looked up,
averaged, projected back onto the vocabulary.

![CBOW forward pass: "Hello my ? is Bes" tokenized to ids 72/44/93/11, embedded to a 4x9 matrix, averaged to 1x9, projected via nn.Linear to 1x128 logits predicting "name"; alongside the exact CBOW class source code with matching vocab dict](assets/fig9-cbow-forward-pass.png)

Scored against the whole vocabulary, that is a softmax:

$$
\mathcal{L}_{\text{softmax}} = -\log \frac{\exp(\mathbf{v}_c^{\top}\mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{v}_c^{\top}\mathbf{v}_w)}
$$

Exact, and $O(|V|)$ per example. At 71,290 words that is the cost the next section exists to avoid.

### 2.4 Training Objective: Negative Sampling

That full softmax has to touch every word in the vocabulary on every single training step just to
get one gradient. Too slow for a real vocabulary. Negative sampling [2] replaces it with cheap
binary classification instead: does this (context, target) pair look real, or fake?

**Figure 11.** Negative sampling objective — the boxed loss $J(v_c, v_w, v_{n_1}, \ldots, v_{n_k})$
matches the equation below it, and the sigmoid/logarithm reference curves show why $\log
\sigma(\cdot)$ is the right shape for a real-or-fake binary classifier.

![Skip-gram slide with the negative-sampling loss formula boxed in red, plus reference plots of the sigmoid and logarithm functions used in that loss](assets/fig10-negative-sampling.png)

For each real context/target pair, $k$ random negative words are sampled, giving the loss:

$$
\mathcal{L}_{\text{NS}} = -\log \sigma(\mathbf{v}_c^{\top} \mathbf{v}_t) - \sum_{i=1}^{k} \log \sigma(-\mathbf{v}_c^{\top} \mathbf{v}_{n_i})
$$

`Word2Vec._negative_sampling_loss()` in
[`libs/word2vec`](https://github.com/RoshBeed/ai-residency/tree/main/libs/word2vec) is this loss
directly — `positive` and `negative` are the two terms above. Only $k+1$ embeddings get touched per
step instead of the whole vocabulary, which is what makes training on a real-sized corpus practical
at all.

Which distribution the $k$ negatives are drawn from matters more than it looks. Uniform sampling
spends almost every negative on a rare word the model already scores near zero; the raw unigram
distribution spends almost all of them on `the`. Mikolov et al. raise the frequencies to the 3/4
power, which sits between the two, and that is what `NegativeSampler` implements.

One asymmetry worth naming, because it decides whether the two architectures are comparable at all:
CBOW scores one positive per row and skip-gram scores $2w$ of them. Drawing $k$ negatives per *row*
would leave skip-gram with ten positives pulling scores up against five pushing them down. The
negatives are drawn per positive pair instead, so both architectures see the same ratio at the same
`num_negatives`.

### 2.5 Embedding Evaluation

Once I had trained embeddings, the obvious next question was whether they were any good — and
answering that without labeled data took some creativity.

#### 3.5.1 Analogy and Similarity Tests

There are no labels anywhere in this pipeline, just raw text, so I needed label-free ways to check
the embeddings actually learned something. Two checks, both lifted from the original word2vec
papers. First, the classic vector-arithmetic analogy test:
$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}$ should land near
$\mathbf{v}_{\text{queen}}$, its nearest neighbor by cosine similarity if the embedding space is
any good.

**Figure 12.** Vector-arithmetic analogy test, Figure 2 from Mikolov et al. 2013b [2]. A real 2D PCA
projection of the paper's own trained Skip-gram vectors: each country's offset to its capital city
is roughly the same direction and magnitude across 15 pairs — the same $v_a - v_b + v_c \approx
v_d$ arithmetic as king−man+woman≈queen, just shown across a whole relation instead of one example.

![Two-dimensional PCA scatter plot titled "Country and Capital Vectors Projected by PCA": 15 countries (China, Russia, Japan, Turkey, Poland, Germany, France, Italy, Greece, Spain, Portugal, and others) each connected by a parallel dashed arrow to their capital city (Beijing, Moscow, Tokyo, Ankara, Warsaw, Berlin, Paris, Rome, Athens, Madrid, Lisbon), the arrows visibly near-parallel across the whole plot](assets/fig11-country-capital-analogy.png)

I score three standard benchmarks rather than hand-picked examples, all in
`word_embeddings/benchmarks.py` and all run at the end of every epoch so I can watch quality climb
during a run rather than only after it:

| benchmark | size | what it measures |
|---|---|---|
| **WordSim-353** | 353 pairs | relatedness, by Spearman rank correlation against human scores |
| **SimLex-999** | 999 pairs | *similarity* specifically — built so a model that only captures relatedness does badly |
| **Google analogies** | 19,544 questions | analogy accuracy, 5 semantic categories and 9 syntactic |

Spearman rather than a straight difference: the question is not "are the numbers equal" but "do
they rank the pairs the same way".

An earlier version of this section reported a Spearman correlation of **1.0**. It was computed over
three hand-picked pairs, and with $n = 3$ the statistic can only take the values
$\{-1, -0.5, 0.5, 1\}$ — so it carried about one bit of information and happened to land on the
good one. Three real benchmarks are the fix.

Two details decide whether an analogy score means anything:

- **All three query words are excluded from the answer.**
  $\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}}$ lands nearest
  to `king` itself far more often than to `queen`, so a scorer that does not exclude $a$, $b$ and
  $c$ reports a near-constant zero no matter how good the vectors are.
- **Candidates are restricted to the 30,000 most frequent words**, the usual convention. Searching
  the full tail lets a rare word with an under-trained vector win by accident, and makes the score
  depend on `min_count` rather than on the embeddings.

I also report **coverage** — the fraction of pairs or questions whose words are in the vocabulary —
next to every score. A correlation over the 60% of pairs a small vocabulary happens to cover is not
comparable to one over 95%, and quoting the score alone is the easiest way to overstate a result by
accident.

Everything above goes to Weights & Biases against `epoch`, alongside per-step loss and learning
rate against `train/step`, and a table of nearest neighbours for a fixed probe list each epoch —
which is the cheapest honest signal that a run is working. A loss curve falls for a model that has
learned only word frequency; `king → queen, prince, throne` does not happen by accident.

#### 3.5.2 Dimensionality Reduction

Those checks only cover a handful of hand-picked pairs. To eyeball the *entire* embedding space
(thousands of words, hundreds of dimensions), I used **t-SNE** to compress every embedding down to
2D while preserving which points were close together in the original space, so it can be
scattered on an ordinary chart. `word_embeddings/evaluate.py` does this — see
[Section 6](#1-results) for the actual plot.

### 2.6 Relation to Transformer Language Models

Everything above shares one constraint: CBOW and Skip-gram only ever look at a small, fixed window
of neighbouring words. Swap that fixed window for self-attention over the whole sequence and you
land on the two pretraining setups behind most modern transformers:

**Figure 13.** Fixed-context word2vec compared with attention-based masked and causal language
modeling, using the running `"Hello my name is Bes"` example throughout: word2vec predicts one
masked word from an averaged window, BERT predicts multiple masked positions from the full
bidirectional sequence, GPT predicts the next token causally.

![Three panels side by side: word2vec predicting the masked "name" from context; BERT predicting two masked tokens ("Hello" and "name") from the full bidirectional sequence; GPT predicting the next token "Bes" causally from <s> Hello my name is](assets/fig12-word2vec-bert-gpt.png)

I implement self-attention itself separately; see the
[vision transformer project](https://github.com/RoshBeed/ai-residency/tree/main/services/vision-transformers) for a from-scratch implementation.

## 3. Architecture

Before the method — here's how the whole thing is wired together at a glance. Two stages: build
the corpus, pretrain the embeddings. The trained vectors are then published for the
content-scoring project to consume frozen.

**Figure 14.** Pipeline architecture.
```mermaid
flowchart LR
    data[("text8 corpus<br/>Hugging Face Hub")] --> etl["Vocabulary & encoding<br/>word_embeddings/corpus.py"]
    etl --> cbow["CBOW / skip-gram training<br/>word_embeddings/train.py"]
    cbow --> eval["Intrinsic evaluation<br/>word_embeddings/benchmarks.py"]
    eval --> hub[("Hugging Face Hub<br/>cbow_model.pt")]

    style data fill:#e1f5fe,stroke:#01579b
    style etl fill:#e1f5fe,stroke:#01579b
    style cbow fill:#f3e5f5,stroke:#4a148c
    style eval fill:#f3e5f5,stroke:#4a148c
    style hub fill:#e8f5e8,stroke:#1b5e20
```

Same pipeline as a sequence diagram. Each `Note over` label marks a separate process run, not one
continuous script, matching the commands in [Section 5](#5-usage):

**Figure 15.** Sequence diagram of the pipeline.
```mermaid
sequenceDiagram
    actor main
    participant dp as data processor
    participant tok as tokeniser
    participant cbow

    Note over main,tok: Process 1 — python -m word_embeddings.etl
    activate main
    main->>+dp: download(data)
    dp-->>-main: data
    main->>+dp: clean(data)
    dp-->>-main: data
    main->>+tok: create(data)
    tok-->>-main: tokens
    deactivate main

    Note over main,cbow: Process 2 — python -m word_embeddings.train
    activate main
    main->>+cbow: create(cbow model)
    main->>cbow: train_model(tokens)
    cbow-->>-main: save(cbow model)
    deactivate main
```

## 4. Implementation

Mapping the architecture from [Architecture](#3-architecture) onto the actual file layout:

```
word_embeddings/          The word-embedding package
├── corpus.py              Fetch text8, build the frequency-ordered vocabulary, encode it
├── data.py                Subsampling + the tensor batcher that feeds training
├── model.py               build_word2vec — this service's view of libs/word2vec
├── train.py               train — the epoch loop, tracking and checkpointing
├── benchmarks.py          WordSim-353, SimLex-999 and Google-analogy scoring
├── evaluate.py            Score a saved checkpoint; t-SNE projection
├── report.py              The figures in Section 6, drawn from the W&B runs
└── etl.py                 The Hacker News corpus, for the sibling content-scoring service
config/__init__.py        Config(BaseConfig) — every hyperparameter and path in one dataclass
serving/api.py            FastAPI app: loads the checkpoint, exposes /nearest and /analogy
checkpoints/              Trained artifacts, under checkpoints/<architecture>/
data/                     text8 and the benchmark files (gitignored, fetched from the Hub)
tests/                    Vocabulary, batcher, benchmark-scoring and reproducibility tests
```

`word_embeddings` is a plain Python package run as modules — `python -m word_embeddings.train` —
with `config/` and `serving/` alongside it. No install step, no `PYTHONPATH` juggling, no
`sys.path` manipulation anywhere.

The word2vec math itself is not here. It lives in
[`libs/word2vec`](https://github.com/RoshBeed/ai-residency/tree/main/libs/word2vec), shared with
the vector-search-retrieval service, which had grown its own copy — the two were not rival CBOWs
but the same pair of matrices under different objectives. This package is the service's view of
it: which corpus, which hyperparameters, how a run is tracked and checkpointed.

Shared behaviour — device selection, logging, checkpoint format, seeding, Hub upload and W&B
tracking — comes from [`libs/common`](https://github.com/RoshBeed/ai-residency/tree/main/libs/common), pinned by tag, so it
is defined once for the whole portfolio instead of reimplemented per repo.

## 5. Usage

There is no corpus stage — `text8` is fetched from the Hub on first use.

**Figure 16.** Reproduction pipeline.
```mermaid
flowchart LR
    fetch["1. Fetch text8<br/>+ benchmarks"] --> pretrain["2. Train<br/>+ score each epoch"] --> hub["3. Publish<br/>to the Hub"]
```

```bash
make install
make train SERVICE=word-embeddings                # ~75s/epoch, 10 epochs

make -C services/word-embeddings train-sample     # a tiny slice, seconds
make -C services/word-embeddings time             # measure throughput, train nothing
make -C services/word-embeddings evaluate         # score a saved checkpoint
make serve SERVICE=word-embeddings                # the API, no Docker
```

Credentials are optional. Without `WANDB_API_KEY` the run logs offline and syncs later; without
`HF_TOKEN` the checkpoint stays local. Both are read from an existing `wandb login` or
`huggingface-cli login`, so a logged-in machine needs no `.env`.

Every hyperparameter is a field on `Config`, overridable from the environment:

```bash
AI_RESIDENCY_NUM_EPOCHS=1 AI_RESIDENCY_EMBED_SIZE=64 make train SERVICE=word-embeddings
AI_RESIDENCY_ARCHITECTURE=skipgram make train SERVICE=word-embeddings
```

**Run it locally on Apple Silicon.** `get_device()` prefers CUDA, then MPS, then CPU, and Docker
Desktop on macOS has no MPS passthrough — so training in a container here is CPU-only and about
half the speed.

The best epoch saves to `checkpoints/<architecture>/<architecture>_model.pt` and pushes to the Hub,
with the vocabulary inside it so the matrix and the mapping that indexes it cannot drift apart. No
optimizer state: Adam's moment buffers are the size of the two embedding matrices, which made the
artifact 515MB where 173MB says everything, and nothing downstream reads them.

### 5.1 Sweeps

`sweep.yaml` defines the search; W&B runs it. Its parameter names are the `AI_RESIDENCY_*`
overrides, so the agent needs no sweep-specific plumbing in the trainer.

```bash
wandb sweep services/word-embeddings/sweep.yaml    # prints a sweep id
make -C services/word-embeddings sweep SWEEP=<id> HOURS=12
```

Two settings there matter more than they look:

- **The metric is `best/score`, not loss.** Loss is not comparable across these runs: the
  random-init value is $(p + pk)\ln 2$, which is 4.16 for CBOW at 5 negatives and 41.59 for
  skip-gram — so minimising it would rank runs by architecture and negative count rather than by
  how good the vectors are.
- **`min_count` and `num_epochs` are fixed.** `min_count` changes the vocabulary, which changes
  benchmark coverage, which makes scores incomparable — a sweep varying it would reward runs for
  being evaluated on an easier subset. Fixing `num_epochs` keeps run times close enough for a
  wall-clock budget to mean anything.

`HOURS` bounds the sweep by wall clock rather than run count, because run times span 6 to 31
minutes and a count chosen for the average lands anywhere between half and double the budget.
Stopping early costs nothing — finished runs are recorded and the next agent reads them back.

### 5.2 Deployment

```bash
make build SERVICE=word-embeddings                # serving + training images
make -C deploy up                                 # the whole stack
cd deploy && docker compose --profile train run --rm train-word-embeddings
```

The checkpoint is not baked into either image: serving pulls from the Hub, so a container always
serves the most recently trained vectors rather than a snapshot from build time. Training is gated
behind a Compose profile so a plain `docker compose up` never starts one by accident.

The UI is not in this repo. It is a thin HTTP client in `services/frontend` calling this API — see
the [Quickstart](#quickstart) for running both from Docker.

## 6. Related Work

- [the content-scoring project](https://github.com/RoshBeed/ai-residency/tree/main/services/content-scoring) —
  the regression head that consumes these frozen vectors, and where the honest downstream result
  lives.
- [the dual-encoder retrieval project](https://github.com/RoshBeed/ai-residency/tree/main/services/vector-search-retrieval) —
  recurrent (GRU) query/document towers trained with a triplet objective, an alternative to
  average-pooled embeddings for sequence representation.
- [the vision transformer project](https://github.com/RoshBeed/ai-residency/tree/main/services/vision-transformers) — self-attention and transformer encoder/decoder
  implemented from scratch.
- [the RLHF/PPO project](https://github.com/RoshBeed/ai-residency/tree/main/services/rlhf-ppo) — reward modeling and policy optimization; the same
  scalar-head-over-frozen-embeddings shape, trained on human preferences instead of raw counts.

## References

[1] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word
Representations in Vector Space.* arXiv:1301.3781.

[2] Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed
Representations of Words and Phrases and their Compositionality.* arXiv:1310.4546.

[3] Harris, Z. S. (1954). Distributional Structure. *Word*, 10(2–3), 146–162.

## Acknowledgments

Built with Peter O'Keeffe, Ben Williams, and Miguel Parracho. Figures 9 and 12 reproduce Figure 6
and Figure 2 from the original word2vec papers [1, 2] directly.

## Dependencies

PyTorch · FastAPI · NLTK · Weights & Biases · Hugging Face Hub · PostgreSQL · Docker ·
Docker Compose · Make
