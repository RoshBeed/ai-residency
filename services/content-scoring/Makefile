.DEFAULT_GOAL := help
.PHONY: etl etl-verify mirror extend-vocab train train-all evaluate evaluate-untrained sweep report predict serve test help

# The pipeline, in order: etl -> extend-vocab -> train -> evaluate. Every stage reads
# public repos, so the whole chain runs on a clean machine with no credentials. A token
# is needed only to push, and for data that happened once, via `mirror`.

etl: ## Fetch the Hacker News mirror and build the train/val/test splits
	uv run python -m content_scoring.etl

etl-verify: ## Cross-check the mirror against the live Hacker News API
	uv run python -m content_scoring.etl --source=api --max-items=$(or $(MAX_ITEMS),2000)

# Run once, by hand. The only place in the data path that needs HF_TOKEN: it copies the
# upstream dataset into our own repo so a published number cannot change under us when
# the upstream updates, which its README says it does regularly.
mirror: ## Create the raw mirror from upstream (needs HF_TOKEN; run once)
	uv run python -m content_scoring.etl --source=upstream

extend-vocab: ## Extend the text8 vocabulary onto the Hacker News titles and continue-train
	uv run python -m content_scoring.vocab --benchmark

train: ## Train one architecture: make train [FUSION=late] [SEED=42]
	uv run python -m content_scoring.train $(if $(FUSION),--fusion $(FUSION),) $(if $(SEED),--seed $(SEED),)

# The comparison table quotes these, not the sweep. The sweep searches the space; these
# twelve runs hold every hyperparameter equal across the four architectures and vary
# only the seed, which is what makes "late beats early by 0.02" a claim rather than a
# coincidence.
train-all: ## Train all four architectures at three seeds each (the comparison table)
	@for fusion in text_only features_only early late; do \
	  for seed in 42 43 44; do \
	    echo "=== $$fusion seed $$seed"; \
	    uv run python -m content_scoring.train --fusion $$fusion --seed $$seed || exit 1; \
	  done; \
	done

# Early fusion has to *discover* the grouping that late fusion is handed by
# construction, and discovering it costs depth. So the two should cross over: late ahead
# where capacity is scarce, early catching up once there is enough to learn the structure
# with. This varies depth alone, everything else at the tuned configuration, three seeds
# a point -- and writes nothing, since the checkpoints are not the output, the curve is.
capacity: ## Depth sweep, early vs late, 3 seeds a point (the crossover experiment)
	@for layers in 1 2 3 4; do \
	  for fusion in early late; do \
	    for seed in 42 43 44; do \
	      echo "=== $$fusion layers=$$layers seed=$$seed"; \
	      AI_RESIDENCY_NUM_LAYERS=$$layers AI_RESIDENCY_SAVE_CHECKPOINTS=false \
	      AI_RESIDENCY_HF_REPO_ID="" \
	      uv run python -m content_scoring.train --fusion $$fusion --seed $$seed || exit 1; \
	    done; \
	  done; \
	done

evaluate: ## Score every trained architecture against the test split
	uv run python -m content_scoring.evaluate $(if $(SPLIT),--split $(SPLIT),)

# Run this *before* any training. Every baseline should already have a number and the
# random-init rows should sit at about zero; an eval harness first exercised on a trained
# model gives you no way to tell a real score from a bug.
evaluate-untrained: ## Score untrained models, to check the harness itself
	uv run python -m content_scoring.evaluate --untrained

# HOURS bounds the sweep by wall clock, COUNT by number of runs. Runs here are minutes
# rather than the tens of minutes word-embeddings takes, so COUNT is usually the more
# useful bound -- an overnight budget would buy far more runs than Bayes can use.
#
# caffeinate because a laptop otherwise sleeps partway through; timeout is GNU
# coreutils, not macOS base -- `brew install coreutils`.
sweep: ## Run a W&B sweep agent: make sweep SWEEP=<id> [COUNT=60] [HOURS=n]
	@test -n "$(SWEEP)" || (echo "Usage: make sweep SWEEP=<sweep id> [COUNT=60] [HOURS=n]   (create one with: wandb sweep sweep.yaml)" && exit 1)
	caffeinate -is $(if $(HOURS),timeout $(HOURS)h,) uv run wandb agent $(if $(COUNT),--count $(COUNT),) $(SWEEP)

# Reads the harness's own JSON rather than recomputing anything, so a figure cannot
# disagree with the table it sits next to.
report: ## Write the write-up figures to assets/
	uv run python -m content_scoring.evaluate --json assets/evaluation.json
	uv run python -m content_scoring.report --report assets/evaluation.json

predict: ## Score one post: make predict TITLE="..." [URL="..."]
	@test -n "$(TITLE)" || (echo 'Usage: make predict TITLE="Show HN: my thing" [URL="https://..."]' && exit 1)
	uv run python -c "from config import Config; from content_scoring.predict import Post, load_predictor, score_post; c=Config(); p=load_predictor(c); s,_=score_post(p, Post(title='''$(TITLE)''', url='''$(URL)'''), c); print(f'{s:.1f} upvotes')"

serve: ## Run the FastAPI service, no Docker
	uv run python -m serving.api

test: ## Run the unit tests (correctness, not model quality -- that is `evaluate`)
	uv run pytest -q

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
