.DEFAULT_GOAL := help
.PHONY: etl etl-verify mirror extend-vocab train train-all fusion-curve fusion-verify evaluate evaluate-untrained sweep report onnx graph predict serve test help

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

# Where the title and the metadata first meet, swept as a hyperparameter. Layers before
# `fusion_layer` are block-diagonal and layers from it on are shared, so `1` is early
# fusion and a larger value delays the meeting without changing anything else.
#
# A shell grid rather than a W&B sweep because `fusion_layer` is only meaningful up to
# the depth of the shape, and `method: grid` cannot express that -- it would spend a
# third of the budget on duplicates that clamp onto each other.
#
# The per-run settings go through `env` rather than prefixing the command as
# `VAR=value uv run ...`. The retry loop puts a `while` between the two, and
# `A=1 B=2 attempt=1` is then parsed as three plain shell assignments that are never
# exported -- so a run trains the *defaults* while being recorded under the shape that
# was asked for. Four runs were logged that way before it was caught.
#
# Each combination is retried twice before the grid gives up. A 28-hour run met exactly
# one transient failure -- `wandb ... the service process is busy and did not respond in
# time` on run 60 of 490 -- and a bare `|| exit 1` turned that into the end of the
# experiment. A flaky API call is not a reason to abandon twenty-five hours of work.
#
# Ten seeds a point, not three. The twelve baseline runs measured a seed spread of 0.106
# Spearman on late fusion against 0.026 on early, so a three-seed mean carries a standard
# error wider than the effect being measured. At n=10 the curve can also show the
# *distribution*, which is the more useful answer if late fusion turns out to be bimodal.
FUSION_SHAPES ?= 320,256 256,256 320,256,192 256,256,256 320,256,192,128 \
                 256,256,256,256 320,256,192,128,96 256,256,256,256,256 \
                 320,256,192,128,96,32 128,128,128,128,128,128 \
                 640,512,384 64,64,64 256,64,256
FUSION_SEEDS  ?= 42 43 44 45 46 47 48 49 50 51
PROGRESS      := assets/fusion-curve.progress

fusion-curve: ## Sweep the fusion point across shapes and seeds: make fusion-curve [RESUME=no]
	@mkdir -p assets
	@if [ "$(RESUME)" = "no" ]; then rm -f $(PROGRESS); fi
	@touch $(PROGRESS)
	@echo "$$(date +%H:%M:%S) starting; $$(wc -l < $(PROGRESS) | tr -d ' ') combinations already done"
	@for widths in $(FUSION_SHAPES); do \
	  depth=$$(echo $$widths | tr ',' '\n' | wc -l | tr -d ' '); \
	  point=1; \
	  while [ $$point -le $$depth ]; do \
	    for seed in $(FUSION_SEEDS); do \
	      key="$$widths|$$point|$$seed"; \
	      if grep -qxF "$$key" $(PROGRESS); then continue; fi; \
	      echo "$$(date +%H:%M:%S) === widths=$$widths fusion_layer=$$point seed=$$seed"; \
	      attempt=1; \
	      while : ; do \
	        if env AI_RESIDENCY_HIDDEN_WIDTHS=$$widths \
	               AI_RESIDENCY_FUSION_LAYER=$$point \
	               AI_RESIDENCY_SAVE_CHECKPOINTS=false \
	               AI_RESIDENCY_PROMOTE=false \
	               AI_RESIDENCY_LOG_TABLES=false \
	               AI_RESIDENCY_HF_REPO_ID="" \
	           uv run python -m content_scoring.train --fusion late --seed $$seed; then \
	          echo "$$key" >> $(PROGRESS); break; \
	        fi; \
	        if [ $$attempt -ge 3 ]; then \
	          echo "$$(date +%H:%M:%S) !!! giving up on $$key after $$attempt attempts"; \
	          exit 1; \
	        fi; \
	        echo "$$(date +%H:%M:%S) retry $$attempt for $$key in 60s"; \
	        attempt=$$((attempt + 1)); sleep 60; \
	      done; \
	    done; \
	    point=$$((point + 1)); \
	  done; \
	done
	@echo "$$(date +%H:%M:%S) done; $$(wc -l < $(PROGRESS) | tr -d ' ') combinations recorded"

# A run exiting zero is not the same as its results being stored. Run this after the
# grid: it compares the progress file against what actually reached W&B, and FIX=1
# requeues anything whose summary never landed.
fusion-verify: ## Check the grid's results actually reached W&B: make fusion-verify [FIX=1]
	uv run python -m content_scoring.verify_curve \
	  --shapes $(FUSION_SHAPES) --seeds $(FUSION_SEEDS) $(if $(FIX),--fix,)

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

# Netron renders an ONNX graph; `uvx netron <file>` runs it without installing anything.
# Exported with small embedding tables by default -- the real ones are 86,438 x 300 and
# draw as the same single Gather node, but add 104MB that makes the file slow to open.
# `--real-shapes` uses the true sizes.
onnx: ## Export each architecture to ONNX for Netron: make onnx
	uv run python -m content_scoring.export_onnx
	@echo ""
	@echo "  uvx netron services/content-scoring/assets/onnx/late.onnx"

# The demo page builds this from the running models on every request, so there is no
# file to open when the stack is down. Needs Graphviz for the image -- a system package,
# not something uv can install -- and writes the DOT either way.
graph: ## Draw the architecture diagram: make graph [FUSION=late] [FORMAT=svg]
	uv run python -m content_scoring.export_graph \
	  $(if $(FUSION),--fusion $(FUSION),) $(if $(FORMAT),--format $(FORMAT),)

predict: ## Score one post: make predict TITLE="..." [URL="..."]
	@test -n "$(TITLE)" || (echo 'Usage: make predict TITLE="Show HN: my thing" [URL="https://..."]' && exit 1)
	uv run python -c "from config import Config; from content_scoring.predict import Post, load_predictor, score_post; c=Config(); p=load_predictor(c); s,_=score_post(p, Post(title='''$(TITLE)''', url='''$(URL)'''), c); print(f'{s:.1f} upvotes')"

serve: ## Run the FastAPI service, no Docker
	uv run python -m serving.api

test: ## Run the unit tests (correctness, not model quality -- that is `evaluate`)
	uv run pytest -q

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
