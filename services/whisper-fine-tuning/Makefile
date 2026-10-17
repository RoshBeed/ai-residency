.DEFAULT_GOAL := help
.PHONY: train demo collateral serve test help

train:  ## Fine-tune on the single demo example and write a checkpoint
	uv run python -m whisper_fine_tuning.training

demo:  ## Print the base-vs-fine-tuned transcript comparison for data/audio.mp3
	uv run python -m whisper_fine_tuning.demo

collateral:  ## WER on other speech before and after the step (needs CREMA-D clips)
	uv run python -m whisper_fine_tuning.collateral

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
