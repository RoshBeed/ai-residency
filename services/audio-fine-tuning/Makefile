        docker-build-train docker-build-serve docker-push-train docker-push-serve \
        docker-up docker-train docker-down clean help

train:  ## Run the Part 1 fine-tuning loop and write a checkpoint
	uv run python -m audio_fine_tuning.training

demo:  ## Print the base-vs-fine-tuned transcript comparison for data/audio.mp3
	uv run python -m audio_fine_tuning.demo

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

prepare-data:  ## Part 2 pipeline: prepare -> manifest -> tokens -> explore
	uv run python -m data_prep.prepare_dataset
	uv run python -m data_prep.create_manifest
	uv run python -m audio_fine_tuning.add_emotion_tokens
	uv run python -m data_prep.explore_data

download-data:  ## Fetch the raw corpora (EmoDB downloads; CREMA-D is a slow git-lfs clone)
	uv run python -m data_prep.download_dataset

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
