.DEFAULT_GOAL := help
.PHONY: download-data prepare-data add-tokens train serve test help

download-data:  ## Fetch the raw corpora (EmoDB downloads; CREMA-D is a slow git-lfs clone)
	uv run python -m data_prep.download_dataset

prepare-data:  ## Organize by emotion, split train/val, write manifests, then explore
	uv run python -m data_prep.prepare_dataset
	uv run python -m data_prep.create_manifest
	uv run python -m data_prep.explore_data

add-tokens:  ## Extend the Whisper tokenizer with one special token per emotion
	uv run python -m speech_emotion_recognition.add_emotion_tokens

train:  ## Fine-tune Whisper to emit an emotion token (needs prepare-data + add-tokens)
	uv run python -m speech_emotion_recognition.training

serve:  ## Run the FastAPI backend on :8000
	uv run python -m serving.api

test:  ## Run the test suite
	uv run pytest -q

help:  ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
