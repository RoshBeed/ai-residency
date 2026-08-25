"""Manifest-backed dataset: audio in, emotion token out.

`data_prep/create_manifest.py` writes one CSV per corpus and split, with two columns:
`audio_path` and `label`. That is the whole interface between the data pipeline and
training, deliberately -- the corpora disagree about directory layout and about how the
emotion is spelled inside a filename, and all of that is resolved before anything gets
here.

The manifests carry no transcript. EmoDB and CREMA-D are fixed-sentence corpora: every
speaker reads the same handful of sentences, and what varies is the delivery. So the
target is the emotion token alone, not a transcript with an emotion token attached.
"""

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import WhisperProcessor

from ai_residency_common import get_logger
from config import Config
from speech import encode_target, features_from_file

__all__ = ["EmotionClip", "EmotionDataset", "load_manifest_rows"]

logger = get_logger(__name__)

# One example: the log-mel features Whisper's encoder expects, and the label ids its
# decoder is trained to emit.
EmotionClip = tuple[torch.Tensor, torch.Tensor]


def load_manifest_rows(config: Config, split: str) -> list[tuple[Path, str]]:
    """Read every manifest for one split, across all configured corpora.

    Args:
        config: Supplies `datasets` and `manifest_path`.
        split: `"train"` or `"val"`.

    Returns:
        `(audio_path, emotion)` pairs. Corpora whose manifest has not been generated
        are skipped rather than raising: CREMA-D needs a multi-gigabyte clone that a
        given host may legitimately not have, and EmoDB alone is enough to train on.
    """
    rows: list[tuple[Path, str]] = []
    for spec in config.datasets:
        manifest = config.manifest_path(spec, split)
        if not manifest.exists():
            logger.info("No %s manifest for %s; skipping", split, spec.display_name)
            continue
        with manifest.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((Path(row["audio_path"]), row["label"]))
    return rows


class EmotionDataset(Dataset[EmotionClip]):
    """Log-mel features paired with the emotion token the decoder should emit.

    Attributes:
        rows: `(audio_path, emotion)` pairs read from the manifests.
    """

    def __init__(self, config: Config, processor: WhisperProcessor, split: str) -> None:
        """Build the dataset for one split.

        Args:
            config: Run configuration.
            processor: Whisper processor whose tokenizer already carries the emotion
                tokens. Passing it in rather than constructing one here keeps a single
                extended vocabulary shared with the model's resized embedding table.
            split: `"train"` or `"val"`.

        Raises:
            FileNotFoundError: If no manifest for this split exists at all.
        """
        self._config = config
        self._processor = processor
        self.rows = load_manifest_rows(config, split)
        if not self.rows:
            raise FileNotFoundError(
                f"No {split} manifests under {config.data_dir}. "
                "Build them with `make -C services/speech-emotion-recognition prepare-data`."
            )
        logger.info("%s split: %d clips", split, len(self.rows))

    def __len__(self) -> int:
        """Number of clips in this split."""
        return len(self.rows)

    def __getitem__(self, index: int) -> EmotionClip:
        """Load one clip and encode its label.

        Args:
            index: Row index.

        Returns:
            `(input_features, labels)` -- the encoder's log-mel input, and the token
            ids the decoder is trained to produce.
        """
        audio_path, emotion = self.rows[index]
        features = features_from_file(
            self._processor,
            audio_path,
            sample_rate=self._config.sample_rate,
            max_duration=self._config.max_duration_seconds,
        )

        # prefix=False is the correctness-critical part, not a tidy-up.
        #
        # This used to pass the tokenizer's full output through as labels, which meant
        # two things at once. Whisper's model prepends <|startoftranscript|> itself
        # when it shifts labels right, so the tokenizer's own copy made the decoder
        # input begin [SOT, SOT, ...]; and the language/task prefix pushed the emotion
        # token four steps in. Serving reads the *first* decoder step
        # (`serving/api.py` feeds [SOT] and takes logits[0, -1]), so it was reading the
        # position training had taught to emit <|startoftranscript|>.
        #
        # Nothing errored. It trained, the loss fell, and the answer came from the
        # wrong slot. The service has never had a trained checkpoint, which is the only
        # reason nobody saw it.
        labels = encode_target(self._processor, self._config.token_for(emotion), prefix=False)
        return features, labels
