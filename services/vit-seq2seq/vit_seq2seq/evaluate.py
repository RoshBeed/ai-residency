"""Score a saved seq2seq checkpoint the way the README quotes it.

This exists because the headline number came out of a training run's validation pass,
which meant the README could not be reproduced without retraining, and because the
metric behind it was measuring something other than what it claimed.

Two things it reports that the old pass did not:

**Accuracy per digit count.** With the labels now the number's own digits rather than a
zero-padded field, a sequence can be one, two or three digits long, and a single
aggregate hides whether the model is simply good at the length it sees most.

**Whether EOS lands in the right place.** The decoder has an EOS token and a sequence
can now end anywhere, so stopping correctly is a decision it can get wrong
independently of the digits. Reported separately, because a model that reads every digit
correctly and then keeps going is wrong in a different way from one that misreads a
digit.

**Per-digit accuracy, aligned by position.** Exact-sequence is the number worth quoting,
but it conflates reading and stopping: one misread digit in three scores zero for the
sequence, the same as emitting four digits. The per-digit figure is what separates
perception from the length decision, and it belongs here rather than in a scratch script
so the README's numbers all come out of one command.
"""

import argparse
import json
from collections import Counter, defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_residency_common import get_device, get_logger, resolve_required, set_seed
from config import Config
from datasets.multi_digit_dataset import Example, MultiDigitMNISTDataset
from vit_seq2seq.seq2seq import Seq2SeqVisionTransformer

__all__ = ["evaluate_checkpoint", "main", "score"]

logger = get_logger(__name__)


def digits_of(sequence: list[int], start_token: int, eos_token: int) -> list[int]:
    """Return the digit tokens a sequence spells, stopping at the first EOS.

    Args:
        sequence: Token ids for one example.
        start_token: Token the sequence was primed with.
        eos_token: Token that ends it.

    Returns:
        The digits, as a list. Compared element-wise rather than folded into an integer,
        which is what makes "exact sequence" mean exact: reading the tokens back as a
        number cannot tell a missing zero from a correct answer.
    """
    digits: list[int] = []
    for token in sequence:
        if token == eos_token:
            break
        if token == start_token:
            continue
        digits.append(int(token))
    return digits


def score(
    model: Seq2SeqVisionTransformer,
    loader: DataLoader[Example],
    device: torch.device,
    config: Config,
    description: str = "Scoring",
    sample: bool = False,
) -> dict[str, float]:
    """Score a model on a loader, overall and per digit count.

    Args:
        model: Model to score.
        loader: Data to score on.
        device: Device to run on.
        config: Supplies the token ids and the decode length.
        description: Progress-bar label.
        sample: Decode by drawing from the softmax instead of taking the argmax. Off by
            default, matching how the model is served. It is an option here because
            what the decode rule is worth is a number this README quotes, and a figure
            measured by a scratch script is one nobody can reproduce -- the two rows of
            that comparison have to come out of the same command on the same split.

    Returns:
        Exact-sequence accuracy overall and per length, EOS-position accuracy, per-digit
        accuracy overall and per length, and the counts each is over.
    """
    arch = config.seq2seq_arch
    model.eval()

    exact = Counter[int]()
    seen = Counter[int]()
    digit_hits = Counter[int]()
    digits_seen = Counter[int]()
    stopped_correctly = 0
    total = 0
    predicted_lengths: dict[int, Counter[int]] = defaultdict(Counter)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc=description, leave=False):
            images = images.to(device)
            generated = model.generate(
                images,
                start_token=arch.start_token,
                max_length=config.decoder_seq_length,
                sample=sample,
            )
            for predicted_row, target_row in zip(generated, targets, strict=True):
                want = digits_of(target_row.tolist(), arch.start_token, arch.eos_token)
                got = digits_of(predicted_row.tolist(), arch.start_token, arch.eos_token)
                length = len(want)
                seen[length] += 1
                exact[length] += int(got == want)
                stopped_correctly += int(len(got) == length)
                predicted_lengths[length][len(got)] += 1
                total += 1
                # Aligned from the left: how many of the target's digits the prediction
                # put in the right place, whatever it did about length.
                for position, digit in enumerate(want):
                    digits_seen[length] += 1
                    digit_hits[length] += int(position < len(got) and got[position] == digit)

    results: dict[str, float] = {
        "exact_sequence_accuracy": 100.0 * sum(exact.values()) / max(total, 1),
        "eos_position_accuracy": 100.0 * stopped_correctly / max(total, 1),
        "per_digit_accuracy": 100.0 * sum(digit_hits.values()) / max(sum(digits_seen.values()), 1),
        "examples": float(total),
    }
    for length in sorted(seen):
        results[f"exact_{length}_digit"] = 100.0 * exact[length] / seen[length]
        results[f"examples_{length}_digit"] = float(seen[length])
        results[f"per_digit_{length}_digit"] = 100.0 * digit_hits[length] / digits_seen[length]

    for length in sorted(predicted_lengths):
        spread = ", ".join(
            f"{predicted}->{count}"
            for predicted, count in sorted(predicted_lengths[length].items())
        )
        logger.info("%d-digit targets, predicted lengths: %s", length, spread)

    return results


def evaluate_checkpoint(config: Config, sample: bool = False) -> dict[str, float]:
    """Load the saved checkpoint and score it on the test split.

    Args:
        config: Run configuration, including which dataset shape to build.
        sample: Decode by sampling rather than greedily. See `score`.

    Returns:
        The metrics `score` produces.
    """
    set_seed(config.seed)
    device = get_device()
    arch = config.seq2seq_arch

    dataset = MultiDigitMNISTDataset(
        data_dir=config.data_dir,
        image_size=arch.img_size,
        max_digits=config.max_digits,
        min_digits=config.min_digits,
        digit_size=config.digit_size,
        augment_probability=config.augment_probability,
        rotation_degrees=config.augment_rotation_degrees,
        shear_degrees=config.augment_shear_degrees,
        scale_range=config.augment_scale_range,
        train_split=config.train_split,
        samples_per_class=config.samples_per_class,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        return_digit_sequence=True,
        add_start_eos=True,
        start_token=arch.start_token,
        eos_token=arch.eos_token,
    )

    model = Seq2SeqVisionTransformer(arch).to(device)
    path, source = resolve_required(
        config.seq2seq_checkpoint_path,
        config.hf_repo_id,
        rebuild_hint="Train it first with `make train`.",
    )
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state))
    logger.info("Scoring %s from %s on %s", config.seq2seq_checkpoint_path.name, source, device)
    logger.info(
        "Digit counts %d-%d, augmentation at %.0f%% (rotation +/-%.0f, shear +/-%.0f, scale %s)",
        config.min_digits,
        config.max_digits,
        100 * config.augment_probability,
        config.augment_rotation_degrees,
        config.augment_shear_degrees,
        config.augment_scale_range,
    )

    logger.info("Decoding %s", "by sampling at temperature 1" if sample else "greedily")
    results = score(
        model,
        dataset.test_loader,
        device,
        config,
        description="Scoring test split",
        sample=sample,
    )

    logger.info(
        "Exact-sequence accuracy %.2f%% over %d examples; EOS lands correctly %.2f%% of the time; "
        "per-digit accuracy %.2f%%",
        results["exact_sequence_accuracy"],
        int(results["examples"]),
        results["eos_position_accuracy"],
        results["per_digit_accuracy"],
    )
    for length in range(config.min_digits, config.max_digits + 1):
        key = f"exact_{length}_digit"
        if key in results:
            logger.info(
                "  %d digit: %.2f%% exact, %.2f%% per digit, over %d examples",
                length,
                results[key],
                results[f"per_digit_{length}_digit"],
                int(results[f"examples_{length}_digit"]),
            )
    return results


def main() -> None:
    """Entry point for `make evaluate`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="decode by sampling instead of greedily, to measure what the decode rule is worth",
    )
    args = parser.parse_args()
    results = evaluate_checkpoint(Config(), sample=args.sample)
    if args.json:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
