"""Fine-tune Whisper to emit an emotion token for a clip.

The shape every training stage in this portfolio follows:

    Config -> set_seed -> baseline -> init_tracking
           -> epoch loop -> log_metrics
           -> on new best: save_checkpoint + push_checkpoint
           -> finish_tracking

The baseline is doing real work here rather than being ceremony. Six emotions are not
evenly represented in either corpus, so "always answer whichever emotion is most common
in the training split" is a genuinely competitive score on a small dataset -- and a
model that fails to beat it has learned nothing about the audio, however good its loss
curve looks.
"""

import collections

import torch
from torch.utils.data import DataLoader

from ai_residency_common import (
    finish_tracking,
    get_logger,
    init_tracking,
    log_metrics,
    push_checkpoint,
    resume_from,
    resume_path_for,
    save_checkpoint,
    save_resume,
    set_seed,
)
from config import Config
from speech import IGNORE_INDEX, collate
from speech_emotion_recognition.data import EmotionClip, EmotionDataset
from speech_emotion_recognition.models import load_model_and_processor

__all__ = ["compute_baseline", "evaluate", "main", "train"]

logger = get_logger(__name__)


def compute_baseline(dataset: EmotionDataset) -> tuple[str, float]:
    """Score always answering the most common emotion in the split.

    Args:
        dataset: The split to score against.

    Returns:
        The majority emotion and the accuracy of always answering it.
    """
    counts = collections.Counter(emotion for _, emotion in dataset.rows)
    emotion, count = counts.most_common(1)[0]
    return emotion, count / len(dataset.rows)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[EmotionClip],
    device: torch.device,
) -> tuple[float, float]:
    """Mean loss and exact-token accuracy over a split.

    Accuracy is measured on the first generated token rather than by decoding text: the
    label is a single emotion token, so the first thing the decoder emits after the
    forced prefix is the whole prediction.

    Args:
        model: The model to evaluate.
        loader: Batches from the validation split.
        device: Device to run on.

    Returns:
        `(mean_loss, accuracy)`.
    """
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            features, labels = features.to(device), labels.to(device)
            output = model(input_features=features, labels=labels)
            total_loss += float(output.loss.item())

            predicted = output.logits.argmax(dim=-1)
            target_positions = labels != IGNORE_INDEX
            correct += int(((predicted == labels) & target_positions).all(dim=-1).sum().item())
            seen += labels.shape[0]

    return total_loss / max(len(loader), 1), correct / max(seen, 1)


def train(config: Config) -> float:
    """Fine-tune the model and return the best validation accuracy.

    Args:
        config: Run configuration.

    Returns:
        The best validation accuracy reached.
    """
    set_seed(config.seed)

    model, processor, device = load_model_and_processor(config)
    train_dataset = EmotionDataset(config, processor, "train")
    val_dataset = EmotionDataset(config, processor, "val")

    train_loader: DataLoader[EmotionClip] = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader: DataLoader[EmotionClip] = DataLoader(
        val_dataset, batch_size=config.batch_size, collate_fn=collate
    )

    majority_emotion, baseline_accuracy = compute_baseline(train_dataset)
    logger.info(
        "Baseline: always answering %r gets %.4f on the training split",
        majority_emotion,
        baseline_accuracy,
    )

    init_tracking(
        config.wandb_project,
        config={
            "model_name": config.model_name,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "train_clips": len(train_dataset),
            "val_clips": len(val_dataset),
            "baseline_accuracy": baseline_accuracy,
        },
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Resume state, shared with every other trainer via `ai_residency_common`. This
    # machine dies when it runs out of RAM, and a full fine-tune here is long enough
    # that losing a run to that matters.
    checkpoint_path = config.checkpoint_dir / config.checkpoint_name
    resume_path = resume_path_for(checkpoint_path)
    fingerprint = {
        "model_name": config.model_name,
        "dataset_keys": list(config.dataset_keys),
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }
    # `default_best=0.0`: this ranks on accuracy, where higher is better, unlike the
    # loss-ranked trainers that pass `inf`.
    resumed = resume_from(
        resume_path,
        model,
        optimizer,
        fingerprint=fingerprint,
        default_best=0.0,
        map_location=str(device),
    )
    best_accuracy, start_epoch = resumed.best_metric, resumed.start_epoch
    if start_epoch > config.num_epochs:
        logger.info("Already at epoch %d of %d; nothing to do.", start_epoch - 1, config.num_epochs)
        finish_tracking()
        return best_accuracy

    for epoch in range(start_epoch, config.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            loss = model(input_features=features, labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += float(loss.item())

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        logger.info(
            "Epoch %d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f (baseline %.4f)",
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
            baseline_accuracy,
        )
        log_metrics(
            {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy},
            step=epoch,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                checkpoint_path,
                model,
                epoch=epoch,
                metrics={"val_accuracy": val_accuracy, "val_loss": val_loss},
                extra={
                    "model_name": config.model_name,
                    "emotions": list(config.emotions),
                    "baseline_accuracy": baseline_accuracy,
                },
            )
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, checkpoint_path)

        # Every epoch, improvement or not, so a crash costs one epoch rather than
        # restarting from the last epoch that happened to be an improvement.
        save_resume(
            resume_path,
            model,
            optimizer,
            epoch=epoch,
            best_metric=best_accuracy,
            fingerprint=fingerprint,
            metrics={"val_accuracy": val_accuracy, "val_loss": val_loss},
        )

    finish_tracking()
    return best_accuracy


def main() -> None:
    """Entry point for `make train`."""
    train(Config())


if __name__ == "__main__":
    main()
