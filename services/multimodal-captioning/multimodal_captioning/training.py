"""Adapter training.

This is the real batched loop that produced `checkpoints/adapter_flickr_batched.pth`,
lifted out of the exploratory notebook (now in
`notebooks/multimodal-captioning/notebook.ipynb`) and put on
the portfolio's canonical shape:

    Config -> set_seed -> build loaders -> compute_baseline -> init_tracking
           -> epoch loop -> log_metrics
           -> on new best: save_checkpoint + push_checkpoint
           -> finish_tracking

It used to live only in that notebook cell, with a second, different, single-example
loop in `final.ipynb`. Two copies of the same pipeline diverging is how this
repo acquired most of its known bugs, so both notebooks now import and call `train`
rather than each carrying their own.

The loss is Qwen3's own next-token cross-entropy over the caption, with the image's soft
token at position 0 masked out of the labels. The baseline is that same loss measured on
the held-out split *before* any training, with the adapter at its random initialization.
That is the number the trained adapter has to beat to have learned anything: because
Qwen3 is frozen and already fluent, a caption loss that merely looks low says nothing on
its own.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ai_residency_common import (
    finish_tracking,
    get_device,
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
from multimodal_captioning.data import (
    Batch,
    Example,
    build_collate_fn,
    build_datasets,
)
from multimodal_captioning.models import (
    Backbones,
    build_adapter,
    image_soft_tokens,
    load_backbones,
)

__all__ = ["caption_loss", "compute_baseline", "evaluate", "main", "train"]

logger = get_logger(__name__)


def caption_loss(
    adapter: nn.Module,
    backbones: Backbones,
    batch: Batch,
    device: torch.device,
) -> torch.Tensor:
    """Return the frozen decoder's loss on one batch of captions.

    The adapter is the only thing gradients flow through: CLIP encodes under
    `no_grad`, and Qwen3's parameters are frozen, so its forward pass still
    backpropagates into the adapter without updating itself.

    Args:
        adapter: The trainable projection.
        backbones: Frozen CLIP and Qwen3.
        batch: Pixel values and tokenized captions, as produced by `build_collate_fn`.
        device: Device to run on.

    Returns:
        Scalar cross-entropy over the caption tokens.
    """
    image_inputs, tokenized = batch
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    batch_size, seq_length = input_ids.shape

    image_latent = image_soft_tokens(adapter, backbones, image_inputs)
    image_tokens = image_latent.shape[1]

    with torch.no_grad():
        text_embeddings = backbones.qwen_model.model.embed_tokens(input_ids)

    full_embeddings = torch.cat([image_latent, text_embeddings], dim=1)

    # -100 across the image positions keeps the soft tokens out of the loss: they have
    # no target tokens, only the caption that follows them does. Widened from a single
    # position when the image became a patch sequence rather than one pooled vector.
    labels = torch.full(
        (batch_size, seq_length + image_tokens), -100, dtype=torch.long, device=device
    )
    labels[:, image_tokens:] = input_ids

    image_mask = torch.ones((batch_size, image_tokens), dtype=torch.long, device=device)
    outputs = backbones.qwen_model(
        inputs_embeds=full_embeddings,
        labels=labels,
        attention_mask=torch.cat([image_mask, attention_mask], dim=1),
    )
    loss: torch.Tensor = outputs.loss
    return loss


def evaluate(
    adapter: nn.Module,
    backbones: Backbones,
    loader: DataLoader[Example],
    device: torch.device,
) -> float:
    """Return the mean caption loss over a loader.

    Args:
        adapter: Projection to score.
        backbones: Frozen CLIP and Qwen3.
        loader: Data to score on.
        device: Device to run on.

    Returns:
        Mean cross-entropy per example.
    """
    adapter.eval()
    total_loss = 0.0
    examples_seen = 0

    with torch.no_grad():
        for batch in loader:
            batch_size = batch[1]["input_ids"].shape[0]
            loss = caption_loss(adapter, backbones, batch, device)
            total_loss += loss.item() * batch_size
            examples_seen += batch_size

    return total_loss / examples_seen


def compute_baseline(
    config: Config,
    backbones: Backbones,
    val_loader: DataLoader[Example],
    device: torch.device,
) -> float:
    """Return the held-out loss of an untrained adapter.

    A randomly initialized projection is close to noise as far as Qwen3 is concerned, so
    this measures what the frozen decoder scores on its own priors alone. Training has to
    beat it to have done anything.

    Args:
        config: Supplies the adapter shape.
        backbones: Frozen CLIP and Qwen3, and the two dimensions.
        val_loader: Held-out split to score on.
        device: Device to run on.

    Returns:
        Mean cross-entropy of the untrained adapter.
    """
    untrained = build_adapter(
        backbones.clip_dim, backbones.qwen_dim, final_layer_norm=config.final_layer_norm
    ).to(device)
    return evaluate(untrained, backbones, val_loader, device)


def train(config: Config) -> float:
    """Train the adapter, tracking metrics and publishing the best checkpoint.

    Args:
        config: Run configuration.

    Returns:
        The best held-out loss reached.
    """
    set_seed(config.seed)
    # See `Config.train_on_cpu`: MPS gives a deterministic forward pass here but an
    # unreliable backward one, so training pins CPU by default while inference does not.
    device = torch.device("cpu") if config.train_on_cpu else get_device()
    logger.info("Training on %s", device)

    backbones = load_backbones(config, device)
    train_dataset, val_dataset = build_datasets(config)
    collate_fn = build_collate_fn(config, backbones)
    train_loader: DataLoader[Example] = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader: DataLoader[Example] = DataLoader(
        val_dataset, batch_size=config.batch_size, collate_fn=collate_fn
    )

    baseline_loss = compute_baseline(config, backbones, val_loader, device)
    logger.info("Baseline (untrained adapter) held-out loss: %.4f", baseline_loss)

    adapter = build_adapter(
        backbones.clip_dim, backbones.qwen_dim, final_layer_norm=config.final_layer_norm
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.learning_rate)

    init_tracking(
        config.wandb_project,
        config={"baseline_loss": baseline_loss, **vars(config)},
    )

    # Resume state, so a run that dies overnight loses at most one epoch rather than
    # all of them. The best-checkpoint save below is for serving and only fires on an
    # improvement; this one fires every epoch and carries the optimizer, so AdamW's
    # moments survive too -- restarting without them is not the same run continued.
    resume_path = resume_path_for(config.checkpoint_path)
    # What makes two runs the same run. Resuming across a different corpus size or a
    # different adapter shape would silently average two experiments together.
    fingerprint = {
        "clip_dim": backbones.clip_dim,
        "qwen_dim": backbones.qwen_dim,
        "final_layer_norm": config.final_layer_norm,
        "num_samples": config.num_samples,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }

    resumed = resume_from(
        resume_path,
        adapter,
        optimizer,
        fingerprint=fingerprint,
        default_best=float("inf"),
        map_location=str(device),
    )
    best_loss, start_epoch = resumed.best_metric, resumed.start_epoch

    if start_epoch > config.num_epochs:
        logger.info("Already at epoch %d of %d; nothing to do.", start_epoch - 1, config.num_epochs)
        finish_tracking()
        return best_loss

    for epoch in range(start_epoch, config.num_epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        examples_seen = 0

        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{config.num_epochs}"):
            batch_size = batch[1]["input_ids"].shape[0]
            loss = caption_loss(adapter, backbones, batch, device)

            optimizer.zero_grad()
            torch.autograd.backward(loss)
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            examples_seen += batch_size

        train_loss = epoch_loss / examples_seen
        val_loss = evaluate(adapter, backbones, val_loader, device)
        log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
        logger.info(
            "Epoch %d/%d  train_loss=%.4f  val_loss=%.4f",
            epoch,
            config.num_epochs,
            train_loss,
            val_loss,
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(
                config.checkpoint_path,
                adapter,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"val_loss": val_loss, "baseline_loss": baseline_loss},
                # The architecture travels with the weights. Not recording it is what
                # broke `adapter_flickr_batched.pth`; see `models.build_adapter`.
                extra={
                    "final_layer_norm": config.final_layer_norm,
                    "clip_dim": backbones.clip_dim,
                    "qwen_dim": backbones.qwen_dim,
                },
            )
            # Only on a new best: these artifacts are large and pushing every
            # improvement would dominate the run.
            if config.hf_repo_id is not None:
                push_checkpoint(config.hf_repo_id, config.checkpoint_path)

        # Every epoch, improvement or not. The best checkpoint above is what gets
        # served; this is what lets the run continue, so it has to record the epoch
        # that finished even when that epoch was worse.
        save_resume(
            resume_path,
            adapter,
            optimizer,
            epoch=epoch,
            best_metric=best_loss,
            fingerprint=fingerprint,
            metrics={"val_loss": val_loss, "baseline_loss": baseline_loss},
        )

    finish_tracking()
    logger.info(
        "Best held-out loss %.4f against baseline %.4f (%s)",
        best_loss,
        baseline_loss,
        "beats baseline" if best_loss < baseline_loss else "does NOT beat baseline",
    )
    return best_loss


def main() -> None:
    """Entry point for `make train`."""
    train(Config())


if __name__ == "__main__":
    main()
