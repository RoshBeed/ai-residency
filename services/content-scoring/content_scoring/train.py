"""Training one fusion architecture, and streaming its evaluation to W&B as it goes.

The loop computes no metrics of its own. Everything charted here comes out of
`evaluate.py`, which is also what `make evaluate` prints, so a number on a W&B chart and
a number in the README cannot disagree.

What is logged, and at what cadence:

| when | what |
|---|---|
| every `eval_every` steps | `train/loss`, `train/step` |
| step 0 | the untrained model's full block, and all four baselines |
| every epoch | the val block, and the baselines re-logged unchanged |
| every epoch | the slice table and the probe titles |
| end of run | the test block, the ablation, and the intrinsic embedding scores |

The baselines are re-logged every epoch on purpose. They are constants, so a single
step-0 point would sit alone at the left edge of a chart nobody scrolls to; repeated,
they draw as flat reference lines across the same axes as the curve they bound.

The expensive parts are computed once. TF-IDF + Ridge takes about twenty seconds to fit
on 710k titles, so it is fitted at step 0 and its predictions cached for the rest of the
run; a sweep run should be dominated by training, not by its own evaluation.
"""

import argparse
import json
import math
import time
from dataclasses import asdict

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from ai_residency_common import (
    define_metrics,
    finish_tracking,
    get_device,
    get_logger,
    init_tracking,
    load_checkpoint,
    log_metrics,
    log_table,
    push_checkpoint,
    save_checkpoint,
    set_seed,
    set_summary,
)
from config import Config
from content_scoring.data import build_splits, iterate_batches
from content_scoring.evaluate import (
    baseline_predictions,
    evaluate_predictions,
    leave_one_group_out,
    predict,
    probe_table,
    score_extended_embeddings,
    slice_table,
    viral_calibration,
)
from content_scoring.model import build_model, load_predictor_weights, predictor_state_dict

__all__ = ["train"]

logger = get_logger(__name__)

_HEADLINE_SLICES = (
    ("title_oov", "high"),
    ("domain", "rare"),
    ("prefix", "show_hn"),
    ("score_quartile", "q4"),
)
"""Slices promoted to scalars so they chart. The table carries all of them; these four
are the ones worth watching move."""


def train(config: Config) -> float:
    """Train one architecture and return its best validation Spearman.

    Args:
        config: Supplies the architecture, the schedule and where things are written.

    Returns:
        The best `val_spearman` seen, which is what the sweep ranks on.

    Raises:
        FileNotFoundError: If the ETL or the vocabulary extension has not run.
    """
    set_seed(config.seed)
    device = get_device()

    if not config.embeddings_path.exists():
        raise FileNotFoundError(
            f"{config.embeddings_path} is missing. Run `make -C services/content-scoring "
            "extend-vocab` first."
        )
    payload = load_checkpoint(config.embeddings_path)
    matrix = payload["model_state_dict"]["in_embed.weight"]
    word_to_index = payload["extra"]["word_to_index"]

    splits = build_splits(config, word_to_index)
    model = build_model(matrix, splits.spec.num_domains, config).to(device)
    optimizer = optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    # Poisson NLL with a log link: the head emits log(lambda) and is scored against the
    # raw count, so `exp` of it is the expected upvote count with no back-transform to be
    # biased. `log_mse` is the older path -- it ranks just as well and predicts about a
    # fifth of the upvotes, which is the comparison the README makes.
    criterion: nn.Module = (
        nn.PoissonNLLLoss(log_input=True, full=False)
        if config.objective == "poisson"
        else nn.MSELoss()
    )

    val_frame = splits.frames["val"]
    # Upvotes, not log1p(upvotes). Every metric is on the scale the task asks about.
    val_actual = val_frame["score"].to_numpy(dtype=np.float64)

    init_tracking(
        config.wandb_project,
        config={**asdict(config), "fusion": config.fusion},
        run_name=f"{config.fusion}-seed{config.seed}",
    )
    define_metrics({"train/*": "train/step", "eval/*": "epoch", "baseline/*": "epoch"})

    # Fitted once. Re-fitting per epoch would more than double a run and change nothing:
    # a baseline that moved between epochs would not be a baseline.
    logger.info("Fitting baselines on the training split")
    baselines = {
        name: evaluate_predictions(val_frame, predicted, val_actual)
        for name, predicted in baseline_predictions(splits, "val").items()
    }
    baseline_scalars = {
        f"baseline/{name}_{metric}": value
        for name, metrics in baselines.items()
        for metric, value in metrics.items()
        if metric in ("spearman", "ndcg@10", "rmse")
    }
    for name, metrics in baselines.items():
        logger.info(
            "baseline %-14s spearman %.4f  ndcg@10 %.4f  rmse %.4f",
            name,
            metrics["spearman"],
            metrics["ndcg@10"],
            metrics["rmse"],
        )

    # Step 0: what an untrained model of this architecture scores. Without it there is
    # no way to tell a real result from a plumbing bug that happens to return numbers.
    untrained = evaluate_predictions(
        val_frame, predict(model, splits.tensors["val"], device, config.objective), val_actual
    )
    logger.info(
        "untrained       spearman %.4f  ndcg@10 %.4f", untrained["spearman"], untrained["ndcg@10"]
    )
    log_metrics(
        {
            **{f"eval/{name}": value for name, value in untrained.items()},
            **baseline_scalars,
        },
        step=0,
    )

    best_score = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    step = 0
    started = time.monotonic()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        batches = iterate_batches(
            splits.tensors["train"], config.batch_size, shuffle=True, device=device
        )
        running = 0.0
        for tokens, mask, numeric, domain, log_target in tqdm(
            batches, desc=f"{config.fusion} epoch {epoch}/{config.num_epochs}", leave=False
        ):
            predicted = model(tokens, mask, numeric, domain)
            # The batcher carries log1p(score); Poisson wants the count itself.
            target = torch.expm1(log_target) if config.objective == "poisson" else log_target
            loss = criterion(predicted, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
            step += 1
            if step % config.eval_every == 0:
                log_metrics({"train/loss": float(loss.detach()), "train/step": step}, step=step)

        predicted_val = predict(model, splits.tensors["val"], device, config.objective)
        metrics = evaluate_predictions(val_frame, predicted_val, val_actual)
        columns, rows = slice_table(val_frame, predicted_val, val_actual, splits, word_to_index)
        headline = {
            f"eval/slice/{family}_{bucket}": float(row[3])
            for family, bucket in _HEADLINE_SLICES
            for row in rows
            if row[0] == family and row[1] == bucket
        }

        log_metrics(
            {
                **{f"eval/{name}": value for name, value in metrics.items()},
                **headline,
                **baseline_scalars,
                "epoch": epoch,
                "train/epoch_loss": running / max(len(batches), 1),
            },
            step=epoch,
        )
        log_table("eval/slices", columns, rows, step=epoch)
        probe_columns, probe_rows = probe_table(model, splits, config, word_to_index)
        log_table("eval/probes", probe_columns, probe_rows, step=epoch)

        logger.info(
            "epoch %2d/%d  loss %.4f  val spearman %.4f  ndcg@10 %.4f  lift %.2f",
            epoch,
            config.num_epochs,
            running / max(len(batches), 1),
            metrics["spearman"],
            metrics["ndcg@10"],
            metrics["top_decile_lift"],
        )

        if metrics["spearman"] > best_score:
            best_score = metrics["spearman"]
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in predictor_state_dict(model).items()
            }
        elif epoch - best_epoch >= config.patience:
            logger.info("No improvement in %d epochs; stopping at %d", config.patience, epoch)
            break

    if best_state:
        load_predictor_weights(model, best_state)
    else:
        # Every epoch's Spearman was NaN, which means the model predicted one constant
        # value for all 92,557 validation posts -- a collapse, not a bad fit. It happens
        # in the high-learning-rate corner of the sweep space and is worth naming rather
        # than leaving as an unexplained `-inf` in the sweep table.
        #
        # Reported as -1.0, the true floor of a rank correlation: finite, so W&B and the
        # sweep table handle it, and still ranked last, which is where a collapsed run
        # belongs.
        best_score = -1.0
        logger.warning(
            "%s collapsed to a constant prediction at lr=%g, dropout=%g, layers=%d -- "
            "no epoch produced a defined Spearman",
            config.fusion,
            config.learning_rate,
            config.dropout,
            config.num_layers,
        )

    test_frame = splits.frames["test"]
    test_actual = test_frame["score"].to_numpy(dtype=np.float64)
    test_predicted = predict(model, splits.tensors["test"], device, config.objective)
    test_metrics = evaluate_predictions(test_frame, test_predicted, test_actual)
    embed_metrics = score_extended_embeddings(config)

    test_columns, test_rows = slice_table(
        test_frame, test_predicted, test_actual, splits, word_to_index
    )
    log_table("test/slices", test_columns, test_rows)
    ablation_columns, ablation_rows = leave_one_group_out(model, splits, "test")
    if ablation_rows:
        log_table("test/ablation", ablation_columns, ablation_rows)

    # `set_summary`, not the last logged value: W&B otherwise summarises a run by where
    # it finished rather than by how good it got, and the sweep table sorts on this.
    set_summary(
        {
            "best/val_spearman": best_score,
            "best/epoch": best_epoch,
            # Recorded so the capacity comparison can be plotted against what the
            # models actually cost rather than against a layer count. At equal
            # width and depth late fusion has five branches and a head where early
            # has one MLP, so "same hyperparameters" is not "same capacity" -- and
            # a curve drawn against layers would quietly credit early fusion for
            # being the smaller model.
            "model/parameters": sum(p.numel() for p in model.parameters()),
            "model/trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            **{f"test/{name}": value for name, value in test_metrics.items()},
            **{f"embed/{name}": value for name, value in embed_metrics.items()},
            "runtime_minutes": (time.monotonic() - started) / 60,
        }
    )
    logger.info(
        "best epoch %d: val spearman %.4f | test spearman %.4f, ndcg@10 %.4f, lift %.2f",
        best_epoch,
        best_score,
        test_metrics["spearman"],
        test_metrics["ndcg@10"],
        test_metrics["top_decile_lift"],
    )
    # NaN-filtered, because predict-the-mean has no ranking at all and therefore no
    # Spearman. Left in, `max` picks whichever key it happens to visit first and the
    # comparison against NaN is always False -- so the run reports "no" against a
    # baseline it beat comfortably, which is the sort of line you believe.
    ranked = {
        name: metrics["spearman"]
        for name, metrics in baselines.items()
        if not math.isnan(metrics["spearman"])
    }
    strongest = max(ranked, key=lambda name: ranked[name])
    logger.info(
        "beat the strongest baseline, %s at %.4f? %s",
        strongest,
        ranked[strongest],
        "yes" if best_score > ranked[strongest] else "no",
    )

    if config.save_checkpoints:
        if not config.freeze_embeddings:
            logger.warning(
                "This run fine-tuned the word vectors, and the checkpoint does not carry "
                "them. It is recorded as such and serving will refuse it. Publish a frozen "
                "run, or extend the checkpoint format to save the tuned matrix."
            )
        config.predictor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            config.predictor_checkpoint,
            model,
            epoch=best_epoch,
            metrics={"val_spearman": best_score, **test_metrics},
            extra={
                "fusion": config.fusion,
                "feature_spec": asdict(splits.spec),
                "num_domains": splits.spec.num_domains,
                # Recorded so the loader can refuse it. The checkpoint deliberately
                # omits the embedding table, which is correct while the table is frozen
                # and shared -- and quietly wrong when a run fine-tuned it, because
                # serving would then pair this head with the *base* vectors and answer
                # 200 with numbers that have nothing to do with what was evaluated.
                "embeddings_finetuned": not config.freeze_embeddings,
                "objective": config.objective,
                # Fitted on validation, which the model did not train on, so serving can
                # report a real chance of reaching the front page rather than the
                # Poisson's analytic tail -- which is off by twenty orders of magnitude
                # on a distribution this overdispersed.
                "viral_calibration": viral_calibration(predicted_val, val_actual),
                "config": asdict(config),
            },
            state_dict=predictor_state_dict(model),
        )
        config.info_path.write_text(
            json.dumps(
                {"fusion": config.fusion, "epoch": best_epoch, **test_metrics},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        if config.hf_repo_id is not None:
            push_checkpoint(config.hf_repo_id, config.predictor_checkpoint, private=False)

    finish_tracking()
    return best_score


def main() -> None:
    """Train from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion",
        default=None,
        choices=("text_only", "features_only", "early", "late"),
        help="Override the configured architecture.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override the configured seed.")
    arguments = parser.parse_args()

    overrides = {}
    if arguments.fusion:
        overrides["fusion"] = arguments.fusion
    if arguments.seed is not None:
        overrides["seed"] = arguments.seed
    train(Config(**overrides))


if __name__ == "__main__":
    main()
