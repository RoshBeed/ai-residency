"""The resume contract every trainer in this repo depends on.

Each service trains on a machine that can die mid-run -- this one runs out of RAM and
takes the process with it -- so every trainer writes resume state each epoch and
restores it on start. That logic lives in `ai_residency_common` rather than in each
trainer, and it is tested here for the same reason: a resume re-implemented per
service is a resume that goes subtly wrong per service.

The two properties worth holding onto are that resuming continues the *same* run
(weights and optimizer moments, not just weights), and that resuming across two
different runs refuses rather than silently averaging them.
"""

from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW

from ai_residency_common import resume_from, resume_path_for, save_resume

FINGERPRINT = {"corpus": "crema_d", "batch_size": 16}


class Tiny(nn.Module):
    """A model with one typed parameter.

    `nn.Linear.weight` is `Tensor | Module` to a type checker, which makes every
    `allclose` and optimizer-state lookup below an error under this repo's strict
    settings -- and there are no `type: ignore`s here. Declaring the parameter is
    cheaper than working around the stubs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.weight: nn.Parameter = nn.Parameter(torch.ones(2, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


def _trained_one_step() -> tuple[Tiny, AdamW]:
    """A model and optimizer with real AdamW moments populated."""
    torch.manual_seed(0)
    model = Tiny()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    return model, optimizer


def _fresh() -> tuple[Tiny, AdamW]:
    torch.manual_seed(1)
    model = Tiny()
    return model, AdamW(model.parameters(), lr=1e-3)


def test_resume_path_sits_beside_the_served_checkpoint() -> None:
    """Derived, not configured: one name per service rather than two."""
    assert resume_path_for("checkpoints/adapter.pth").name == "adapter_resume.pth"


def test_nothing_to_resume_starts_at_epoch_one(tmp_path: Path) -> None:
    """A first run must not need the file to exist."""
    model, optimizer = _fresh()
    resumed = resume_from(
        tmp_path / "absent_resume.pth",
        model,
        optimizer,
        fingerprint=FINGERPRINT,
        default_best=0.0,
    )
    assert (resumed.start_epoch, resumed.best_metric, resumed.resumed) == (1, 0.0, False)


def test_resume_restores_weights_optimizer_and_best(tmp_path: Path) -> None:
    """The optimizer is the part that is easy to drop and hard to notice.

    Restoring weights alone gives a different trajectory, not the same run
    continued: AdamW's moments are state.
    """
    model, optimizer = _trained_one_step()
    saved_weight = model.weight.detach().clone()
    saved_moment = optimizer.state[model.weight]["exp_avg"].detach().clone()

    path = resume_path_for(tmp_path / "ckpt.pth")
    save_resume(path, model, optimizer, epoch=7, best_metric=0.72, fingerprint=FINGERPRINT)

    model, optimizer = _fresh()
    resumed = resume_from(path, model, optimizer, fingerprint=FINGERPRINT, default_best=0.0)

    assert resumed.resumed
    # Epoch 7 finished, so there is nothing to redo before 8.
    assert resumed.start_epoch == 8
    assert resumed.best_metric == 0.72
    assert torch.allclose(model.weight, saved_weight)
    assert torch.allclose(optimizer.state[model.weight]["exp_avg"], saved_moment)


def test_a_different_run_is_refused_and_leaves_the_model_alone(tmp_path: Path) -> None:
    """Resuming one experiment into another would average two runs silently.

    The refusal has to happen before anything is loaded: a model half overwritten by
    a rejected checkpoint is worse than either outcome.
    """
    model, optimizer = _trained_one_step()
    path = resume_path_for(tmp_path / "ckpt.pth")
    save_resume(path, model, optimizer, epoch=3, best_metric=0.5, fingerprint=FINGERPRINT)

    model, optimizer = _fresh()
    untouched = model.weight.detach().clone()
    resumed = resume_from(
        path,
        model,
        optimizer,
        fingerprint={**FINGERPRINT, "batch_size": 8},
        default_best=0.0,
    )

    assert not resumed.resumed
    assert resumed.start_epoch == 1
    assert torch.allclose(model.weight, untouched)


def test_loss_ranked_trainers_get_their_own_default(tmp_path: Path) -> None:
    """Captioning ranks on loss and speech on accuracy, so the fresh-start value
    cannot be baked in: `inf` and `0.0` are both right, for different callers."""
    model, optimizer = _fresh()
    resumed = resume_from(
        tmp_path / "absent_resume.pth",
        model,
        optimizer,
        fingerprint=FINGERPRINT,
        default_best=float("inf"),
    )
    assert resumed.best_metric == float("inf")
