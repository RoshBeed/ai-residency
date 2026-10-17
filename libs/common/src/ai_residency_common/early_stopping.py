"""Ranking epochs, and knowing when to stop.

Every trainer here had the same two lines -- `if val_loss < best_loss:` or
`if val_accuracy > best_accuracy:` -- and none of them had the third: a count of how
many epochs have passed without an improvement. So runs were sized by guessing a
number of epochs up front, and the guesses were poor in both directions.

Two measured examples from this repo. `speech-emotion-recognition` was run for 20
epochs: held-out accuracy peaked at epoch 10 (0.7225) with epoch 6 effectively tied
(0.7218, a difference of 0.0007), and the last ten epochs bought nothing while
training loss fell to 0.0178. `multimodal-captioning` was run for 12: held-out loss
bottomed at epoch 3 (2.2043) and rose at epoch 4, leaving eight epochs of a turned
curve, about 2.8 hours.

`patience=3` would have ended both at roughly the right place. On the speech run it
would have stopped at epoch 9 and kept 0.7218 instead of 0.7225 -- which is noise --
for half the wall clock.

The mode also removes a smaller duplication: what "no best yet" means. A
loss-ranked run starts at `inf` and an accuracy-ranked one at `0.0`, which every
caller previously spelled out for itself and for `resume_from`.
"""

from typing import Literal

from ai_residency_common.logging import get_logger

__all__ = ["BestMetric", "Mode"]

logger = get_logger(__name__)

Mode = Literal["min", "max"]


class BestMetric:
    """Tracks the best epoch so far, and whether to keep going.

    Replaces a hand-written comparison in each trainer, so that "is this a new
    best?" and "has this stopped improving?" are answered the same way everywhere.

    Example:
        tracker = BestMetric(mode="min", patience=3, best=resumed.best_metric)
        for epoch in ...:
            val_loss = ...
            if tracker.improved(val_loss):
                save_checkpoint(...)
            if tracker.should_stop:
                break
    """

    def __init__(
        self,
        *,
        mode: Mode,
        patience: int | None = 3,
        best: float | None = None,
    ) -> None:
        """
        Args:
            mode: `"min"` when a lower score is better (a loss), `"max"` when higher
                is (an accuracy). There is no default: getting it wrong silently
                saves the worst epoch instead of the best, so it has to be stated.
            patience: Stop after this many consecutive epochs without an
                improvement. `None` never stops early, for a run whose length is
                deliberately fixed.
            best: Seed from a resumed run's best, so a restart does not overwrite a
                better checkpoint an earlier attempt already saved. Defaults to the
                worst possible value for `mode`.
        """
        self.mode: Mode = mode
        self.patience = patience
        self.best: float = self.worst_for(mode) if best is None else best
        self.stale_epochs = 0

    @staticmethod
    def worst_for(mode: Mode) -> float:
        """The starting `best` for a mode: what "nothing seen yet" means."""
        return float("inf") if mode == "min" else 0.0

    @property
    def default_best(self) -> float:
        """This tracker's fresh-start value, to hand to `resume_from`."""
        return self.worst_for(self.mode)

    def improved(self, metric: float) -> bool:
        """Record an epoch's score. Returns whether it is a new best.

        Updates the staleness counter either way, so a caller that only asks this
        still gets `should_stop` for free.
        """
        better = metric < self.best if self.mode == "min" else metric > self.best
        if better:
            self.best = metric
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        return better

    @property
    def should_stop(self) -> bool:
        """Whether `patience` consecutive epochs have passed without a best."""
        if self.patience is None:
            return False
        stop = self.stale_epochs >= self.patience
        if stop:
            logger.info(
                "Stopping early: %d epochs without improving on %.4f",
                self.stale_epochs,
                self.best,
            )
        return stop
