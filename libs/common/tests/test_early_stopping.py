"""Ranking epochs in both directions, and stopping when a curve has turned.

The cases below are the two real curves this helper was written from, so a change
that breaks the behaviour they justify fails here rather than in a training run
nobody is watching.
"""

import pytest

from ai_residency_common import BestMetric, Mode

# speech-emotion-recognition, 20 epochs of whisper-tiny on CREMA-D. Accuracy is
# noisy: epoch 6 and epoch 10 are tied to within 0.0007, and everything between is
# worse.
SPEECH_ACCURACY = [
    0.2071,
    0.5013,
    0.6052,
    0.6535,
    0.7058,
    0.7218,
    0.6977,
    0.6944,
    0.7011,
    0.7225,
    0.7151,
    0.7071,
]
# multimodal-captioning, held-out loss. Bottoms at epoch 3 and turns.
CAPTIONING_LOSS = [2.3097, 2.2384, 2.2043, 2.2772]


def _run(metrics: list[float], *, mode: Mode, patience: int | None) -> tuple[int, float]:
    """Feed a curve in and report the epoch it stopped on and the best seen."""
    tracker = BestMetric(mode=mode, patience=patience)
    for epoch, metric in enumerate(metrics, start=1):
        tracker.improved(metric)
        if tracker.should_stop:
            return epoch, tracker.best
    return len(metrics), tracker.best


def test_a_fresh_tracker_starts_at_the_worst_value_for_its_mode() -> None:
    """What "nothing seen yet" means, so callers stop spelling it out."""
    assert BestMetric(mode="min").best == float("inf")
    assert BestMetric(mode="max").best == 0.0
    assert BestMetric(mode="min").default_best == float("inf")


@pytest.mark.parametrize(
    ("mode", "metrics", "expected_best"),
    [("max", SPEECH_ACCURACY, 0.7225), ("min", CAPTIONING_LOSS, 2.2043)],
)
def test_without_patience_it_ranks_but_never_stops(
    mode: Mode, metrics: list[float], expected_best: float
) -> None:
    """A deliberately fixed-length run still wants the ranking."""
    stopped_at, best = _run(metrics, mode=mode, patience=None)
    assert stopped_at == len(metrics)
    assert best == expected_best


def test_patience_ends_the_speech_run_nine_epochs_early_for_noise() -> None:
    """The case that justifies the default.

    Accuracy peaked at 0.7225 on epoch 10. Stopping on three stale epochs ends it
    after epoch 9 with 0.7218 -- a loss of 0.0007, which is noise -- and skips the
    eleven epochs that followed.
    """
    stopped_at, best = _run(SPEECH_ACCURACY, mode="max", patience=3)
    assert stopped_at == 9
    assert best == pytest.approx(0.7218)
    assert abs(best - max(SPEECH_ACCURACY)) < 0.001


def test_a_turned_loss_curve_is_not_stopped_on_one_bad_epoch() -> None:
    """Patience is there because one worse epoch means little.

    Captioning's loss rose at epoch 4 after bottoming at 3; a patience of 3 keeps
    going, which is the point -- epoch 10 of the speech run was a new best after
    three worse ones.
    """
    stopped_at, best = _run(CAPTIONING_LOSS, mode="min", patience=3)
    assert stopped_at == len(CAPTIONING_LOSS)
    assert best == 2.2043


def test_staleness_resets_on_an_improvement() -> None:
    tracker = BestMetric(mode="min", patience=2)
    for metric in (5.0, 6.0):
        tracker.improved(metric)
    assert tracker.stale_epochs == 1
    assert tracker.improved(4.0)
    assert tracker.stale_epochs == 0
    assert not tracker.should_stop


def test_a_resumed_best_is_not_overwritten_by_a_worse_epoch() -> None:
    """A restart must not save over a better checkpoint an earlier attempt wrote."""
    tracker = BestMetric(mode="max", patience=3, best=0.7225)
    assert not tracker.improved(0.7011)
    assert tracker.best == 0.7225
