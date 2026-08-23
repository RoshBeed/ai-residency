"""Evaluation for the score predictor: mean squared error against a baseline.

The honest comparison is against predicting the training-set mean — with 782
labelled posts the trained model does not clearly beat that baseline, and reporting
the loss without that context would overstate the result.
"""

from typing import Any

import torch
from ai_residency_common import get_logger
from torch import nn
from torch.utils.data import DataLoader

__all__ = ["compute_baseline", "evaluate_scorer"]

logger = get_logger(__name__)


def evaluate_scorer(model: nn.Module, test_loader: DataLoader[Any], device: torch.device) -> float:
    """Return the model's mean squared error over the test loader."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for embeddings, scores in test_loader:
            embeddings, scores = embeddings.to(device), scores.to(device)
            loss = criterion(model(embeddings), scores)
            total_loss += loss.item() * len(embeddings)
            total_examples += len(embeddings)

    return total_loss / total_examples


def compute_baseline(dataset: Any, train_subset: Any, test_subset: Any) -> float:
    """Return the test MSE of predicting the training-set mean for every example."""
    train_scores = dataset.scores[train_subset.indices]
    test_scores = dataset.scores[test_subset.indices]
    baseline_prediction = train_scores.mean()
    baseline_mse = float(((test_scores - baseline_prediction) ** 2).mean().item())
    logger.info(
        "Baseline: predicting the train-mean score (%.2f) for every test example gives MSE %.2f",
        float(baseline_prediction.item()),
        baseline_mse,
    )
    return baseline_mse
