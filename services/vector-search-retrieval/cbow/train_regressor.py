"""Legacy: fit a regressor on top of frozen CBOW embeddings.

Was `02_train_reg.py`, carried over from Week 1's Hacker News score prediction. Nothing
in the retrieval pipeline calls it and the README already marks it as legacy, but it is
documented, so it is kept and made to run rather than quietly deleted.

It was broken on two counts before this pass, both of which only showed up once the
file became importable enough to type-check. It loaded a hardcoded checkpoint file name
(`2025_04_17__11_04_09.5.cbow.pth`) that does not exist in this repo, and it built the
model with a hardcoded vocabulary size of 63,642 when the vocabulary here has 7,692
entries, so `load_state_dict` could not have succeeded either. Both now come from the
same config and loader every other stage uses.

What it demonstrates is unchanged and deliberately trivial: one fabricated title and
one fabricated score, fitted a hundred times. It shows the mechanism, not a result.
"""

import torch
import torch.nn.functional as F
from ai_residency_common import get_logger, set_seed

from cbow.model import Regressor
from cbow.vocabulary import build_embedding_layer
from config import Config

__all__ = ["main", "train_regressor"]

logger = get_logger(__name__)


def train_regressor(config: Config) -> float:
    """Fit the regressor against a single synthetic example.

    Args:
        config: Run configuration.

    Returns:
        The final L1 loss.
    """
    set_seed(config.seed)
    embedding_layer = build_embedding_layer(config)

    model = Regressor(config.embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.regressor_learning_rate)

    target = torch.tensor([[125.0]])
    title_tokens = torch.tensor([[45, 27, 4536, 456]])

    loss_value = 0.0
    for step in range(config.regressor_steps):
        with torch.no_grad():
            pooled = embedding_layer(title_tokens).mean(dim=1)
        loss = F.l1_loss(model(pooled), target)
        optimizer.zero_grad()
        torch.autograd.backward(loss)
        optimizer.step()
        loss_value = loss.item()
        if step % 10 == 0:
            logger.info("Step %d  loss=%.4f", step, loss_value)

    logger.info("Final loss %.4f", loss_value)
    return loss_value


def main() -> None:
    """Entry point for the legacy regressor demo."""
    train_regressor(Config())


if __name__ == "__main__":
    main()
