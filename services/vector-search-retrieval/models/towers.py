"""The two encoder towers.

A dual encoder is two networks that map different kinds of input into one shared vector
space, trained together so that a query and its relevant document land near each other.
They are architecturally identical here and still have to be separate objects with
separate weights: a query and a passage are not the same kind of text, and running a
query through the document tower is wrong even when both are trained.

The same `GruTower` class defines both, but it was previously written out three times —
once in the trainer, once in the query engine and once in the indexer. Three copies of
a model definition is how a repo ends up loading weights into an architecture that has
since drifted, so there is one copy now and the two roles are separate instances.

`embedding_layer` is deliberately shared between the two towers rather than duplicated.
The word vectors underneath are frozen, and both sides must read the same ones for a
cosine similarity between their outputs to mean anything.
"""

import torch
from torch import nn

__all__ = ["GruTower", "encode"]


class GruTower(nn.Module):
    """A GRU encoder over frozen word embeddings.

    Attributes:
        embedding: Frozen CBOW word embeddings, shared with the sibling tower.
        rnn: The GRU whose final hidden state is the encoding.
    """

    def __init__(self, embedding_layer: nn.Embedding, hidden_size: int) -> None:
        """Build the tower.

        Args:
            embedding_layer: Pretrained word embeddings. Frozen here rather than
                assumed frozen, so a tower cannot silently start fine-tuning them.
            hidden_size: Width of the GRU hidden state, and so of the output vector.
        """
        super().__init__()
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = False
        self.rnn = nn.GRU(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode one token sequence.

        Args:
            tokens: Token ids of shape `(1, sequence_length)`.

        Returns:
            The final hidden state, shape `(hidden_size,)`.
        """
        embeddings = self.embedding(tokens)
        _, hidden = self.rnn(embeddings)
        output: torch.Tensor = hidden.squeeze(0).squeeze(0)
        return output


def encode(tower: GruTower, token_ids: list[int]) -> torch.Tensor | None:
    """Encode a token id list, returning None for an empty one.

    Tokenizing a short query against a small vocabulary can leave nothing at all, and a
    GRU cannot consume a zero-length sequence. Callers skip those triples rather than
    padding them, which would invent a vector for text that has no known words in it.

    Args:
        tower: Tower to encode with.
        token_ids: Token ids, possibly empty.

    Returns:
        The encoding, or None if `token_ids` was empty.
    """
    if not token_ids:
        return None
    tokens = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    encoding: torch.Tensor = tower(tokens)
    return encoding
