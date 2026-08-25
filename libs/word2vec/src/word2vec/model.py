"""word2vec: CBOW and skip-gram, with a negative-sampling or full-softmax objective.

The two axes are independent, which is the whole reason this is one module rather
than four. Mikolov et al. (2013) present them that way too -- the architecture
decides *which* vectors are combined, the objective decides *how* the score becomes
a loss -- but implementations usually fuse them, so a repo ends up with a
"CBOW with negative sampling" class and a separate "CBOW" class that share nothing.

    architecture   how context and centre relate
      cbow         average the context vectors, predict the centre word
      skipgram     take the centre vector, predict each context word

    objective      how a score becomes a loss
      negative_sampling   score the true target against `num_negatives` sampled
                          words; logsigmoid. O(num_negatives) per example.
      softmax             score against the entire vocabulary; cross-entropy.
                          O(vocab_size) per example, and exact.

Both objectives use the same two matrices, and that is not a coincidence:

    in_embed   (vocab_size, embed_size)   the vectors you keep
    out_embed  (vocab_size, embed_size)   the vectors you discard after training

A full-softmax implementation usually writes the second one as
`nn.Linear(embed_size, vocab_size, bias=False)`, whose weight is also
`(vocab_size, embed_size)`. It is the same matrix under another name, which is what
makes a checkpoint trained by one implementation loadable by the other -- see
`state_dict_from_legacy` in `compat.py`, which exists only until the models are
retrained under this module.

`forward` returns a **loss**, not logits, for either objective. That is the point of
the abstraction: a training loop written against this module does not change when
the architecture or the objective does.
"""

import torch
import torch.nn.functional as F
from torch import nn

from word2vec.config import Word2VecConfig

__all__ = ["Word2Vec"]


class Word2Vec(nn.Module):
    """Word vectors trained by predicting words from their neighbours.

    Attributes:
        config: The architecture, objective and dimensions this was built with.
        in_embed: Input embeddings -- the trained word vectors, and the only part
            of the model anything downstream wants.
        out_embed: Output embeddings, used to score targets during training and
            thrown away afterwards.
    """

    def __init__(self, config: Word2VecConfig) -> None:
        """Build the two embedding matrices.

        Args:
            config: Dimensions, architecture and objective.
        """
        super().__init__()
        self.config = config
        self.in_embed = nn.Embedding(config.vocab_size, config.embed_size)
        self.out_embed = nn.Embedding(config.vocab_size, config.embed_size)

        # Initialised the way the reference implementation does: input vectors
        # uniform in +/- 0.5/embed_size, output vectors at zero. Starting both at
        # random makes the first few thousand updates fight noise in the matrix
        # that is only ever a scoring device.
        #
        # One consequence worth knowing, because it is invisible in a loss curve:
        # gradient reaches in_embed only *through* out_embed, so while out_embed is
        # all zeros the input vectors get exactly zero gradient. The first optimiser
        # step moves only out_embed. A loss that drops on step one therefore says
        # nothing yet about the vectors you keep. `tests/test_word2vec.py` pins this
        # down so it cannot be mistaken for a bug later.
        bound = 0.5 / config.embed_size
        nn.init.uniform_(self.in_embed.weight, -bound, bound)
        nn.init.zeros_(self.out_embed.weight)

    @property
    def embeddings(self) -> torch.Tensor:
        """The trained word vectors, `(vocab_size, embed_size)`.

        `in_embed.weight`, named for what callers want rather than for where it
        sits. Everything downstream -- nearest neighbours, analogies, a frozen
        encoder for another model -- reads this and ignores `out_embed`.
        """
        return self.in_embed.weight

    def _project(self, centre: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Reduce one training pair to the vector that does the predicting.

        Args:
            centre: Centre word ids, `(batch,)`.
            context: Context word ids, `(batch, context_size)`.

        Returns:
            `(batch, embed_size)` for CBOW -- the averaged context. For skip-gram,
            `(batch, embed_size)` too, but the centre word's own vector.
        """
        if self.config.architecture == "cbow":
            pooled: torch.Tensor = self.in_embed(context).mean(dim=1)
            return pooled
        centre_vectors: torch.Tensor = self.in_embed(centre)
        return centre_vectors

    def _targets(self, centre: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """The ids each projected vector is asked to predict.

        CBOW predicts the centre word from its context; skip-gram predicts each
        context word from the centre. This is the only other place the two differ.

        Args:
            centre: Centre word ids, `(batch,)`.
            context: Context word ids, `(batch, context_size)`.

        Returns:
            `(batch, 1)` for CBOW, `(batch, context_size)` for skip-gram.
        """
        if self.config.architecture == "cbow":
            return centre.unsqueeze(1)
        return context

    def forward(
        self,
        centre: torch.Tensor,
        context: torch.Tensor,
        negatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the training loss for a batch of (centre, context) pairs.

        Args:
            centre: Centre word ids, `(batch,)`.
            context: Context word ids, `(batch, context_size)`.
            negatives: Sampled negative ids, `(batch, num_negatives)`. Required for
                the negative-sampling objective, ignored by softmax. Draw them with
                `NegativeSampler`.

        Returns:
            A scalar loss, averaged over the batch.

        Raises:
            ValueError: If the objective needs negatives and none were given.
        """
        projected = self._project(centre, context)
        targets = self._targets(centre, context)

        if self.config.objective == "softmax":
            return self._softmax_loss(projected, targets)

        if negatives is None:
            raise ValueError(
                "the negative_sampling objective needs `negatives`; "
                "draw them with NegativeSampler, or build the model with "
                "objective='softmax'"
            )
        return self._negative_sampling_loss(projected, targets, negatives)

    def _softmax_loss(self, projected: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over the whole vocabulary.

        Exact, and O(vocab_size) per example -- which is what negative sampling
        exists to avoid, and why this is the wrong default above a small vocabulary.

        Args:
            projected: `(batch, embed_size)`.
            targets: `(batch, n_targets)` word ids.

        Returns:
            Scalar loss.
        """
        # out_embed.weight is (vocab, embed), so this is the same matmul a
        # Linear(embed, vocab, bias=False) would do.
        logits = projected @ self.out_embed.weight.T
        n_targets = targets.shape[1]
        if n_targets == 1:
            return F.cross_entropy(logits, targets.squeeze(1))
        # Skip-gram: the same distribution scored against each context word.
        repeated = logits.repeat_interleave(n_targets, dim=0)
        return F.cross_entropy(repeated, targets.reshape(-1))

    def _negative_sampling_loss(
        self, projected: torch.Tensor, targets: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        """Binary logistic loss: true targets up, sampled negatives down.

        Args:
            projected: `(batch, embed_size)`.
            targets: `(batch, n_targets)` word ids.
            negatives: `(batch, num_negatives)` word ids.

        Returns:
            Scalar loss.
        """
        vector = projected.unsqueeze(2)  # (batch, embed, 1)

        positive_scores = torch.bmm(self.out_embed(targets), vector).squeeze(2)
        positive = F.logsigmoid(positive_scores).sum(dim=1)

        # Negated: a negative sample should score *low*, which is logsigmoid(-score).
        negative_scores = torch.bmm(self.out_embed(negatives), vector).squeeze(2)
        negative = F.logsigmoid(-negative_scores).sum(dim=1)

        return -(positive + negative).mean()
