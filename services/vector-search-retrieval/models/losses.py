"""The two training objectives, and why they are not equally informative.

Both ask the towers for the same thing: put a query near the document that answers it
and far from documents that do not. They differ in how many "do not" examples each
query gets to see per step.

**Triplet** scores a query against the one negative sampled into its triple. One
comparison per query per step, and because the negative was drawn uniformly from the
corpus it is almost always obviously irrelevant, so the margin is usually satisfied
already and the gradient is zero. Most of a batch does no work.

**In-batch negatives** treat every *other* document in the batch as a negative for every
query. A batch of 64 gives each query 63 negatives for one matrix multiply, because the
encodings are already computed -- the cost is the dot product between two matrices that
exist either way. This is the standard dual-encoder objective (DPR, SimCSE, CLIP's
contrastive loss) and the reason is throughput of *comparisons*, not of examples.

Measured here on MS MARCO, 5,000-document pool, same towers and same epochs:
`libs/common` has no opinion on which to use, but Section 6 of this service's README
carries the numbers.
"""

import torch
import torch.nn.functional as F

__all__ = ["in_batch_loss", "triplet_batch_loss"]


def in_batch_loss(
    queries: torch.Tensor, documents: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Cross-entropy over every query-document pairing in the batch.

    Row `i` of the similarity matrix holds query `i` scored against all documents, and
    the correct answer is always document `i`, so the targets are just `arange`. That
    diagonal structure is the whole trick: no negative has to be sampled or stored,
    because the other rows of the batch already are the negatives.

    Args:
        queries: Query encodings, `(batch, hidden)`.
        documents: Encodings of the matching documents, `(batch, hidden)`, aligned so
            that `documents[i]` is the positive for `queries[i]`.
        temperature: Divides the similarities before the softmax. Cosine similarity is
            bounded to [-1, 1]; at temperature 1 the softmax over such a narrow range is
            nearly uniform and the loss barely distinguishes a good ranking from a bad
            one.

    Returns:
        Mean cross-entropy over the batch.
    """
    queries = F.normalize(queries, dim=1)
    documents = F.normalize(documents, dim=1)
    similarities = queries @ documents.T / temperature
    targets = torch.arange(len(queries), device=similarities.device)
    loss: torch.Tensor = F.cross_entropy(similarities, targets)
    return loss


def triplet_batch_loss(
    queries: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Margin loss against each triple's own sampled negative.

    The batched form of what this service trained with before. Batching changes the
    arithmetic only by averaging, so it stays the honest comparison for the in-batch
    objective: same towers, same pairs, same optimizer, one negative instead of many.

    Args:
        queries: Query encodings, `(batch, hidden)`.
        positives: Encodings of the relevant documents, `(batch, hidden)`.
        negatives: Encodings of the sampled irrelevant documents, `(batch, hidden)`.
        margin: How much further the negative must be before the loss is zero.

    Returns:
        Mean hinge loss over the batch.
    """
    similarity_positive = F.cosine_similarity(queries, positives, dim=1)
    similarity_negative = F.cosine_similarity(queries, negatives, dim=1)
    loss: torch.Tensor = torch.clamp(
        margin - (similarity_positive - similarity_negative), min=0.0
    ).mean()
    return loss
