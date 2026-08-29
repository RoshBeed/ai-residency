"""Per-token KL divergence between the policy and the reference model."""

from rlhf_ppo.types import SAPair

__all__ = ["KLDivergence"]


class KLDivergence:
    """Measures how far the policy has drifted from the reference model."""

    @staticmethod
    def compute_kl_per_token(
        logprobs_policy: list[float], logprobs_ref: list[float]
    ) -> list[float]:
        """Return the per-token KL estimate for one sequence.

        This is the single-sample estimator `log pi(a|s) - log pi_ref(a|s)`, taken at
        the token that was actually sampled rather than summed over the whole
        vocabulary. It is unbiased in expectation under the policy, and it costs one
        lookup instead of a full distribution comparison per token.

        Args:
            logprobs_policy: Policy log-probabilities of the sampled tokens.
            logprobs_ref: Reference model's log-probabilities of the same tokens.

        Returns:
            One KL estimate per token.
        """
        return [policy - ref for policy, ref in zip(logprobs_policy, logprobs_ref, strict=True)]

    @staticmethod
    def add_kl_to_sa_pairs(sa_pairs: list[SAPair]) -> list[SAPair]:
        """Add a `kl_div` key to every pair, in place.

        Args:
            sa_pairs: Pairs carrying `logprob` and `ref_logprob`.

        Returns:
            The same list, with `kl_div` set on every pair.
        """
        for pair in sa_pairs:
            pair["kl_div"] = pair["logprob"] - pair["ref_logprob"]
        return sa_pairs
