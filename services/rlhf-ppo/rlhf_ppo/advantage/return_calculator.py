"""Reward-to-go, the regression target the value model is trained on."""

from rlhf_ppo.types import SAPair

__all__ = ["ReturnCalculator"]


class ReturnCalculator:
    """Recovers returns from advantages."""

    @staticmethod
    def add_reward_to_go(sa_pairs: list[SAPair]) -> list[SAPair]:
        """Add a `reward_to_go` key to every pair, in place.

        The return is recovered from the advantage rather than accumulated
        separately: since `A(s, a) = Q(s, a) - V(s)`, the return estimate is
        `A + V`. Deriving it from the same GAE numbers keeps the value model's
        target consistent with the advantage the policy is updated on, instead of
        letting the two drift apart with different discounting.

        Args:
            sa_pairs: Pairs carrying `advantage` and `value`.

        Returns:
            The same list, with `reward_to_go` set on every pair.
        """
        for pair in sa_pairs:
            pair["reward_to_go"] = pair["advantage"] + pair["value"]
        return sa_pairs
