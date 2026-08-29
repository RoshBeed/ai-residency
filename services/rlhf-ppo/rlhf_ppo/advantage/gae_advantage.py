"""Generalised Advantage Estimation."""

from collections import defaultdict

from rlhf_ppo.types import SAPair

__all__ = ["GAEAdvantage"]


class GAEAdvantage:
    """Turns per-step TD errors into GAE advantages."""

    @staticmethod
    def compute_gae_advantages(
        sa_pairs: list[SAPair], gamma: float = 0.99, lam: float = 0.95
    ) -> list[SAPair]:
        """Add an `advantage` key to every pair, in place.

        GAE is an exponentially weighted sum of future TD errors, so it interpolates
        between the low-variance one-step estimate (`lam=0`) and the high-variance
        full Monte Carlo return (`lam=1`). It is computed with a single backwards
        pass per sequence: each step's advantage is its own TD error plus a
        discounted copy of the advantage of the step after it.

        Args:
            sa_pairs: Pairs carrying `td_error`, `prompt_idx` and `step`.
            gamma: Discount factor.
            lam: GAE lambda, the bias/variance dial described above.

        Returns:
            The same list, with `advantage` set on every pair.
        """
        grouped: dict[int, list[SAPair]] = defaultdict(list)
        for pair in sa_pairs:
            grouped[pair["prompt_idx"]].append(pair)

        for pairs in grouped.values():
            ordered = sorted(pairs, key=lambda item: int(item["step"]))
            gae = 0.0
            for pair in reversed(ordered):
                gae = float(pair["td_error"]) + gamma * lam * gae
                pair["advantage"] = gae

        return sa_pairs
