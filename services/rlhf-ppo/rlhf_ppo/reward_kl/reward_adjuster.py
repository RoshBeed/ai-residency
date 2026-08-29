"""The KL penalty applied to the reward model's score."""

from rlhf_ppo.types import SAPair

__all__ = ["RewardAdjuster"]


class RewardAdjuster:
    """Penalises a sequence's reward by how far the policy drifted to produce it."""

    @staticmethod
    def adjust_rewards_with_kl(
        sa_pairs: list[SAPair], rewards: list[float], kl_coef: float = 0.1
    ) -> list[SAPair]:
        """Add an `adjusted_reward` key to every pair, in place.

        Without this term the policy is free to walk arbitrarily far from the
        reference model to chase whatever the reward model happens to like, which is
        how reward hacking and degenerate text show up in practice. The sequence-level
        reward is the same for every token in that sequence; only the KL term varies
        per token.

        Args:
            sa_pairs: Pairs carrying `kl_div` and `prompt_idx`.
            rewards: One reward per prompt, indexed by `prompt_idx`.
            kl_coef: Weight on the KL penalty.

        Returns:
            The same list, with `adjusted_reward` set on every pair.
        """
        for pair in sa_pairs:
            reward = rewards[pair["prompt_idx"]]
            pair["adjusted_reward"] = reward - kl_coef * pair["kl_div"]
        return sa_pairs
