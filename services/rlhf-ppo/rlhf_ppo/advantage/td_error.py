"""Temporal-difference errors, the input GAE smooths over."""

from collections import defaultdict

from rlhf_ppo.types import SAPair

__all__ = ["TDError"]


class TDError:
    """Computes the one-step TD error for each (s, a) pair."""

    @staticmethod
    def compute_td_errors(sa_pairs: list[SAPair], gamma: float = 0.99) -> list[SAPair]:
        """Add a `td_error` key to every pair, in place.

        The TD error is how much better the outcome turned out than the value model
        expected: `r + gamma * V(s') - V(s)`. Pairs are grouped by prompt and ordered
        by step first, because "the next state" only means anything within one
        generated sequence.

        Args:
            sa_pairs: Pairs carrying `adjusted_reward`, `value`, `prompt_idx`, `step`.
            gamma: Discount applied to the successor state's value.

        Returns:
            The same list, with `td_error` set on every pair.
        """
        grouped: dict[int, list[SAPair]] = defaultdict(list)
        for pair in sa_pairs:
            grouped[pair["prompt_idx"]].append(pair)

        for pairs in grouped.values():
            ordered = sorted(pairs, key=lambda item: int(item["step"]))
            for step, pair in enumerate(ordered):
                # The final token of a sequence has no successor state, so V(s') is 0.
                is_last = step == len(ordered) - 1
                next_value = 0.0 if is_last else float(ordered[step + 1]["value"])
                pair["td_error"] = (
                    float(pair["adjusted_reward"]) + gamma * next_value - float(pair["value"])
                )

        return sa_pairs
