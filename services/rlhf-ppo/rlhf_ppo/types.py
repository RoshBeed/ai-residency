"""The (state, action) pair that every PPO stage reads from and writes to.

A PPO step here is a pipeline of small components, and the thing flowing between
them is a list of these dictionaries. Each stage adds its own key rather than
returning a new structure, so by the end of `PPOTrainer.ppo_step` a single pair
carries the whole trace for one generated token:

    generation      -> state_text, state_token_ids, action_token, action_token_id,
                       logprob, ref_logprob, prompt_idx, step
    KL divergence   -> kl_div
    reward adjust   -> adjusted_reward
    value model     -> value
    TD error        -> td_error
    GAE             -> advantage
    returns         -> reward_to_go

`state_text` and `state_token_ids` are the same state in two forms, and which one a
stage reads is not arbitrary. The value model scores text. The PPO ratio scores the
ids, because it has to measure the same tokens the generator did, and
decode-then-encode is not always the identity.

The values are deliberately heterogeneous (strings, ints, floats), so the alias is
`dict[str, Any]` rather than a `TypedDict`: stages run in sequence and each one
legitimately sees a pair that lacks the keys the later stages add, which a
`TypedDict` would either reject or force into `total=False` and lose all the
checking benefit anyway.
"""

from typing import Any

__all__ = ["SAPair"]

SAPair = dict[str, Any]
