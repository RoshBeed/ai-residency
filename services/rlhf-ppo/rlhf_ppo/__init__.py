"""RLHF with PPO, built from the pieces rather than from a library.

One package, laid out by the stage of a PPO step it belongs to:

    base_setup/         the shared backbone, tokenizer and LoRA adapters
    models/             policy, reference (SFT), reward and value heads on that backbone
    ppo/                one PPO step end to end -- generate, score, update
    reward_kl/          the KL penalty and the reward it adjusts
    advantage/          TD error, GAE, rewards-to-go
    training/           the losses, the ratio, and the run loop
    token_attribution/  (state, action) pairs recovered from generated text
    buffer.py           the experience one update reads
    types.py            SAPair, the dictionary every stage above reads and adds to

These were nine top-level packages, which put generic names -- `models`, `training`,
`advantage` -- into a namespace shared with eight other services that each have their
own `config` and their own idea of what `models` means. mypy already has to run per
service because of that collision; this stops making it worse. `sa_pair` and
`experience_buffer` were whole packages holding one type alias and one class.
"""

__all__: list[str] = []
