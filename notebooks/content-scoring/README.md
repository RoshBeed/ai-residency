# Encoder and objective probes

Experiments run against the trained service to answer questions the write-up needed and
the service itself does not need to ship. They are kept as a record of how the numbers in
`services/content-scoring/README.md` §1.6 were obtained, not as part of the pipeline.

Each script expects to run from `services/content-scoring/`, which is where `Config`'s
relative paths resolve:

```bash
cd services/content-scoring
uv run python ../../notebooks/content-scoring/encoder_probes.py mini
```

They are deliberately outside the service: they hardcode an `mps` device, they download
pretrained models the service does not depend on, and they have no tests. Shipping them
beside the service would mean CI type-checking research scaffolding.

## What they measured

| probe | result |
|---|---|
| MiniLM frozen vs fine-tuned | Frozen buys +0.018 over mean-pooled word2vec; fine-tuned buys +0.060. Testing only the frozen case is how the write-up briefly concluded the text side was exhausted. |
| MiniLM at full data, 3 seeds | ρ 0.383 ±0.003, PR-AUC 0.107 -- against the shipped model's 0.366 / 0.090. |
| Qwen2.5-0.5B + LoRA | ρ 0.379 and PR-AUC **0.117** on 300k rows, the best tail detection measured. |
| Learning rate 1e-5 / 3e-5 / 5e-5 | 0.368 / 0.381 / 0.377. Barely matters; the untuned 2e-5 guess was near-optimal. |
| Objective: regression vs classification | A binary head on `score >= 100` reaches PR-AUC 0.106 and collapses rank correlation to 0.207. |
| Momentum features | No effect, three seeds, every metric. Removed from the service; see §1.6. |
