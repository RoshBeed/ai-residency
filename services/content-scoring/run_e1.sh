#!/bin/sh
# The interaction sweep, sharded by alpha across five workers.
#
# One worker per alpha group rather than per run: each shard rebuilds the pooled-title
# tensors once (~1GB) and then reuses them, where sharding any finer would pay that cost
# repeatedly. Two threads each, because torch defaults to 8 and five unrestrained workers
# on ten cores spend their time fighting rather than training.
cd "$(dirname "$0")"
mkdir -p assets/e1

shard() {
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  uv run python -u ../../notebooks/content-scoring/fusion_probes.py interaction \
    --alphas $1 --hidden 64 128 256 512 \
    --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 3 \
    --out "assets/e1/shard-$2.json" > "/tmp/e1-shard-$2.log" 2>&1 \
    && echo "$(date +%H:%M:%S) shard $2 finished" \
    || echo "$(date +%H:%M:%S) shard $2 FAILED"
}

shard "0 0.125 0.25" a &
shard "0.5 0.75 1"   b &
shard "1.5 2"        c &
shard "3 4"          d &
shard "6 8"          e &
wait
echo "E1_DONE"
