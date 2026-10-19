#!/bin/sh
# The raw-loss "interaction" is confounded with the loss itself: --target changes only the
# training objective, never the metric (Spearman ranks its inputs, so log1p/raw evaluate
# identically). These three runs decide whether anything survives.
set -e
cd "$(dirname "$0")"
mkdir -p assets/edge

run() {
  name=$1; shift
  out="assets/edge/$name.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip $name"; return 0; }
  echo "$(date '+%F %T') === $name"
  ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
    --out "$out" "$@" >> "/tmp/bk3-$name.log" 2>&1 \
    && echo "$(date '+%F %T') --- $name done" \
    || echo "$(date '+%F %T') !!! $name failed" >&2
}

# 1. The one pro-interaction explanation still standing: raw MSE puts 76% of its gradient
#    on the top 1% of posts, so if the interaction is real and confined to the tail, this
#    is the run that shows it. Trained on raw, scored on the tail.
run raw-topdec --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --target raw --slice top_decile

# 2. Is the raw-loss gap just convergence speed under a noisy gradient? Early stopping
#    already scores each model at its own peak, but 25 epochs may not reach it on a loss
#    this heavy-tailed. If the additive model catches up at 60, the gap was optimisation.
run raw-long --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 60 --target raw

# 3. The ceiling test. On the loss where both models fit well, with a long budget, does
#    the joint model EVER exceed the additive model's best? At 25 epochs it does not
#    (0.3379 vs 0.3378 on title x all metadata). That is the number that matters.
run rank-long --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 60 --target rank

# Never idle.
pass=0
while true; do
  pass=$((pass + 1)); base=$((200 + pass * 10))
  seeds="$base $((base+1)) $((base+2)) $((base+3)) $((base+4)) $((base+5)) $((base+6)) $((base+7)) $((base+8)) $((base+9))"
  run "ceiling$pass" --seeds $seeds --epochs 60 --target rank
done
