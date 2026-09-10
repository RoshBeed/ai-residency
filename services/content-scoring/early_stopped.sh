#!/bin/sh
# The interaction comparison with each model scored at its own best validation epoch.
#
# Without early stopping a fixed budget compares two models that are both past their peak,
# and the two architectures do not peak together: at 25 epochs the additive model had
# fallen from 0.291 to 0.270 while the joint model held at 0.285, which reads as the joint
# model finding an interaction when it is really the additive one overfitting sooner.
#
# All three targets, because additivity is a property of a coordinate system: raw counts
# showed six significant interactions where log1p and rank showed none.
set -e
cd "$(dirname "$0")"
mkdir -p assets/edge
for t in log1p raw rank; do
  out="assets/edge/es-$t.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip es-$t"; continue; }
  echo "$(date '+%F %T') === early-stopped, target=$t"
  if ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
       --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --target "$t" --out "$out" \
       >> "/tmp/es-$t.log" 2>&1; then
    echo "$(date '+%F %T') --- es-$t done"
  else
    echo "$(date '+%F %T') !!! es-$t failed" >&2
  fi
done
echo "$(date '+%F %T') EARLY_STOPPED_DONE"
