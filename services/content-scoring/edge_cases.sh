#!/bin/sh
# Rule out the alternatives to "Hacker News is additive".
#
# A null is only as strong as the explanations it has eliminated. Each run below varies
# one thing that could manufacture a null on its own, holding everything else fixed. The
# title x metadata pairings are the ones the fusion question turns on, so every variant
# reruns all of them.
#
# Resumable: a completed variant writes its JSON and is skipped on a rerun.
set -e
cd "$(dirname "$0")"
mkdir -p assets/edge

run() {  # name, then arguments
  name=$1; shift
  out="assets/edge/$name.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip $name"; return 0; }
  echo "$(date '+%F %T') === $name"
  if ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
       --seeds 42 43 44 45 46 47 48 49 50 51 --out "$out" "$@" >> "/tmp/edge-$name.log" 2>&1; then
    echo "$(date '+%F %T') --- $name done"
  else
    echo "$(date '+%F %T') !!! $name failed" >&2
  fi
}

# 1. The transform. Additive in log1p is multiplicative in raw counts, so the finding is
#    a statement about a coordinate system until checked in more than one. `rank` is the
#    scale Spearman actually scores on.
run target-raw    --target raw    --epochs 3
run target-rank   --target rank   --epochs 3

# 2. The training budget. Interactions are harder to learn than main effects, so a short
#    run finds the additive part and stops. If the gap grows with epochs, the null was
#    under-training rather than a property of the data.
run epochs-10     --target log1p  --epochs 10
run epochs-25     --target log1p  --epochs 25

# 3. Subgroups. An interaction confined to Show HN posts or the top decile averages away
#    over 69,000 rows.
run slice-showhn  --target log1p  --epochs 3 --slice show_hn
run slice-topdec  --target log1p  --epochs 3 --slice top_decile
run slice-hasurl  --target log1p  --epochs 3 --slice has_url

echo "$(date '+%F %T') EDGE_CASES_DONE"
