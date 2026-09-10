#!/bin/sh
# Standing backlog. Runs to completion, then keeps replicating rather than idling.
set -e
cd "$(dirname "$0")"
mkdir -p assets/edge assets/queue

run() {  # name, then args to hn_interactions
  name=$1; shift
  out="assets/edge/$name.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip $name"; return 0; }
  echo "$(date '+%F %T') === $name"
  ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
    --out "$out" "$@" >> "/tmp/bk-$name.log" 2>&1 \
    && echo "$(date '+%F %T') --- $name done" \
    || echo "$(date '+%F %T') !!! $name failed" >&2
}

# Slices, now with early stopping -- the earlier slice results used a fixed budget and are
# not comparable to the early-stopped table.
run es-showhn   --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --slice show_hn
run es-topdec   --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --slice top_decile
run es-hasurl   --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --slice has_url

# The one interaction that survived every coordinate system deserves an independent
# replication on fresh seeds before the write-up leans on it.
run es-replicate --seeds 62 63 64 65 66 67 68 69 70 71 --epochs 25

# The synthetic hierarchy, unfinished: which interactions a dot product cannot reach.
for kind in max disagree; do
  out="assets/fusion-bilinear-$kind.json"
  [ -f "$out" ] && continue
  echo "$(date '+%F %T') === bilinear-$kind"
  ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/fusion_probes.py \
    interaction --kind "$kind" --alphas 0 0.5 1 2 4 8 --hidden 128 \
    --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 3 --out "$out" \
    >> "/tmp/bk-bilinear-$kind.log" 2>&1 || echo "$(date '+%F %T') !!! bilinear-$kind failed" >&2
done

# Nothing left: keep replicating the main table on new seeds rather than stopping.
pass=0
while true; do
  pass=$((pass + 1))
  lo=$((100 + pass * 10)); hi=$((109 + pass * 10))
  echo "$(date '+%F %T') === replication pass $pass (seeds $lo-$hi)"
  ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
    --seeds $(seq $lo $hi | tr '\n' ' ') --epochs 25 \
    --out "assets/edge/es-pass$pass.json" >> /tmp/bk-passes.log 2>&1 \
    || echo "$(date '+%F %T') !!! pass $pass failed" >&2
done
