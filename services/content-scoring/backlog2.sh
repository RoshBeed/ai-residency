#!/bin/sh
# Standing backlog, second pass. The published claim is about RAW upvote counts, so the
# replications and the mechanism test run on raw -- the first backlog replicated log1p,
# which is the coordinate system in which the interaction is invisible.
set -e
cd "$(dirname "$0")"
mkdir -p assets/edge assets/queue

run() {
  name=$1; shift
  out="assets/edge/$name.json"
  [ -f "$out" ] && { echo "$(date '+%F %T') skip $name"; return 0; }
  echo "$(date '+%F %T') === $name"
  ./guarded.sh --floor 6 -- uv run python -u ../../notebooks/content-scoring/hn_interactions.py \
    --out "$out" "$@" >> "/tmp/bk2-$name.log" 2>&1 \
    && echo "$(date '+%F %T') --- $name done" \
    || echo "$(date '+%F %T') !!! $name failed" >&2
}

# 1. The headline, on fresh seeds. Three pairings were called significant on raw counts
#    from seeds 42-51; nothing has replicated them.
run raw-replicate --seeds 62 63 64 65 66 67 68 69 70 71 --epochs 25 --target raw

# 2. The mechanism. Same parameters as the additive control, "*" instead of "+".
run raw-multiplicative --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --target raw
run raw-multiplicative-b --seeds 62 63 64 65 66 67 68 69 70 71 --epochs 25 --target raw

# 3. Does one multiplication survive the transforms too? If the product model closes the
#    gap on raw and matches the additive one on log1p, the story is complete.
run log-multiplicative --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 25 --target log1p

# 4. The original backlog's unfinished tail.
run es-replicate --seeds 62 63 64 65 66 67 68 69 70 71 --epochs 25

# 5. Never idle: keep replicating raw and log1p on fresh seed blocks.
pass=0
while true; do
  pass=$((pass + 1))
  base=$((100 + pass * 10))
  seeds="$base $((base+1)) $((base+2)) $((base+3)) $((base+4)) $((base+5)) $((base+6)) $((base+7)) $((base+8)) $((base+9))"
  for target in raw log1p; do
    run "pass$pass-$target" --seeds $seeds --epochs 25 --target "$target"
  done
done
