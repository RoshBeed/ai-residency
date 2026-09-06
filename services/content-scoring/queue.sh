#!/bin/sh
# Work through the experiment backlog, newest-value first, and never stop on its own.
#
#   nohup ./queue.sh > /tmp/queue.log 2>&1 &
#
# Every job is resumable or idempotent, so a kill, a crash or a reboot costs the run in
# flight and nothing else. Each runs under the memory guard: a killed job is recoverable,
# a kernel panic is not.
#
# The loop at the end is deliberate. Compute here is local and already paid for, so an
# idle machine is waste; when the backlog empties it keeps adding seeds to whichever
# comparison is least able to resolve its own effect. Stopping is the user's call:
#   pkill -f queue.sh
set -e
cd "$(dirname "$0")"
GUARD="./guarded.sh --floor 6 --"
mkdir -p assets/queue
step() {
  name=$1; shift
  marker="assets/queue/$name.done"
  [ -f "$marker" ] && { echo "$(date '+%F %T') skip $name (done)"; return 0; }
  echo "$(date '+%F %T') === $name"
  if $GUARD "$@" >> "/tmp/queue-$name.log" 2>&1; then
    touch "$marker"; echo "$(date '+%F %T') --- $name complete"
  else
    echo "$(date '+%F %T') !!! $name failed (exit $?), continuing" >&2
  fi
}

# 1. Fill the under-powered points in the fusion grid; part B is the headline
#    early-vs-late table, which rests on 3 seeds and needs about 23.
step fill-gaps ./fill_gaps.sh

# 2. The literature's case FOR late fusion: graceful degradation when a modality is
#    missing or noisy. Run additive and interacting, since the two should differ.
step robust-additive uv run python -u ../../notebooks/content-scoring/fusion_probes.py \
  robustness --alpha 0 --hidden 128 --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 3 \
  --out assets/fusion-robustness-a0.json
step robust-interacting uv run python -u ../../notebooks/content-scoring/fusion_probes.py \
  robustness --alpha 2 --hidden 128 --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 3 \
  --out assets/fusion-robustness-a2.json

# 3. A finer interaction curve either side of the crossover, where the architectures
#    separate and the existing grid has only two points.
step interaction-fine uv run python -u ../../notebooks/content-scoring/fusion_probes.py \
  interaction --alphas 0.6 0.7 0.8 0.9 1.1 1.25 --hidden 128 \
  --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 3 \
  --out assets/fusion-interaction-fine.json

# 4. The hyperparameter search that never finished.
step bayes sh -c 'id=$(uv run wandb sweep sweep.yaml 2>&1 | grep -oE "[a-z0-9]+/[a-z0-9-]+/[a-z0-9]{8}" | tail -1); [ -n "$id" ] && timeout 6h uv run wandb agent "$id"'

# 5. Backlog empty: keep adding seeds to the fusion grid rather than idling. Each pass
#    tightens every point it touches; `fusion-curve` skips what it already has.
pass=0
while true; do
  pass=$((pass + 1))
  echo "$(date '+%F %T') === extra seed pass $pass"
  $GUARD make fusion-curve \
    FUSION_SEEDS="$(seq $((52 + pass * 5)) $((56 + pass * 5)) | tr '\n' ' ')" \
    >> /tmp/queue-extra-seeds.log 2>&1 || echo "$(date '+%F %T') !!! pass $pass failed" >&2
done
