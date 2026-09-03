#!/bin/sh
# Step 2: does per-input tower shape buy late fusion anything, and does a big enough
# early-fusion model erase the advantage?
#
# Three architectures at four budgets, every model with exactly the same number of hidden
# units. late-uniform and late-tuned differ only in tower *depth*, taken from the shape
# probe, so a gap between them is the value of shape specialisation and nothing else.
#
# Resumable via assets/tower-shapes.progress.
set -e
set -o pipefail
cd "$(dirname "$0")"
PROGRESS=assets/tower-shapes.progress
[ "$1" = "fresh" ] && rm -f "$PROGRESS"
touch "$PROGRESS"

# Depths chosen by the probe, written by pick_depths.py once Step 1 finishes.
[ -f assets/tower-depths.env ] || { echo "assets/tower-depths.env missing; run pick_depths.py" >&2; exit 1; }
. ./assets/tower-depths.env

ARCHES="early late-uniform late-tuned"
[ "$TUNED_IS_DEGENERATE" = "1" ] && ARCHES="early late-uniform"

for budget in 128 256 512 1024 2048; do
  for arch in $ARCHES; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
      key="$budget|$arch|$seed"
      grep -qxF "$key" "$PROGRESS" && continue
      # A hyphen is not legal in a shell variable name, so `$late-uniform_128_widths`
      # parses as `$late` followed by literal text. The spec file uses underscores.
      var=$(echo "$arch" | tr '-' '_')
      eval "widths=\$${var}_${budget}_widths"
      eval "towers=\$${var}_${budget}_towers"
      eval "fuse=\$${var}_${budget}_fuse"
      echo "$(date +%H:%M:%S) === budget=$budget $arch seed=$seed"
      attempt=1
      while : ; do
        if env AI_RESIDENCY_HIDDEN_WIDTHS="$widths" \
               AI_RESIDENCY_TOWER_SHAPES="$towers" \
               AI_RESIDENCY_FUSION_LAYER="$fuse" \
               AI_RESIDENCY_BLOCK_ALLOCATION=equal \
               AI_RESIDENCY_SAVE_CHECKPOINTS=false AI_RESIDENCY_PROMOTE=false \
               AI_RESIDENCY_LOG_TABLES=false AI_RESIDENCY_HF_REPO_ID="" \
           uv run python -m content_scoring.train --fusion late --seed "$seed"; then
          echo "$key" >> "$PROGRESS"; break
        fi
        [ $attempt -ge 3 ] && { echo "$(date +%H:%M:%S) !!! giving up on $key"; exit 1; }
        echo "$(date +%H:%M:%S) retry $attempt for $key in 60s"
        attempt=$((attempt + 1)); sleep 60
      done
    done
  done
done
echo "$(date '+%F %T') TOWER_SHAPES_DONE ($(wc -l < "$PROGRESS" | tr -d ' ') runs)"
