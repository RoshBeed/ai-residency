#!/bin/sh
# Fill the under-powered and missing points, in priority order.
#
#   A  the fusion curve under a *fixed* allocation, all six points, 10 seeds.
#      We have the whole curve under proportional allocation, which strangles four of
#      the five inputs, and only its endpoints under the fixed ones. Without the middle
#      there is no way to tell whether the cliff at fusion 3 was the bottleneck or the
#      fusion point.
#   B  more seeds where the headline table rests on three runs. early-vs-late is a 0.019
#      effect against a 0.023 seed SD, so n=3 cannot see it; 30 seeds can.
#   C  more seeds on the remaining allocation variants, to separate them or show they
#      cannot be separated.
#
# Resumable: a completed combination is appended to the progress file and skipped on a
# rerun. Runs write nothing and log no table artifacts.
set -e
cd "$(dirname "$0")"
PROGRESS=assets/fill-gaps.progress
[ "$1" = "fresh" ] && rm -f "$PROGRESS"
touch "$PROGRESS"

run() {  # widths fusion alloc projection dim seed
  key="$1|$2|$3|$4|$5|$6"
  grep -qxF "$key" "$PROGRESS" && return 0
  echo "$(date +%H:%M:%S) === $1 fuse=$2 $3/$4($5) seed=$6"
  attempt=1
  while : ; do
    if env AI_RESIDENCY_HIDDEN_WIDTHS="$1" AI_RESIDENCY_FUSION_LAYER="$2" \
           AI_RESIDENCY_BLOCK_ALLOCATION="$3" AI_RESIDENCY_INPUT_PROJECTION="$4" \
           AI_RESIDENCY_PROJECTION_DIM="$5" \
           AI_RESIDENCY_SAVE_CHECKPOINTS=false AI_RESIDENCY_PROMOTE=false \
           AI_RESIDENCY_LOG_TABLES=false AI_RESIDENCY_HF_REPO_ID="" \
       uv run python -m content_scoring.train --fusion late --seed "$6"; then
      echo "$key" >> "$PROGRESS"; return 0
    fi
    [ $attempt -ge 3 ] && { echo "$(date +%H:%M:%S) !!! giving up on $key"; return 1; }
    echo "$(date +%H:%M:%S) retry $attempt for $key in 60s"
    attempt=$((attempt + 1)); sleep 60
  done
}

SIX=256,256,256,256,256,256

echo "$(date '+%F %T') A: fusion curve under fixed allocation"
for seed in 42 43 44 45 46 47 48 49 50 51; do
  for point in 1 2 3 4 5 6; do
    run "$SIX" "$point" equal features_up 64 "$seed"
    run "$SIX" "$point" equal none        64 "$seed"
  done
done

echo "$(date '+%F %T') B: more seeds on the headline early-vs-late pair"
for seed in 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71; do
  run 256,256 1 proportional none 64 "$seed"
  run 256,256 2 proportional none 64 "$seed"
done

echo "$(date '+%F %T') C: separate the remaining allocation variants"
for seed in 45 46 47 48 49 50 51; do
  for point in 1 6; do
    run "$SIX" "$point" proportional none      64 "$seed"
    run "$SIX" "$point" equal        text_down 32 "$seed"
    run "$SIX" "$point" equal        all       64 "$seed"
  done
done

echo "$(date '+%F %T') FILL_GAPS_DONE ($(wc -l < "$PROGRESS" | tr -d ' ') combinations)"
