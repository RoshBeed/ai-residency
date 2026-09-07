#!/bin/sh
# Finish the block-allocation ranking, which was screened at three seeds and could not
# separate its five arms: the gaps spanned 0.041 against a within-cell spread of 0.045.
# Ten seeds brings the standard error to about 0.005, which resolves them or shows they
# genuinely do not differ.
#
# Six constant-width layers so the taper cannot confound it; fusion at the bottom and the
# top, which is where the allocation should matter most.
set -e
cd "$(dirname "$0")"
PROGRESS=assets/allocation.progress
[ "$1" = "fresh" ] && rm -f "$PROGRESS"
touch "$PROGRESS"

for variant in "reference:proportional:none:64" \
               "V0-equal:equal:none:64" \
               "V1-title-down:equal:text_down:32" \
               "V2-features-up:equal:features_up:64" \
               "V3-all-same:equal:all:64"; do
  name=$(echo "$variant" | cut -d: -f1)
  alloc=$(echo "$variant" | cut -d: -f2)
  proj=$(echo "$variant" | cut -d: -f3)
  dim=$(echo "$variant" | cut -d: -f4)
  for point in 1 6; do
    for seed in 42 43 44 45 46 47 48 49 50 51; do
      key="$name|$point|$seed"
      grep -qxF "$key" "$PROGRESS" && continue
      echo "$(date +%H:%M:%S) === $name fusion=$point seed=$seed"
      if env AI_RESIDENCY_HIDDEN_WIDTHS=256,256,256,256,256,256 \
             AI_RESIDENCY_FUSION_LAYER="$point" \
             AI_RESIDENCY_BLOCK_ALLOCATION="$alloc" \
             AI_RESIDENCY_INPUT_PROJECTION="$proj" \
             AI_RESIDENCY_PROJECTION_DIM="$dim" \
             AI_RESIDENCY_SAVE_CHECKPOINTS=false AI_RESIDENCY_PROMOTE=false \
             AI_RESIDENCY_LOG_TABLES=false AI_RESIDENCY_HF_REPO_ID="" \
         uv run python -m content_scoring.train --fusion late --seed "$seed"; then
        echo "$key" >> "$PROGRESS"
      else
        echo "$(date +%H:%M:%S) !!! failed $key" >&2
      fi
    done
  done
done
echo "$(date '+%F %T') ALLOCATION_DONE ($(wc -l < "$PROGRESS" | tr -d ' ') runs)"
