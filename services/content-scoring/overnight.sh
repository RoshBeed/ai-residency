#!/bin/sh
# The fusion-point experiment, start to finish, unattended.
#
#   ./overnight.sh            # curve, then search
#   ./overnight.sh curve      # curve only
#
# Two phases, in this order deliberately: the grid is the finding and the Bayes search is
# the extra, so if the machine is reclaimed early the half that matters is already done.
# The grid is resumable -- rerunning skips combinations already in assets/fusion-curve.progress
# -- so a crash or a reboot costs the run in flight and nothing else.
#
# `caffeinate -is` because a laptop otherwise sleeps partway through. It does not prevent
# a macOS auto-update restart; turn that off separately.
set -e
# `cmd | tee log` exits with tee's status, not cmd's, so `set -e` cannot see a failing
# phase through a pipe. Without this the curve died on run 60 and the script announced
# "phase 1 complete" and moved on to the search, burning five hours on the wrong half of
# the experiment. Silence is not success.
set -o pipefail
cd "$(dirname "$0")"

LOG_DIR=${LOG_DIR:-/tmp}
BAYES_HOURS=${BAYES_HOURS:-10}

echo "$(date '+%F %T') phase 1/2: fusion curve"
if ! caffeinate -is make fusion-curve 2>&1 | tee "$LOG_DIR/fusion-curve.log"; then
  echo "$(date '+%F %T') fusion curve did not complete; not starting the search" >&2
  echo "rerun ./overnight.sh to resume from assets/fusion-curve.progress" >&2
  exit 1
fi
echo "$(date '+%F %T') phase 1 complete"

if [ "$1" = "curve" ]; then
  echo "curve only; stopping here"
  exit 0
fi

echo "$(date '+%F %T') phase 2/2: Bayes search, ${BAYES_HOURS}h"
# The id is on stderr among other output; the URL form is what `wandb agent` accepts.
SWEEP_ID=$(uv run wandb sweep sweep.yaml 2>&1 | tee "$LOG_DIR/sweep-create.log" \
  | grep -oE '[a-z0-9]+/[a-z0-9-]+/[a-z0-9]{8}' | tail -1)
if [ -z "$SWEEP_ID" ]; then
  echo "could not read a sweep id from wandb; see $LOG_DIR/sweep-create.log" >&2
  exit 1
fi
echo "$(date '+%F %T') sweep $SWEEP_ID"

# Bounded by wall clock, not --count: run times vary 2-4x with the shape drawn, so a
# count chosen for the average finishes anywhere between half and double the budget.
caffeinate -is timeout "${BAYES_HOURS}h" uv run wandb agent "$SWEEP_ID" 2>&1 \
  | tee "$LOG_DIR/fusion-bayes.log" || true
echo "$(date '+%F %T') phase 2 complete"
