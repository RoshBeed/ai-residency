#!/bin/sh
# Run a long job under a memory guard.
#
#   ./guarded.sh [--floor GB] -- <command>...
#
# macOS has an OOM killer (jetsam), but it did not save this machine: a probe allocated
# 27 GB in one tensor gather, the compressor hit 100% of its segment limit, the VM
# subsystem stalled, and the kernel panicked on a watchdog timeout because `watchdogd`
# had not been scheduled for 91 seconds. By the time the kernel is in that state nothing
# in userspace can react -- so this reacts earlier, from outside, on a threshold well
# above where the system gets into trouble.
#
# Kills the job, not the machine. A killed sweep is resumable; a kernel panic is not.
set -e
FLOOR_GB=6
INTERVAL=5

while [ $# -gt 0 ]; do
  case "$1" in
    --floor) FLOOR_GB=$2; shift 2 ;;
    --interval) INTERVAL=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
[ $# -gt 0 ] || { echo "usage: guarded.sh [--floor GB] -- <command>..." >&2; exit 2; }

kill_tree() {
  # Signal a process and its descendants. Not `kill -PGID`: the guard shares the job's
  # process group, so signalling the group kills the guard before it can escalate --
  # and, when run from a shell, the shell too.
  for child in $(pgrep -P "$1" 2>/dev/null); do kill_tree "$child" "$2"; done
  kill -"$2" "$1" 2>/dev/null || true
}

available_gb() {
  # Free + inactive + speculative is what the system can hand out without compressing.
  vm_stat | awk '
    /page size of/ { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+$/) { page = $i; break } }
    /Pages free/            { gsub(/\./, "", $3); free = $3 }
    /Pages inactive/        { gsub(/\./, "", $3); inactive = $3 }
    /Pages speculative/     { gsub(/\./, "", $3); spec = $3 }
    END { printf "%d", (free + inactive + spec) * page / 1073741824 }'
}

"$@" &
JOB=$!
echo "$(date +%H:%M:%S) guarded: pid $JOB, floor ${FLOOR_GB}GB, polling ${INTERVAL}s"

while kill -0 "$JOB" 2>/dev/null; do
  FREE=$(available_gb)
  if [ "$FREE" -lt "$FLOOR_GB" ]; then
    echo "$(date +%H:%M:%S) MEMORY FLOOR HIT: ${FREE}GB available < ${FLOOR_GB}GB — killing $JOB" >&2
    kill_tree "$JOB" TERM
    sleep 5
    kill_tree "$JOB" KILL
    exit 3
  fi
  sleep "$INTERVAL"
done
wait "$JOB"
STATUS=$?
echo "$(date +%H:%M:%S) guarded: job exited $STATUS"
exit $STATUS
