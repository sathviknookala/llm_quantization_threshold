#!/bin/bash
# Detached completion chain for the D11 sweep.
#
# The main process was started with --job all before the refine-slo phase existed, so its
# "all" covers decode + prefill + refine only. This waits for that PID to exit, then resumes
# anything unfinished (per-cell resume is idempotent, and it also re-measures the BF16 C=32
# cell whose engine was killed by an unrelated teardown), then bisects the SLO crossing.
#
# Usage: finish_sweep.sh <pid-of-main-sweep>
set -u
REPO=/home/sathvik/llm_quantization_threshold
PY=/home/sathvik/miniconda3/envs/qnt/bin/python
LOG=$REPO/results/sweep/run.log
MAIN_PID=${1:?need the main sweep pid}
cd "$REPO"

say() { echo "[$(date '+%F %H:%M:%S')] FOLLOWUP: $*" >> "$LOG"; }

say "armed, waiting on main sweep pid=$MAIN_PID"
while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 60; done
say "main sweep pid=$MAIN_PID exited"

# never start a timed run against a GPU something else still holds
for _ in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 512 ] && break
    sleep 10
done
say "GPU free (${used:-?} MiB); resuming unfinished cells"

$PY scripts/harness/run_sweep.py --job all --reps 1,2,3 >> "$LOG" 2>&1
say "resume pass exited rc=$?"

$PY scripts/harness/run_sweep.py --job refine-slo >> "$LOG" 2>&1
say "refine-slo exited rc=$?"

$PY scripts/harness/summarize_sweep.py >> "$LOG" 2>&1
say "ALL PHASES COMPLETE"
date -Iseconds > "$REPO/results/sweep/COMPLETE"
