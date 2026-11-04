#!/bin/bash
# One job per node count (np = nodes * PPN), with all selected configurations inside it.
# Examples:
#   DRY=1 ./submit_plan.sh plan_main.tsv
#   EXE=./main ./submit_plan.sh plan_main.tsv capacity
#   EXE=./main ./submit_plan.sh plan_main.tsv plan_batch.tsv
# Options: NODES="2 4", PPN=104, QUEUE=<eligible queue>, WALLTIME=H:MM:SS,
#          BIND=depth|list, NOGPU=1, SKIP_FLAGGED=1 (explicitly excludes flagged rows),
#          GROUP=auto|tier0|none (auto: <=GROUP_MAX=64 nodes go into one rack), TIME_SAFETY, TIME_MARGIN_SEC,
#          AUTO_CALIB=0 (keep the plan's est_min), CALIB_DATA=dir_or_csv[:...] (extra timing data),
#          BW_GBS / LAUNCH_MIN / EXTRAP_PAD (see calibrate_plan.py).
# Before every job, est_min is recomputed from your own finished results (calibrate_plan.py).
# Groups never get split into separate jobs. debug-scaling defaults to 1 hour.
# WALLTIME can override the default. See README.md.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
exec python3 "$SCRIPT_DIR/submit_plan.py" "$@"
