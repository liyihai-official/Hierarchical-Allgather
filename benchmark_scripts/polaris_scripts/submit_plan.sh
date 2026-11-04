#!/bin/bash
# Polaris: one job per node count (np = nodes * PPN), all selected configurations inside it.
# Examples:
#   DRY=1 EXE=./main ./submit_plan.sh plan_main.tsv
#   EXE=./main ./submit_plan.sh plan_main.tsv prod            # 16..256 nodes, one thread per node count, in parallel
#   EXE=./main ./submit_plan.sh plan_main.tsv debug-scaling   # 4, 8 nodes, one after another
#   EXE=./main ./submit_plan.sh plan_main.tsv debug           # 1, 2 nodes
# Options: NODES="2 4", PPN=32, QUEUE=<queue>, WALLTIME=H:MM:SS, PARALLEL=10 (prod threads),
#          GROUP=auto|tier0|tier1|none (auto: <=14 nodes one rack, <=GROUP_MAX=56 one dragonfly group),
#          TIME_SAFETY, TIME_MARGIN_SEC, AUTO_CALIB=0, CALIB_DATA=dir_or_csv[:...], BW_GBS, LAUNCH_MIN.
# Each job runs its configurations in plan order; one that no longer fits in the remaining
# walltime is skipped (DEFERRED) and the script submits a new job for what is left.
# Before each qsub it waits while you already have an ag_n<N> job in PBS and, for debug /
# debug-scaling, while you have any job in that queue. Rerunning the same command resumes.
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
exec python3 "$SCRIPT_DIR/submit_plan.py" "$@"
