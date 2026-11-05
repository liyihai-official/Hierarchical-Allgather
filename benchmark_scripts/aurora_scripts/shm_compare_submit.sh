#!/bin/bash
# Submit the MPI_Allgather / radix_batch / node_shm comparison; run from the project root after "make bench_shm".
#   DRY=1 benchmark_scripts/aurora_scripts/shm_compare_submit.sh
#   NODES="1 2" RUN=test QUEUE=debug WALL=00:15:00 SANITY=1 BENCH_ARGS="counts=1,64 reps=5" ...shm_compare_submit.sh
# Options: NODES, PPN (102), RUN (result folder name), ACCOUNT (MPICH_MCS), QUEUE, WALL, SANITY, BENCH_ARGS,
#          GROUP (tier0 or none: keep the nodes of a job in one rack).
# 1 to 16 nodes go to capacity, larger jobs to debug-scaling (the only queue that fits, one running job per user).
set -u
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || exit 1
NODES=${NODES:-1 2 4 8 16 32 64}
PPN=${PPN:-102}
RUN=${RUN:-$(date +%Y%m%dT%H%M%S)}
ACCOUNT=${ACCOUNT:-MPICH_MCS}
GROUP=${GROUP:-tier0}
OUTDIR=$PWD/results/Aurora/shm_compare/$RUN

[[ -x $PWD/bench_shm ]] || { echo "Run from the project root after: make bench_shm"; exit 1; }
if [[ ${DRY:-0} != 1 ]]; then
  mkdir -p -- "$OUTDIR"
  # qsub -v splits on commas, so bench_shm's arguments travel in a file.
  printf '%s\n' "${BENCH_ARGS:-}" > "$OUTDIR/bench_args"
fi

for n in $NODES; do
  if (( n <= 8 )); then wall=00:30:00; elif (( n <= 16 )); then wall=00:40:00; else wall=00:55:00; fi
  wall=${WALL:-$wall}
  IFS=: read -r h m s <<< "$wall"
  # Leave five minutes to finish the running configuration and close the files.
  max_seconds=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s - 300 ))
  if (( n <= 16 )); then queue=capacity; else queue=debug-scaling; fi
  queue=${QUEUE:-$queue}
  place=scatter
  if [[ $GROUP != none ]] && (( n > 1 )); then place=scatter:group=$GROUP; fi
  cmd=(qsub -A "$ACCOUNT" -q "$queue" -N "shmcmp_${n}n" -l "select=$n" -l "walltime=$wall" -l "place=$place"
       -o "$OUTDIR/job_n$n.log"
       -v "OUTDIR=$OUTDIR,PPN=$PPN,MAX_SECONDS=$max_seconds,SANITY=${SANITY:-0}"
       "$SCRIPT_DIR/shm_compare_run.sh")
  if [[ ${DRY:-0} == 1 ]]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    echo "n=$n queue=$queue wall=$wall place=$place: $("${cmd[@]}")"
  fi
done
echo "results: $OUTDIR"
