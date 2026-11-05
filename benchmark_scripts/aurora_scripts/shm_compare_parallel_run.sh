#!/bin/bash -l
# Several node counts of the MPI_Allgather / radix_batch / node_shm comparison at once, on disjoint
# nodes of one allocation. Queues such as prod have a minimum job size larger than the runs need.
#
#   mkdir -p $OUT; echo "64 32 16" > $OUT/node_sets
#   qsub -A MPICH_MCS -q prod -l select=256 -l walltime=00:45:00 -l place=scatter \
#        -o $OUT/job.log -v OUTDIR=$OUT,MAX_SECONDS=2400 benchmark_scripts/aurora_scripts/shm_compare_parallel_run.sh
#
# OUTDIR/node_sets and OUTDIR/bench_args are read when the job starts, so they can still be changed
# while it is queued. Runs share the network, so heavy runs can slow each other down a little.
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe

set -u
cd -- "${PBS_O_WORKDIR:?}" || exit 1
EXE=$PWD/bench_shm
OUTDIR=${OUTDIR:?set OUTDIR}
PPN=${PPN:-102}
MAX_SECONDS=${MAX_SECONDS:-0}
NODE_SETS=$(cat "$OUTDIR/node_sets") || exit 1
BENCH_ARGS=$(cat "$OUTDIR/bench_args" 2>/dev/null)
JOB=${PBS_JOBID%%.*}
mapfile -t HOSTS < "${PBS_NODEFILE:?}"

NEEDED=0
for n in $NODE_SETS; do NEEDED=$(( NEEDED + n )); done
if (( NEEDED > ${#HOSTS[@]} )); then
  echo "node_sets '$NODE_SETS' need $NEEDED nodes, the allocation has ${#HOSTS[@]}"; exit 1
fi
if (( PPN % 2 != 0 || PPN > 102 )); then
  echo "PPN must be even and at most 102 (cores 0 and 52 are reserved)"; exit 1
fi
HALF=$(( PPN / 2 ))
BIND="list:$(seq -s: 1 "$HALF"):$(seq -s: 53 $(( 52 + HALF )))"

{
  echo "date: $(date -Is)"; echo "jobid: $PBS_JOBID"; echo "queue: ${PBS_QUEUE:-}"
  echo "allocated nodes: ${#HOSTS[@]}"; echo "node sets: $NODE_SETS"; echo "ppn: $PPN"; echo "bench args: $BENCH_ARGS"
  echo "--- binary links:"; ldd "$EXE" | grep -Ei 'mpi|fabric|xpmem'
  echo "--- mpich version:"; mpichversion 2>/dev/null | head -5
  echo "--- modules:"; module list 2>&1
  echo "--- env:"; env | grep -E '^(MPICH_|MPIR_CVAR_|FI_|PALS_)' | sort
} > "$OUTDIR/job_$JOB.meta" 2>&1

offset=0
index=0
pids=()
tags=()
for n in $NODE_SETS; do
  tag=n${n}_ppn${PPN}_${JOB}_set${index}
  printf '%s\n' "${HOSTS[@]:offset:n}" > "$OUTDIR/$tag.hosts"
  echo "=== start $tag on nodes $offset to $(( offset + n - 1 ))"
  mpiexec --hostfile "$OUTDIR/$tag.hosts" -n $(( n * PPN )) -ppn "$PPN" --cpu-bind "$BIND" --env OMP_NUM_THREADS=1 \
    "$EXE" nodes="$n" ppn="$PPN" max_seconds="$MAX_SECONDS" out="$OUTDIR/$tag.csv" $BENCH_ARGS \
    < /dev/null > "$OUTDIR/$tag.log" 2>&1 &
  pids+=($!)
  tags+=("$tag")
  offset=$(( offset + n ))
  index=$(( index + 1 ))
done

status=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"
  rc=$?
  echo "=== ${tags[$i]} exit code: $rc elapsed_sec: $SECONDS"
  (( rc == 0 )) || status=$rc
done
exit "$status"
