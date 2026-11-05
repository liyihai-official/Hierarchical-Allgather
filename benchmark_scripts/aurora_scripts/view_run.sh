#!/bin/bash -l
# Contract tests (TEST=1) and the node_view comparison, one node count after another on the
# first nodes of the allocation. Writes only under OUTDIR. Run from the project root after
# "make bench_shm test_shm_view"; OUTDIR/bench_args holds extra arguments for bench_shm.
#
#   OUT=$PWD/results/run1; mkdir -p $OUT
#   qsub -A MPICH_MCS -q debug -l select=2 -l walltime=00:20:00 -l place=scatter -o $OUT/job_a.log \
#        -v OUTDIR=$OUT,NODE_LIST=1:2,TEST=1,MAX_SECONDS=900 benchmark_scripts/aurora_scripts/view_run.sh
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe

set -u
cd -- "${PBS_O_WORKDIR:?}" || exit 1
OUTDIR=${OUTDIR:?set OUTDIR}
NODE_LIST=${NODE_LIST:?set NODE_LIST}
NODE_LIST=${NODE_LIST//:/ }   # colons between node counts: qsub -v has trouble with spaces and commas
MAX_SECONDS=${MAX_SECONDS:-0}
BENCH_ARGS=$(cat "$OUTDIR/bench_args" 2>/dev/null)
JOB=${PBS_JOBID%%.*}
PPN=102
BIND="list:$(seq -s: 1 51):$(seq -s: 53 103)"
mapfile -t HOSTS < "${PBS_NODEFILE:?}"
TOTAL=${#HOSTS[@]}
status=0

{
  echo "date: $(date -Is)"; echo "jobid: $PBS_JOBID"; echo "queue: ${PBS_QUEUE:-}"
  echo "allocated nodes: $TOTAL"; echo "node list: $NODE_LIST"; echo "bench args: $BENCH_ARGS"
  echo "--- binary links:"; ldd "$PWD/bench_shm" | grep -Ei 'mpi|fabric|xpmem'
  echo "--- racks:"; sed -E 's/c[0-9]+s.*//' "$PBS_NODEFILE" | sort | uniq -c | paste -sd' '
  echo "--- nodes:"; cat "$PBS_NODEFILE"
} > "$OUTDIR/job_$JOB.meta" 2>&1

if [[ ${TEST:-0} == 1 ]]; then
  run_test() {  # label, binary, ppn
    local label=$1 binary=$2 ppn=$3
    echo "=== TEST $label ppn=$ppn"
    if (( ppn == PPN )); then
      mpiexec -n $(( TOTAL * ppn )) -ppn "$ppn" --cpu-bind "$BIND" "$binary" "$ppn" < /dev/null
    else
      mpiexec -n $(( TOTAL * ppn )) -ppn "$ppn" -d 1 --cpu-bind depth "$binary" "$ppn" < /dev/null
    fi
    echo "=== TEST $label ppn=$ppn exit code: $?"
  }
  run_test contract "$PWD/test_shm_view" 102
  run_test contract "$PWD/test_shm_view" 2
  run_test contract "$PWD/test_shm_view" 1     # one rank per group: several batches on two nodes
  # The test must be able to catch the bug it exists for. test_shm_view_nowait is an optional build
  # of test_shm_view.cc with the spin_until(..., copied, ...) line removed from exchange().
  [[ -x $PWD/test_shm_view_nowait ]] && run_test expected_to_fail "$PWD/test_shm_view_nowait" 102
fi

for n in $NODE_LIST; do
  tag=n${n}_ppn${PPN}_${JOB}
  printf '%s\n' "${HOSTS[@]:0:n}" > "$OUTDIR/$tag.hosts"
  echo "=== MAIN nodes=$n"
  T0=$SECONDS
  mpiexec --hostfile "$OUTDIR/$tag.hosts" -n $(( n * PPN )) -ppn "$PPN" --cpu-bind "$BIND" --env OMP_NUM_THREADS=1 \
    "$PWD/bench_shm" nodes="$n" ppn="$PPN" view=1 read_pass=1 max_seconds="$MAX_SECONDS" \
    out="$OUTDIR/$tag.csv" $BENCH_ARGS < /dev/null > "$OUTDIR/$tag.log" 2>&1
  rc=$?
  echo "=== nodes=$n exit code: $rc elapsed_sec: $(( SECONDS - T0 ))"
  (( rc == 0 )) || status=$rc
done
exit "$status"
