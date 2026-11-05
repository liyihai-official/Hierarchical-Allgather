#!/bin/bash -l
# One job of the MPI_Allgather / radix_batch / node_shm comparison. Submitted by shm_compare_submit.sh.
# Environment: OUTDIR (required; OUTDIR/bench_args holds extra arguments for bench_shm),
#              PPN (default 102), MAX_SECONDS, SANITY=1 (also run the small correctness cases first).
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe

set -u
cd -- "${PBS_O_WORKDIR:?}" || exit 1
EXE=$PWD/bench_shm
OUTDIR=${OUTDIR:?set OUTDIR}
PPN=${PPN:-102}
BENCH_ARGS=$(cat "$OUTDIR/bench_args" 2>/dev/null)
MAX_SECONDS=${MAX_SECONDS:-0}
NNODES=$(wc -l < "${PBS_NODEFILE:?}")
NP=$(( NNODES * PPN ))
JOB=${PBS_JOBID%%.*}
TAG=n${NNODES}_ppn${PPN}_${JOB}
mkdir -p -- "$OUTDIR"

if (( PPN % 2 != 0 || PPN > 102 )); then
  echo "PPN must be even and at most 102 (cores 0 and 52 are reserved)"; exit 1
fi
HALF=$(( PPN / 2 ))
BIND="list:$(seq -s: 1 "$HALF"):$(seq -s: 53 $(( 52 + HALF )))"

{
  echo "date: $(date -Is)"; echo "jobid: $PBS_JOBID"; echo "queue: ${PBS_QUEUE:-}"
  echo "nodes: $NNODES ppn: $PPN np: $NP"; echo "bench args: $BENCH_ARGS"
  echo "--- binary links:"; ldd "$EXE" | grep -Ei 'mpi|fabric|xpmem'
  echo "--- mpich version:"; mpichversion 2>/dev/null | head -5
  echo "--- modules:"; module list 2>&1
  echo "--- env:"; env | grep -E '^(MPICH_|MPIR_CVAR_|FI_|PALS_)' | sort
  echo "--- compute node facilities:"
  mpiexec -n 1 -ppn 1 bash -c 'hostname; ls -l /dev/xpmem 2>&1; \
    echo "yama ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope 2>&1)"; \
    numactl --hardware 2>/dev/null | head -8' < /dev/null
  echo "--- racks (nodes per rack):"; sed -E 's/c[0-9]+s.*//' "$PBS_NODEFILE" | sort | uniq -c | paste -sd' '
  echo "--- nodes:"; cat "$PBS_NODEFILE"
} > "$OUTDIR/$TAG.meta" 2>&1

if [[ ${SANITY:-0} == 1 ]]; then
  SMALL="counts=1,7,1024 radices=2,4 reps=5 warmup=1"
  run_sanity() {  # label, ppn, extra arguments
    local ppn=$2
    echo "=== SANITY $1"
    mpiexec -n $(( NNODES * ppn )) -ppn "$ppn" -d 1 --cpu-bind depth "$EXE" \
      nodes="$NNODES" ppn="$ppn" $SMALL out="$OUTDIR/${TAG}_sanity_$1.csv" "${@:3}" < /dev/null
    echo "=== SANITY $1 exit code: $?"
  }
  run_sanity ppn2 2                    # G >= B needs several batches once there are 3 or more nodes
  run_sanity ppn2_inplace 2 inplace=1
  run_sanity ppn1 1                    # B = 1: every group is one process
  # A group smaller than the node must be refused by node_shm, not produce wrong results.
  run_sanity group_too_small 2 b=1
fi

echo "=== MAIN nodes=$NNODES ppn=$PPN"
T0=$SECONDS
mpiexec -n "$NP" -ppn "$PPN" --cpu-bind "$BIND" --env OMP_NUM_THREADS=1 "$EXE" \
  nodes="$NNODES" ppn="$PPN" max_seconds="$MAX_SECONDS" out="$OUTDIR/$TAG.csv" $BENCH_ARGS < /dev/null
RC=$?
echo "exit code: $RC" >> "$OUTDIR/$TAG.meta"
echo "elapsed_sec: $(( SECONDS - T0 ))" >> "$OUTDIR/$TAG.meta"
exit "$RC"
