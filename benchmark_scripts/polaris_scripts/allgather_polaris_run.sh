#!/bin/bash -l
# One PBS allocation runs all configurations for a fixed np, sequentially.
# submit_plan.sh supplies RUN_PLAN (immutable TSV), PLAN_NODES, PLAN_HELPER, PPN, EXE, WALL_SEC.
# Time guard: a configuration whose est_min does not fit in the remaining walltime is
# skipped (DEFERRED) instead of being killed halfway; submit_plan.py resubmits the rest.
# Manual single-configuration mode still accepts B, BASE=1, NITER=14, KINC=1, SUITE=adhoc.
# Polaris: 32 ranks per node, -d 1 --cpu-bind depth (as the author). Placement is chosen by
# submit_plan.py (GROUP=auto|tier0|tier1|none). Results keep the original chunk naming convention.
#PBS -M parleywong98@gmail.com
#PBS -m a
#PBS -l place=scatter
#PBS -l filesystems=home:eagle
#PBS -A dist_relational_alg
#PBS -k doe
#PBS -j oe

set -u
cd -- "${PBS_O_WORKDIR:?}" || exit 1
WORKDIR=$PWD
NNODES=$(wc -l < "${PBS_NODEFILE:?}")
NNODES=$(( NNODES ))
NRANKS=${PPN:-32}
NTOTRANKS=$(( NNODES * NRANKS ))
BIND=${BIND:-depth}
EXE=${EXE:-${WORKDIR}/out}
HELPER=${PLAN_HELPER:-${WORKDIR}/benchmark_scripts/polaris_scripts/submit_plan.py}
JOB=${PBS_JOBID%%.*}
[[ -x $EXE && -f $HELPER ]] || { echo "Missing executable or helper: $EXE / $HELPER"; exit 1; }
if [[ -n ${PLAN_NODES:-} && $PLAN_NODES != "$NNODES" ]]; then
  echo "Allocation mismatch: expected $PLAN_NODES nodes, received $NNODES"; exit 1
fi
[[ $BIND == depth ]] || { echo "BIND must be depth on Polaris"; exit 1; }
CPU_BIND=(-d 1 --cpu-bind depth)

check_result() {
  python3 "$HELPER" check "$1" "$SUITE" "$NNODES" "$B" "$BASE" "$NITER" "$KINC" "$NRANKS"
}

run_one() {
  local OUTDIR="$WORKDIR/results/Polaris/$SUITE"
  local TAG="results${NRANKS}_${NNODES}_${B}"
  local STEM="${TAG}__base${BASE}_it${NITER}_inc${KINC}"
  local previous
  for previous in "$OUTDIR/${STEM}_"*.csv; do
    if [[ -f $previous ]] && check_result "$previous"; then
      echo "DONE already: np=$NTOTRANKS suite=$SUITE b=$B base=$BASE n_iter=$NITER"
      return 0
    fi
  done
  # Time guard: skip this configuration if its estimate does not fit in what is left.
  if [[ -n ${WALL_SEC:-} && -n ${EST:-} ]]; then
    local need
    need=$(awk -v e="$EST" -v f="${TIME_SAFETY:-1.0}" 'BEGIN{printf "%d", e*60*f}') || need=0
    if (( SECONDS + need + ${TIME_MARGIN_SEC:-180} > WALL_SEC )); then
      echo "DEFERRED (needs ~$(( need / 60 )) min, $(( (WALL_SEC - SECONDS) / 60 )) min left): np=$NTOTRANKS suite=$SUITE b=$B base=$BASE n_iter=$NITER"
      return 99
    fi
  fi
  local RUNDIR="$OUTDIR/runs/${JOB}_n${NNODES}_b${B}_base${BASE}_it${NITER}_inc${KINC}"
  mkdir -p -- "$RUNDIR" && cd -- "$RUNDIR" || return 1
  local T0=$SECONDS
  {
    echo "date: $(date -Is)"; echo "jobid: $PBS_JOBID"; echo "queue: ${PBS_QUEUE:-}"
    echo "exe: $EXE"; echo "--- binary links:"; ldd "$EXE" | grep -Ei 'mpi|fabric'
    echo "--- mpich version:"; (mpichversion 2>/dev/null || echo "${CRAY_MPICH_VERSION:-unknown}") | head -5
    echo "--- modules:"; module list 2>&1
    echo "--- env:"; env | grep -E '^(MPICH_|MPIR_CVAR_|FI_|PALS_)' | sort
    echo "--- params: NNODES=$NNODES PPN=$NRANKS NP=$NTOTRANKS b=$B base=$BASE n_iter=$NITER radix_increment=$KINC"
    echo "--- cpu bind: ${CPU_BIND[*]}"
    head -n 1 "$PBS_NODEFILE" > hosts_first
    mpiexec --hostfile hosts_first -n "$NRANKS" -ppn "$NRANKS" "${CPU_BIND[@]}" \
      bash -c 'echo "$PALS_LOCAL_RANKID: $(grep Cpus_allowed_list /proc/self/status | cut -f2)"' < /dev/null 3<&- | sort -n | paste -sd' '
    echo "--- racks (nodes per rack):"; sed -E 's/c[0-9]+s.*//' "$PBS_NODEFILE" | sort | uniq -c | paste -sd' '
    echo "--- dragonfly groups (tier1, nodes per group):"
    for h in $(cut -d. -f1 "$PBS_NODEFILE"); do pbsnodes "$h" 2>/dev/null | awk -F' = ' '/resources_available.tier1/{print $2}'; done | sort | uniq -c | paste -sd' '
    echo "--- nodes:"; cat "$PBS_NODEFILE"
  } > "$TAG.meta" 2>&1
  echo "RUN np=$NTOTRANKS suite=$SUITE b=$B base=$BASE n_iter=$NITER radix_increment=$KINC"
  mpiexec -n "$NTOTRANKS" -ppn "$NRANKS" "${CPU_BIND[@]}" --env OMP_NUM_THREADS=1 \
    "$EXE" "$NITER" --overwrite "b=$B" "base=$BASE" "num_nodes=$NNODES" "radix_increment=$KINC" < /dev/null 3<&-
  local RC=$?
  echo "exit code: $RC" >> "$TAG.meta"
  echo "elapsed_sec: $(( SECONDS - T0 ))" >> "$TAG.meta"     # read by calibrate_plan.py
  local CHUNK="$OUTDIR/${STEM}_${JOB}.csv"
  if (( RC != 0 )); then
    [[ ! -f $TAG.csv ]] || cp -- "$TAG.csv" "${CHUNK%.csv}.FAILED"
    return "$RC"
  fi
  if ! check_result "$TAG.csv"; then
    echo "Incomplete or incorrect result: $TAG.csv" | tee -a "$TAG.meta"
    [[ ! -f $TAG.csv ]] || cp -- "$TAG.csv" "${CHUNK%.csv}.FAILED"
    return 1
  fi
  # A killed job never leaves a partial file with the successful .csv suffix.
  cp -- "$TAG.csv" "$CHUNK.partial" && mv -- "$CHUNK.partial" "$CHUNK"
}

if [[ -n ${RUN_PLAN:-} ]]; then
  [[ -f $RUN_PLAN ]] || { echo "Plan not found: $RUN_PLAN"; exit 1; }
  # Validate the entire snapshot before starting any configuration.
  count=0
  while IFS=$'\t' read -r -u 3 suite nodes b base niter inc queue wall est note; do
    [[ $suite == suite || -z $suite || $suite == \#* ]] && continue
    [[ $nodes == "$NNODES" ]] || { echo "Plan contains nodes=$nodes, expected $NNODES"; exit 1; }
    [[ $suite =~ ^[A-Za-z0-9_-]+$ && $b =~ ^[0-9]+$ && $base =~ ^[0-9]+$ && $niter =~ ^[0-9]+$ && $inc =~ ^[0-9]+$ ]] || exit 1
    (( b >= 3 && base > 0 && niter > 0 && inc > 0 )) || exit 1
    count=$(( count + 1 ))
  done 3< "$RUN_PLAN"
  (( count > 0 )) || { echo 'Empty run plan'; exit 1; }
  # Keep plan input on FD 3; MPI and benchmark stdin stay disconnected.
  processed=0 deferred=0
  while IFS=$'\t' read -r -u 3 suite nodes b base niter inc queue wall est note; do
    [[ $suite == suite || -z $suite || $suite == \#* ]] && continue
    SUITE=$suite B=$b BASE=$base NITER=$niter KINC=$inc EST=$est
    run_one < /dev/null; rc=$?
    if (( rc == 99 )); then deferred=$(( deferred + 1 )); continue; fi
    (( rc == 0 )) || exit $rc
    processed=$(( processed + 1 ))
  done 3< "$RUN_PLAN"
  if (( processed + deferred != count )); then
    echo "INCOMPLETE PLAN: np=$NTOTRANKS configurations=$processed/$count"
    exit 1
  fi
  echo "PLAN DONE FOR THIS JOB: np=$NTOTRANKS finished=$processed deferred=$deferred total=$count"
else
  B=${B:?set B or RUN_PLAN}
  BASE=${BASE:-1} NITER=${NITER:-14} KINC=${KINC:-1} SUITE=${SUITE:-adhoc}
  run_one < /dev/null || exit $?
fi
exit 0
