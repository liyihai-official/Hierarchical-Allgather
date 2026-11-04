#!/usr/bin/env python3
"""Submit one PBS allocation per node count, across one or more Polaris plans.

Polaris port of the Aurora script: same plan format, same runner logic (time guard,
automatic resubmission, calibration). Differences: 32 ranks per node, placement
one rack (tier0, 14 nodes) or one dragonfly group (tier1), Polaris queue limits, and
prod node counts are submitted in parallel (one submit thread per node count)."""
import argparse
import collections
import csv
import datetime
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
import time

FIELDS = ('suite', 'nodes', 'b', 'base', 'n_iter', 'radix_increment',
          'queue', 'walltime', 'est_min', 'note')


def complete(path, row, ppn):
    """Validate every expected (algorithm, k, size), including correctness and repetitions."""
    b, base, it, inc, nodes = (int(row[k]) for k in
                              ('b', 'base', 'n_iter', 'radix_increment', 'nodes'))
    expected = collections.Counter()
    for i in range(it):
        size = base << i
        for k in range(2, b, inc):
            expected['allgather_radix_batch', k, b, size] = 50
        expected['allgather_standard', 0, 0, size] = 50
    observed = collections.Counter()
    try:
        with open(path, newline='') as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ['algorithm_name', 'k', 'b', 'nprocs', 'send_count', 'time', 'is_correct']:
                return False
            for r in reader:
                if int(r['nprocs']) != ppn * nodes or r['is_correct'] != '1':
                    return False
                elapsed = float(r['time'])
                if not math.isfinite(elapsed) or elapsed < 0:
                    return False
                observed[r['algorithm_name'], int(r['k']), int(r['b']), int(r['send_count'])] += 1
        return observed == expected
    except (OSError, ValueError, TypeError, KeyError, csv.Error):
        return False


def done(row, ppn, root):
    directory = root / 'results' / 'Polaris' / row['suite']
    pattern = (f"results{ppn}_{row['nodes']}_{row['b']}__base{row['base']}"
               f"_it{row['n_iter']}_inc{row['radix_increment']}_*.csv")
    return any(complete(path, row, ppn) for path in directory.glob(pattern))


def placement(nodes):
    """PBS group for this job on Polaris: one rack (tier0, 14 nodes) when it fits, else one
    dragonfly group (tier1, ~4 racks) up to GROUP_MAX nodes, else none.
    GROUP=auto (default) | tier0 | tier1 | none."""
    mode = os.environ.get('GROUP', 'auto')
    if mode == 'none':
        return None
    if mode == 'auto':
        if nodes <= 14:
            return 'tier0'
        return 'tier1' if nodes <= int(os.environ.get('GROUP_MAX', '56')) else None
    if mode not in ('tier0', 'tier1'):
        raise ValueError('GROUP must be auto, tier0, tier1 or none')
    return mode


PLACE_NAME = {'tier0': 'one rack', 'tier1': 'one dragonfly group'}


def minutes(value):
    h, m, s = map(int, value.split(':'))
    if h < 0 or not 0 <= m < 60 or not 0 <= s < 60:
        raise ValueError('walltime must be H:MM:SS')
    return h * 60 + m + s / 60


def max_minutes(queue, nodes):
    """Longest walltime Polaris allows for this queue and node count (prod routes by size)."""
    if queue == 'prod':
        return 180 if nodes <= 24 else 360 if nodes <= 99 else 1440
    return {'debug': 60, 'debug-scaling': 60, 'preemptable': 4320}.get(queue, 1440)


def walltime(value):
    m = max(15, math.ceil(value * 1.5 / 5) * 5)
    return f'{m // 60}:{m % 60:02d}:00'


def load(paths):
    rows, seen = [], set()
    for path in paths:
        with open(path, newline='') as stream:
            reader = csv.DictReader(stream, delimiter='\t')
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f'{path}: unexpected plan header')
            for r in reader:
                if not r['suite'] or r['suite'].startswith('#'):
                    continue
                if not re.fullmatch(r'[A-Za-z0-9_-]+', r['suite']):
                    raise ValueError(f'{path}: invalid suite')
                for key in ('nodes', 'b', 'base', 'n_iter', 'radix_increment'):
                    if int(r[key]) < 1:
                        raise ValueError(f'{path}: {key} must be positive')
                if int(r['b']) < 3:
                    raise ValueError(f'{path}: b must be at least 3')
                estimate = float(r['est_min'])
                if not math.isfinite(estimate) or estimate < 0:
                    raise ValueError(f'{path}: invalid est_min')
                key = tuple(r[k] for k in FIELDS[:6])
                if key not in seen:
                    seen.add(key)
                    rows.append(r)
    return rows


def recalibrate(rows, root, verbose=False):
    """Re-estimate est_min from your own results (calibrate_plan.py). AUTO_CALIB=0 keeps the plan's
    est_min; CALIB_DATA=dir_or_csv[:...] adds older result folders or a merged results_all.csv."""
    if os.environ.get('AUTO_CALIB', '1') == '0' or not rows:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import calibrate_plan
    extra = [p for p in os.environ.get('CALIB_DATA', '').split(':') if p]
    model = calibrate_plan.Model(calibrate_plan.default_sources(str(root), {r['suite'] for r in rows}) + extra)
    model.apply(rows)
    if verbose:
        print(model.summary())


def submit(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plans', nargs='+', help='plan TSV files; optional trailing queue filter')
    parser.add_argument('--queue', help='override the queue for every selected node count')
    parser.add_argument('--walltime', help='override grouped H:MM:SS walltime')
    args = parser.parse_args(argv)
    # Preserve the original "plan_main.tsv capacity" / "debug-scaling" syntax.
    qfilter = ''
    if len(args.plans) > 1 and not Path(args.plans[-1]).is_file() and not args.plans[-1].endswith('.tsv'):
        qfilter = args.plans.pop()
    root = Path.cwd()
    ppn = int(os.environ.get('PPN', '32'))
    if ppn < 1:
        raise ValueError('PPN must be positive')
    rows = load(args.plans)
    recalibrate(rows, root, verbose=True)
    selected = os.environ.get('NODES', '').split()
    grouped = collections.defaultdict(list)
    for row in rows:
        if qfilter and row['queue'] != qfilter:
            continue
        if selected and row['nodes'] not in selected:
            continue
        if os.environ.get('SKIP_FLAGGED') == '1' and 'MAY_EXCEED_1H' in row['note']:
            print(f"SKIP flagged: {row['suite']} n={row['nodes']} b={row['b']} base={row['base']}")
            continue
        if done(row, ppn, root):
            print(f"DONE already: {row['suite']} n={row['nodes']} b={row['b']} base={row['base']}")
            continue
        grouped[int(row['nodes'])].append(row)
    dry = os.environ.get('DRY') == '1'
    if not grouped:
        print('No pending configurations.')
        return 0
    exe = Path(os.environ.get('EXE', './out')).resolve()
    if not dry and (not exe.is_file() or not os.access(exe, os.X_OK)):
        raise ValueError(f'binary not found or not executable: {exe} (set EXE=./main)')
    jobs, errors = [], []
    for nodes, configs in sorted(grouped.items()):
        queues = {r['queue'] for r in configs}
        queue = args.queue or os.environ.get('QUEUE')
        if not queue:
            if len(queues) != 1:
                raise ValueError(f'nodes={nodes}: inconsistent queues; use --queue')
            queue = queues.pop()
        estimate = sum(float(r['est_min']) for r in configs)
        wall = args.walltime or os.environ.get('WALLTIME') or walltime(estimate)
        cap = max_minutes(queue, nodes)
        if not (args.walltime or os.environ.get('WALLTIME')) and minutes(wall) > cap:
            wall = f'{cap // 60}:{cap % 60:02d}:00'     # the time guard + resubmission handle the rest
        duration = minutes(wall)
        limits = {'debug': (1, 2, 60), 'debug-scaling': (1, 10, 60),
                  'prod': (10, 496, max_minutes('prod', nodes)), 'preemptable': (1, 10, 4320)}
        problems = []
        margin_s = int(os.environ.get('TIME_MARGIN_SEC', '180'))
        safety_f = float(os.environ.get('TIME_SAFETY', '1.0'))
        unfit = [r for r in configs if float(r['est_min']) * 60 * safety_f + margin_s > duration * 60]
        if queue in limits:
            low, high, maximum = limits[queue]
            if not low <= nodes <= high or not 5 <= duration <= maximum:
                problems.append(f'{queue} supports {low}..{high} nodes and 5..{maximum} min; '
                                'choose an eligible long-running queue with --queue/QUEUE')
        group = placement(nodes)
        print(f"{'DRY ' if dry else ''}JOB np={nodes * ppn} nodes={nodes} "
              f"configs={len(configs)} suites={','.join(sorted({r['suite'] for r in configs}))} "
              f"queue={queue} walltime={wall} est={estimate:.1f}min "
              f"placement={PLACE_NAME[group] + ' (group=' + group + ')' if group else 'not grouped'}")
        fit = sum(float(r['est_min']) for r in configs if r not in unfit)
        if duration < fit:
            print(f'  NOTE: {fit:.0f} min of work > walltime {wall}: runs as ~{math.ceil(fit / max(duration - margin_s / 60, 1))} '
                  'consecutive jobs (each job runs the chunks that still fit; the next job continues).')
        for r in unfit:
            print(f"  WILL NOT SUBMIT b={r['b']} base={r['base']} n_iter={r['n_iter']}: est {r['est_min']} min "
                  f'> walltime {wall} minus margin; re-checked after every job (raise WALLTIME if needed)')
        for problem in problems:
            print(f'  QUEUE/TIME ERROR: {problem}')
        errors.extend(problems)
        jobs.append((nodes, configs, queue, wall))
    print(f'{len(jobs)} grouped job(s), {sum(len(j[1]) for j in jobs)} pending configuration chunk(s).')
    if dry:
        return 0
    # Check actual queue limits before submitting; runtime estimates are advisory.
    if errors:
        raise ValueError('queue/time checks failed; no jobs submitted (DRY=1 previews all groups)')
    if os.environ.get('SKIP_FLAGGED') == '1':
        print('SKIP_FLAGGED=1: flagged configurations are excluded.')
    runner = Path(__file__).with_name('allgather_polaris_run.sh').resolve()
    logpath = root / ('submit_' + '_'.join(Path(p).stem for p in args.plans) +
                      (f'_{qfilter}' if qfilter else '') + '.log')
    safety = float(os.environ.get('TIME_SAFETY', '1.0'))
    margin = int(os.environ.get('TIME_MARGIN_SEC', '180'))
    stuck = []
    lock = threading.Lock()

    def run_node(nodes, configs, queue, wall):
        """All jobs of one node count, one after another (resubmitting what is left)."""
        mine = []
        wall_sec = minutes(wall) * 60
        pending, attempt = list(configs), 0
        while pending:
            # Every finished job adds measured durations: re-estimate before each submission,
            # so rows that looked too long may become runnable (and the reverse).
            recalibrate(pending, root)
            too_long = [r for r in pending if float(r['est_min']) * 60 * safety + margin > wall_sec]
            runnable = [r for r in pending if r not in too_long]
            if not runnable:
                for r in too_long:
                    print(f"  NOT SUBMITTED nodes={nodes} b={r['b']} base={r['base']} n_iter={r['n_iter']}: "
                          f"est {r['est_min']} min does not fit in walltime {wall}; raise WALLTIME", flush=True)
                mine += [(nodes, r) for r in too_long]
                break
            attempt += 1
            rc = submit_one(nodes, runnable, queue, wall, ppn, exe, runner, root, logpath, attempt, lock)
            left = [r for r in pending if not done(r, ppn, root)]
            finished = len(pending) - len(left)
            print(f'  nodes={nodes} job #{attempt}: rc={rc}, finished {finished}, still pending {len(left)}',
                  flush=True)
            if finished == 0 and left:
                print(f'  STOP nodes={nodes}: no progress in job #{attempt} (rc={rc}); see {logpath} '
                      'and the job .o file.', file=sys.stderr, flush=True)
                mine += [(nodes, r) for r in left]
                break
            pending = left
        return mine

    # debug / debug-scaling: one node count after another (per-user limits);
    # other queues (prod): every node count in its own thread, so they queue and run in parallel.
    serial = [j for j in jobs if j[2] in ('debug', 'debug-scaling')]
    parallel = [j for j in jobs if j[2] not in ('debug', 'debug-scaling')]
    for job in serial:
        stuck += run_node(*job)
    if parallel:
        with ThreadPoolExecutor(max_workers=int(os.environ.get('PARALLEL', '10'))) as pool:
            for result in pool.map(lambda j: run_node(*j), parallel):
                stuck += result
    if stuck:
        print(f'{len(stuck)} configuration(s) not finished:', file=sys.stderr)
        for nodes, r in stuck:
            print(f"    nodes={nodes} b={r['b']} base={r['base']} n_iter={r['n_iter']} est={r['est_min']}min",
                  file=sys.stderr)
        return 1
    return 0


def my_jobs(*args):
    """Job ids of this user's queued/running/held jobs matching qselect args (empty if qselect fails)."""
    user = os.environ.get('USER', '')
    try:
        out = subprocess.run(['qselect', '-u', user, '-s', 'QRHW', *args],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             universal_newlines=True, timeout=120).stdout   # Python 3.6 compatible
    except (OSError, subprocess.SubprocessError):
        return []
    return out.split()


def wait_until_free(nodes, queue, logpath):
    """Do not stack jobs: wait while another ag_n<nodes> job of yours exists (e.g. from an older
    submit loop), and, for debug/debug-scaling (1 job per user), while any of your jobs is in that queue."""
    waited = False
    while True:
        busy = my_jobs('-N', f'ag_n{nodes}')
        if queue in ('debug', 'debug-scaling'):
            busy += my_jobs('-q', queue)
        if not busy:
            return
        if not waited:
            line = (f'{datetime.datetime.now().isoformat()} WAIT nodes={nodes}: your job(s) '
                    f'{" ".join(sorted(set(busy)))} still in PBS; submitting when they are gone')
            print(line, flush=True)
            with open(logpath, 'a') as log:
                log.write(line + '\n')
            waited = True
        time.sleep(60)


def submit_one(nodes, configs, queue, wall, ppn, exe, runner, root, logpath, attempt, lock=None):
    """Submit one blocking job that runs `configs` (sequentially, with the time guard)."""
    wait_until_free(nodes, queue, logpath)
    snapshot_dir = root / 'results' / 'Polaris' / 'plans'
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', prefix=f'n{nodes}_', suffix='.tsv',
                                     dir=snapshot_dir, delete=False, newline='') as stream:
        writer = csv.DictWriter(stream, FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(configs)
        snapshot = Path(stream.name)
    variables = {'RUN_PLAN': str(snapshot), 'PLAN_NODES': str(nodes), 'PPN': str(ppn), 'EXE': str(exe),
                 'PLAN_HELPER': str(Path(__file__).resolve()), 'WALL_SEC': str(int(minutes(wall) * 60))}
    for key in ('BIND', 'TIME_SAFETY', 'TIME_MARGIN_SEC'):
        if key in os.environ:
            variables[key] = os.environ[key]
    if any(',' in v or '\n' in v for v in variables.values()):
        raise ValueError('PBS -v paths/values cannot contain commas or newlines')
    group = placement(nodes)
    place = ['-l', f'place=scatter:group={group}'] if group else []
    cmd = ['qsub', '-W', 'block=true', '-q', queue, '-l', f'select={nodes}:system=polaris', '-l', f'walltime={wall}', *place,
           '-N', f'ag_n{nodes}', '-v', ','.join(f'{k}={v}' for k, v in variables.items()), str(runner)]
    with open(logpath, 'a') as log:
        line = (f'{datetime.datetime.now().isoformat()} SUBMIT #{attempt} ({len(configs)} configs) '
                f'{" ".join(map(shlex.quote, cmd))}')
        print(line, flush=True)
        log.write(line + '\n')
        log.flush()
        result = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        log.write(f'FINISHED rc={result.returncode}: nodes={nodes}\n')
    return result.returncode


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'check':
        # Used by the runner to validate a result, including legacy chunks.
        path, suite, nodes, b, base, it, inc, ppn = sys.argv[2:]
        row = dict(zip(FIELDS[:6], (suite, nodes, b, base, it, inc)))
        return 0 if complete(path, row, int(ppn)) else 1
    try:
        return submit(sys.argv[1:])
    except (OSError, ValueError, csv.Error) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
