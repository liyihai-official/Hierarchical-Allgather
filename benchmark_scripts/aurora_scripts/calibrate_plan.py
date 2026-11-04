#!/usr/bin/env python3
"""Re-estimate est_min of every plan row from YOUR OWN Aurora results.

Model, per send_count s of a row (nodes n, batch b, radix_increment inc, nk = #k values):
    timed(n,b,s)  your summed allgather time for 50 reps of every k (+ MPI_Allgather),
                  measured if you already ran it, otherwise scaled from your nearest
                  smaller node count (linear in nodes) and largest size (linear in size),
                  times EXTRAP_PAD for the uncertainty
    traffic       untimed work in main.cpp: per rep every rank zero-fills recvbuf and
                  compares it with refbuf (3 * s*np*4 bytes), 104 ranks per node
est_min = sum over sizes [ timed + traffic / BW ] + LAUNCH

BW (memory bandwidth per node) and LAUNCH are fitted from the real job durations in
results/Aurora/<suite>/runs/*/*.meta (start = "date:" line, end = "elapsed_sec:" line or the
file's modification time). BW takes the most pessimistic run, so estimates stay on the safe
side. Until there are usable runs: BW_GBS (default 270 GB/s) and LAUNCH_MIN (default 2).
The defaults come from your Oct 16-17 debug-scaling jobs (one job at a time, so the gap between
two job starts bounds the first job's duration): 52 nodes b=104 16384 ran <= 26.2 min with
14.1 min timed -> BW >= 270 GB/s even with zero launch time. Copies whose modification times
were reset (cp without -p, download, zip on another machine) are detected and ignored.

usage (from the project root):
    python3 calibrate_plan.py plan_main.tsv                    # print, do not change the plan
    python3 calibrate_plan.py plan_main.tsv --write            # rewrite est_min/walltime/flags
    python3 calibrate_plan.py plan_main.tsv --data results_all.csv --data old_results_dir
submit_plan.py imports this file and re-estimates before every job automatically.
"""
import argparse, collections, csv, datetime, glob, math, os, re, statistics, sys

PPN = int(os.environ.get('PPN', '104'))
EXTRAP_PAD = float(os.environ.get('EXTRAP_PAD', '1.2'))
DEFAULT_BW = float(os.environ.get('BW_GBS', '270')) * 1e9
DEFAULT_LAUNCH = float(os.environ.get('LAUNCH_MIN', '2.0'))
FIT_MIN = 60 - int(os.environ.get('TIME_MARGIN_SEC', '180')) / 60
REPS = 50
CHUNK_RE = re.compile(r'results(\d+)_(\d+)_(\d+)__base\d+_it\d+_inc\d+_[^/]*\.csv$')


def nks(b, inc):
    return len(range(2, b, inc))


def traffic_bytes(nodes, b, inc, size):
    """Bytes per node moved by the untimed fill + correctness check of one (row, size)."""
    np_ = nodes * PPN
    per_rank = size * np_ * 4
    algs = nks(b, inc) + 1
    return algs * (REPS * 3 + 2) * per_rank * PPN


class Model:
    def __init__(self, sources):
        self.sum = collections.defaultdict(float)     # (nodes,b,size) -> seconds summed
        self.cnt = collections.Counter()              # (nodes,b,size) -> radix rows
        self.runs = []                                # (nodes,b,inc,[sizes],wall_min,timed_min)
        self.skipped = 0                              # runs whose file times are not real
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'timing_baseline.csv')
        if os.environ.get('CALIB_BASELINE', '1') != '0' and os.path.exists(base):
            self._baseline(base)
        for src in sources:
            if os.path.isdir(src):
                self._dir(src)
            elif os.path.isfile(src):
                self._merged(src)
        self.bw, self.launch, self.fit_note = self._fit()

    # ---- data ----
    def _add(self, nodes, alg, b, size, t):
        if alg == 'allgather_radix_batch':
            self.sum[nodes, b, size] += t
            self.cnt[nodes, b, size] += 1

    def _baseline(self, path):
        """Aggregated timings of earlier runs (timing_baseline.csv), used until new runs exist."""
        with open(path, newline='') as f:
            for r in csv.DictReader(line for line in f if not line.startswith('#')):
                k = (int(r['nodes']), int(r['b']), int(r['send_count']))
                self.sum[k] += float(r['sum_sec'])
                self.cnt[k] += int(r['count'])

    def _merged(self, path):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if 'nodes' not in r:
                    return
                try:
                    self._add(int(r['nodes']), r['algorithm_name'], int(r['b']), int(r['send_count']), float(r['time']))
                except (ValueError, KeyError):
                    pass

    def _chunk(self, path, nodes):
        with open(path, newline='') as f:
            for r in csv.reader(f):
                if len(r) == 7 and r[0] != 'algorithm_name':
                    try:
                        self._add(nodes, r[0], int(r[2]), int(r[4]), float(r[5]))
                    except ValueError:
                        pass

    def _dir(self, d):
        for p in glob.glob(os.path.join(d, 'results*__*.csv')):
            m = CHUNK_RE.search(p)
            if m and int(m.group(1)) == PPN:
                self._chunk(p, int(m.group(2)))
        for meta in glob.glob(os.path.join(d, 'runs', '*', '*.meta')):
            self._run(meta)

    def _run(self, meta):
        try:
            text = open(meta).read()
        except OSError:
            return
        if not re.search(r'^exit code: 0\s*$', text, re.M):
            return
        p = re.search(r'NNODES=(\d+) PPN=(\d+) .*?b=(\d+) base=(\d+) n_iter=(\d+) radix_increment=(\d+)', text)
        d = re.search(r'^date: (\S+)', text, re.M)
        if not p or not d or int(p.group(2)) != PPN:
            return
        nodes, _, b, base, it, inc = map(int, p.groups())
        e = re.search(r'^elapsed_sec: (\d+)', text, re.M)
        if e:
            wall = int(e.group(1)) / 60
        else:
            try:
                start = datetime.datetime.fromisoformat(d.group(1)).timestamp()
            except ValueError:
                return
            first = os.path.join(os.path.dirname(meta), 'hosts_first')   # written right after "date:"
            if os.path.exists(first):
                start = os.path.getmtime(first)
            wall = (os.path.getmtime(meta) - start) / 60
        csvs = glob.glob(os.path.join(os.path.dirname(meta), f'results{PPN}_{nodes}_{b}.csv'))
        if not csvs or wall <= 0:
            return
        timed = 0.0
        with open(csvs[0], newline='') as f:
            for r in csv.reader(f):
                if len(r) == 7 and r[0] != 'algorithm_name':
                    try:
                        timed += float(r[5])
                    except ValueError:
                        pass
        sizes = [base << i for i in range(it)]
        # copied files (all mtimes reset) give wall ~ 0 or days: not a real duration
        if wall < timed / 60 * 0.95 or wall > 24 * 60:
            self.skipped += 1
            return
        self.runs.append((nodes, b, inc, sizes, wall, timed / 60))

    # ---- fit ----
    def _fit(self):
        env_bw, env_l = 'BW_GBS' in os.environ, 'LAUNCH_MIN' in os.environ
        small, big = [], []
        for nodes, b, inc, sizes, wall, timed in self.runs:
            tb = sum(traffic_bytes(nodes, b, inc, s) for s in sizes)
            ref = tb / 300e9 / 60                       # minutes at 300 GB/s
            if ref < 0.1:
                small.append(wall - timed)
            elif ref >= 0.25:
                big.append((wall - timed, tb, nodes, b, sizes))
        launch = DEFAULT_LAUNCH
        if small and not env_l:
            res = sorted(small)
            launch = min(10.0, max(0.2, res[int(0.9 * (len(res) - 1))]))
        bw, note = DEFAULT_BW, 'default'
        # The fill + check is node-local, so BW per node does not depend on the node count.
        # All untimed time is charged to it (launch not subtracted): BW comes out low = safe side.
        implied = [(tb / (res * 60), n, b, s) for res, tb, n, b, s in big if res > 0]
        if implied:                                     # measured runs always win over BW_GBS
            worst = min(implied)
            bw = worst[0]
            note = (f'fitted from {len(implied)} run(s), slowest = {worst[1]} nodes b={worst[2]} '
                    f'sizes {worst[3][0]}..{worst[3][-1]}')
        elif env_bw:
            note = 'BW_GBS from environment, no usable timed runs yet'
        return bw, launch, note

    # ---- estimate ----
    def timed(self, nodes, b, size):
        """(minutes for 50 reps of every k, 'measured'|'extrap') or (None, None)."""
        def val(n, s):
            return self.sum[n, b, s] / self.cnt[n, b, s] * REPS / 60      # per k, x nk later
        if self.cnt[nodes, b, size]:
            return val(nodes, size), 'measured'
        have = sorted({n for (n, bb, s) in self.cnt if bb == b})
        if not have:
            return None, None
        with_size = [n for n in have if self.cnt[n, b, size]]
        lower = [n for n in with_size if n < nodes]
        if lower:
            m = max(lower)
            return val(m, size) * nodes / m * EXTRAP_PAD, 'extrap'
        lower = [n for n in have if n <= nodes]
        m = max(lower) if lower else min(have)
        sizes = sorted(s for (n, bb, s) in self.cnt if n == m and bb == b)
        top = max(s for s in sizes if s <= size) if any(s <= size for s in sizes) else min(sizes)
        return val(m, top) * size / top * nodes / m * EXTRAP_PAD, 'extrap'

    def estimate(self, row):
        n, b, base, it, inc = (int(row[k]) for k in ('nodes', 'b', 'base', 'n_iter', 'radix_increment'))
        total, how = self.launch, set()
        for i in range(it):
            s = base << i
            t, h = self.timed(n, b, s)
            if t is None:
                return None, 'author'
            total += t * (nks(b, inc) + 1)          # every k, plus MPI_Allgather ~ one more k
            total += traffic_bytes(n, b, inc, s) / self.bw / 60
            how.add(h)
        return total, 'extrap' if 'extrap' in how else 'measured'

    def apply(self, rows):
        """Update est_min / walltime / note of plan rows in place. Returns #changed."""
        changed = 0
        for r in rows:
            est, how = self.estimate(r)
            if est is None:
                continue
            tags = [t for t in r['note'].split(';') if t and t != '-' and not t.startswith('est_')
                    and t not in ('MAY_EXCEED_1H', 'needs_k_split', 'observed_lt_60min')]
            tags.append('est_' + how)
            if r['queue'] in ('debug', 'debug-scaling') and est > FIT_MIN:
                tags += ['MAY_EXCEED_1H', 'needs_k_split']
            m = max(15, math.ceil(est * 1.5 / 5) * 5)
            if r['queue'] in ('debug', 'debug-scaling'):
                m = min(m, 60)
            new = dict(est_min=f'{est:.1f}', walltime=f'{m // 60}:{m % 60:02d}:00', note=';'.join(tags))
            if any(r[k] != v for k, v in new.items()):
                changed += 1
            r.update(new)
        return changed

    def summary(self):
        return (f'calibration: BW={self.bw / 1e9:.0f} GB/s per node ({self.fit_note}), '
                f'launch={self.launch:.1f} min, {len(self.runs)} timed run(s)'
                + (f' ({self.skipped} ignored: copied files, times not real)' if self.skipped else '') + ', '
                f'{len(self.cnt)} measured (nodes,b,size) points')


def default_sources(root, suites):
    return [os.path.join(root, 'results', 'Aurora', s) for s in sorted(suites)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('plan')
    ap.add_argument('--data', action='append', default=[],
                    help='extra results dir (with runs/) or merged results_all.csv; repeatable')
    ap.add_argument('--write', action='store_true', help='rewrite the plan file')
    a = ap.parse_args()
    with open(a.plan, newline='') as f:
        rd = csv.DictReader(f, delimiter='\t')
        fields, rows = rd.fieldnames, list(rd)
    model = Model(default_sources(os.getcwd(), {r['suite'] for r in rows}) + a.data)
    print(model.summary())
    old = {id(r): r['est_min'] for r in rows}
    model.apply(rows)
    print(f"{'nodes':>5} {'b':>3} {'base':>6} {'it':>2}  {'old':>6} {'new':>6}  note")
    for r in rows:
        if int(r['nodes']) >= 26 or 'needs_k_split' in r['note']:
            print(f"{r['nodes']:>5} {r['b']:>3} {r['base']:>6} {r['n_iter']:>2}  {old[id(r)]:>6} {r['est_min']:>6}  {r['note']}")
    k = sum('needs_k_split' in r['note'] for r in rows)
    print(f'{k} row(s) still estimated above {FIT_MIN:.0f} min in a 1-hour queue (need a k-range split).')
    if a.write:
        with open(a.plan, 'w', newline='') as f:
            w = csv.DictWriter(f, fields, delimiter='\t', lineterminator='\n')
            w.writeheader(); w.writerows(rows)
        print(f'wrote {a.plan}')


if __name__ == '__main__':
    main()
