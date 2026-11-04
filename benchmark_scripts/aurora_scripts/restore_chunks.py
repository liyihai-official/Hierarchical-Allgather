#!/usr/bin/env python3
"""Rebuild missing result chunks from run folders, so finished work is recognised as DONE
and merged again.

Every job keeps its own copy in <suite dir>/runs/<job>_n<N>_b<B>_base<BASE>_it<IT>[_inc<K>]/
results<PPN>_<N>_<B>.csv. submit_plan.py and merge_results.sh only look at the chunk files
<suite dir>/results<PPN>_<N>_<B>__base<BASE>_it<IT>_inc<K>_<job>.csv. If those chunk files
were moved or deleted but runs/ is still there, this script copies each complete, successful
run back to its chunk name. Existing chunk files are never overwritten.

usage (from the project root):
    python3 restore_chunks.py results/Aurora/main            # copy
    DRY=1 python3 restore_chunks.py results/Aurora/main      # only show what it would do
"""
import glob, os, re, shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from submit_plan import complete      # same check that decides DONE

INC = {8: 1, 13: 1, 26: 2, 52: 5, 104: 3}     # radix_increment of the main suite, by b
PPN = int(os.environ.get('PPN', '104'))
suite_dir = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/Aurora/main')
suite = suite_dir.name
dry = os.environ.get('DRY') == '1'
copied = present = bad = 0
for run in sorted(suite_dir.glob('runs/*/')):
    m = re.match(r'(\d+)_n(\d+)_b(\d+)_base(\d+)_it(\d+)(?:_inc(\d+))?$', run.name)
    if not m:
        continue
    job, n, b, base, it, inc = m.groups()
    meta = run / f'results{PPN}_{n}_{b}.meta'
    text = meta.read_text() if meta.exists() else ''
    if not inc:
        k = re.search(r'radix_increment=(\d+)', text)
        inc = k.group(1) if k else str(INC.get(int(b), 1))
    src = run / f'results{PPN}_{n}_{b}.csv'
    row = dict(suite=suite, nodes=n, b=b, base=base, n_iter=it, radix_increment=inc)
    if not re.search(r'^exit code: 0\s*$', text, re.M) or not src.exists() or not complete(src, row, PPN):
        bad += 1
        print(f'skip (failed or incomplete): {run.name}')
        continue
    stem = f'results{PPN}_{n}_{b}__base{base}_it{it}_inc{inc}'
    if any(complete(p, row, PPN) for p in suite_dir.glob(f'{stem}_*.csv')):
        present += 1
        continue
    dst = suite_dir / f'{stem}_{job}.csv'
    print(f"{'would copy' if dry else 'copy'}: {run.name} -> {dst.name}")
    if not dry:
        shutil.copy2(src, dst)
    copied += 1
print(f'{copied} restored, {present} already present, {bad} skipped')
