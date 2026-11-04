#!/usr/bin/env python3
"""Compare your merged results with the CHiArA author's CSVs (Polaris or Aurora).

usage: python3 compare_with_author.py <your merged dir> <author results dir>
  e.g. python3 compare_with_author.py results/Polaris/main/merged  <repo>/.../Tests/Polaris
       python3 compare_with_author.py results/Polaris/batch/merged <repo>/.../Tests/Polaris/batch_comparisons

Files are matched by content (nprocs, b), not by name, because the author's older
Polaris files are named results<nodes>.csv instead of results<ppn>_<nodes>_<b>.csv.
Only the files directly inside the author dir are used (not subfolders).

For every matched pair it prints row counts, failed correctness checks, and per
send_count the median time of MPI_Allgather (standard) and of the fastest k of
allgather_radix_batch, yours vs author, with the ratio yours/author.
The first repetition of each (alg, k, size) is dropped as warm-up.
"""
import csv, collections, os, statistics, sys

H = ["alg", "k", "b", "np", "sc", "t", "ok"]


def load(path):
    reps = collections.defaultdict(list)
    rows = bad = 0; key = None
    for r in csv.reader(open(path)):
        if len(r) != 7 or r[0] == "algorithm_name":
            continue
        r = dict(zip(H, r)); rows += 1
        if r["ok"] != "1":
            bad += 1
        if r["alg"] != "allgather_standard" and key is None:
            key = (int(r["np"]), int(r["b"]))
        reps[(r["alg"], int(r["k"]), int(r["sc"]))].append(float(r["t"]))
    med = {k: statistics.median(v[1:] if len(v) > 1 else v) for k, v in reps.items()}
    return key, rows, bad, med


def summary(med):
    out = {}
    for sc in sorted({k[2] for k in med}):
        std = med.get(("allgather_standard", 0, sc))
        rb = [(t, k) for (a, k, s), t in med.items() if a == "allgather_radix_batch" and s == sc]
        best = min(rb) if rb else (None, None)
        out[sc] = (std, best[0], best[1])
    return out


def us(x):
    return f"{x * 1e6:9.1f}" if x is not None else "        -"


def ratio(a, b):
    return f"{a / b:5.2f}" if a and b else "    -"


mine_dir, author_dir = sys.argv[1], sys.argv[2]
author = {}
for name in sorted(os.listdir(author_dir)):
    p = os.path.join(author_dir, name)
    if name.endswith(".csv") and os.path.isfile(p):
        key, rows, bad, med = load(p)
        if key:
            author[key] = (name, rows, bad, med)

for name in sorted(os.listdir(mine_dir)):
    if not name.endswith(".csv"):
        continue
    key, mrows, mbad, mmed = load(os.path.join(mine_dir, name))
    if key not in author:
        print(f"\n### {name} (np={key[0] if key else '?'}, b={key[1] if key else '?'}): no author file with this np and b")
        continue
    aname, arows, abad, amed = author[key]
    print(f"\n### {name}  vs author {aname}   rows: yours {mrows} / author {arows}"
          f"   failed checks: yours {mbad} / author {abad}")
    ms, as_ = summary(mmed), summary(amed)
    print("   send_count | standard us: yours  author  ratio | best radix_batch us: yours (k)  author (k)  ratio")
    for sc in sorted(set(ms) | set(as_)):
        mstd, mrb, mk = ms.get(sc, (None, None, None))
        astd, arb, ak = as_.get(sc, (None, None, None))
        print(f"   {sc:10d} | {us(mstd)} {us(astd)} {ratio(mstd, astd)} |"
              f" {us(mrb)} ({mk if mk is not None else '-':>3}) {us(arb)} ({ak if ak is not None else '-':>3}) {ratio(mrb, arb)}")
