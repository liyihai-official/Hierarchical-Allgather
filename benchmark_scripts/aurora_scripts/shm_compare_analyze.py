#!/usr/bin/env python3
"""Summarize the MPI_Allgather / radix_batch / node_shm comparison.

    shm_compare_analyze.py results/Aurora/shm_compare/<run> [--metric time_max|time_barrier]

Every number is the median over the repetitions of one configuration. For radix_batch, each
message size shows its best radix: that is the best radix_batch can do if k is tuned per size.
Writes summary.csv into the run folder and prints one table per node count, ratios below 1
meaning node_shm is faster.
"""
import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

STAGES = ("s1_max", "s2_max", "s3_max")


def load(folder):
    rows = defaultdict(list)
    files = [f for f in sorted(glob.glob(os.path.join(folder, "*.csv")))
             if "_sanity_" not in f and os.path.basename(f) != "summary.csv"]
    for path in files:
        with open(path, newline="") as stream:
            for row in csv.DictReader(stream):
                key = (int(row["nodes"]), int(row["send_count"]), row["algorithm"], int(row["k"]))
                rows[key].append(row)
    return rows, files


def summarize(rows, metric):
    summary = {}
    for key, group in rows.items():
        summary[key] = {
            "time": st.median(float(r[metric]) for r in group),
            "reps": len(group),
            "wrong": sum(1 for r in group if r["is_correct"] != "1"),
            **{s: st.median(float(r[s]) for r in group) for s in STAGES},
        }
    return summary


def fmt(seconds):
    if seconds is None:
        return "-"
    if seconds < 1e-3:
        return "%.1f us" % (seconds * 1e6)
    if seconds < 1:
        return "%.2f ms" % (seconds * 1e3)
    return "%.3f s" % seconds


def ratio(numerator, denominator):
    return "%.2f" % (numerator / denominator) if numerator and denominator else "-"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--metric", default="time_max", choices=("time_max", "time_barrier"))
    args = parser.parse_args()

    rows, files = load(args.folder)
    if not rows:
        raise SystemExit("no result files in " + args.folder)
    summary = summarize(rows, args.metric)
    print("files: %d, metric: %s, wrong results: %d of %d repetitions" % (
        len(files), args.metric,
        sum(v["wrong"] for v in summary.values()), sum(v["reps"] for v in summary.values())))

    with open(os.path.join(args.folder, "summary.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["nodes", "send_count", "algorithm", "k", "reps", "wrong", args.metric, *STAGES])
        for key in sorted(summary):
            value = summary[key]
            writer.writerow([*key, value["reps"], value["wrong"], value["time"], *(value[s] for s in STAGES)])

    for nodes in sorted({key[0] for key in summary}):
        print("\n## %d nodes" % nodes)
        print("| elements | bytes/rank | MPI_Allgather | radix_batch best (k) | node_shm | shm/MPI | shm/radix |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        stage_lines = []
        for count in sorted({key[1] for key in summary if key[0] == nodes}):
            mpi = summary.get((nodes, count, "mpi_allgather", 0))
            shm = summary.get((nodes, count, "node_shm", 0))
            radix = {key[3]: value for key, value in summary.items()
                     if key[:3] == (nodes, count, "radix_batch")}
            best_k = min(radix, key=lambda k: radix[k]["time"]) if radix else None
            best = radix[best_k] if radix else None
            mpi_t, shm_t, best_t = (x["time"] if x else None for x in (mpi, shm, best))
            print("| %d | %d | %s | %s (k=%s) | %s | %s | %s |" % (
                count, count * 4, fmt(mpi_t), fmt(best_t), best_k, fmt(shm_t),
                ratio(shm_t, mpi_t), ratio(shm_t, best_t)))
            if best and shm:
                stage_lines.append("| %d | %s | %s | %s | %s | %s | %s |" % (
                    count, *(fmt(best[s]) for s in STAGES), *(fmt(shm[s]) for s in STAGES)))
        if stage_lines:
            print("\nSlowest rank in each stage (radix_batch at its best k, then node_shm):")
            print("| elements | radix s1 | radix s2 | radix s3 | shm s1 | shm s2 | shm s3 |")
            print("|---:|---:|---:|---:|---:|---:|---:|")
            print("\n".join(stage_lines))


if __name__ == "__main__":
    main()
