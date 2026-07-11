#!/usr/bin/env python3
"""Build the partial-cross-source sweep table from eval logs.

Reads NM eval dirs (eval_<model>_to_<dataset>_nm_<shots>shot_<nway>way*) produced by
eval_ckpts_all_graph_tasks_tucker.py, pulls test_{metric} from data/metrics_test*.json,
and prints a table ordered by p = neighbor_sampling_cross_source_prob:

    p=0.00 (within) ... p=1.00 (naive)   x   {test:ukr, test:covid}

Verdict per (domain, metric): is the best p an ENDPOINT (within=0 or naive=1) or an
INTERIOR value? An interior optimum on cross-domain transfer supports remedy #4 (a
little cross-source signal beats both pure within-source and pure naive). Stdlib only.

    python build_sweep.py --log-root /dataMeR1/phil/gfm/prodigy/log --metric all --out-csv sweep.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# run-name -> (display label, p value), in sweep order.
MODELS = {
    "nm_pxsrc_p000": ("p=0.00 (within)", 0.00),
    "nm_pxsrc_p010": ("p=0.10", 0.10),
    "nm_pxsrc_p025": ("p=0.25", 0.25),
    "nm_pxsrc_p050": ("p=0.50", 0.50),
    "nm_pxsrc_p100": ("p=1.00 (naive)", 1.00),
}
DATASETS = {"ukr_rus_twitter": "test:ukr", "covid19_twitter": "test:covid"}
RUN_RE = re.compile(r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_nm_(?P<shots>\d+)shot(?:_.*)?$")


def step_of(p: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", p.name)
    return int(m.group(1)) if m else -1


def latest_metric(run_dir: Path, metric: str):
    key = f"test_{metric}"
    for p in sorted((run_dir / "data").glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            v = json.loads(p.read_text()).get(key)
        except (OSError, json.JSONDecodeError):
            continue
        if v is not None:
            return float(v)
    return None


def collect(log_root: Path, shots: str, nway: str, metric: str):
    cells = {}
    for run_dir in sorted(log_root.glob(f"eval_*_to_*_nm_{shots}shot_{nway}way*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m or m["model"] not in MODELS or m["dataset"] not in DATASETS:
            continue
        v = latest_metric(run_dir, metric)
        if v is not None:
            cells[(m["model"], m["dataset"])] = v
    return cells


def report(cells, metric: str, csv_rows, eps: float = 0.005) -> None:
    dcols = list(DATASETS)
    print(f"\n=== {metric} (rows = cross-source prob p) ===")
    print(f"{'p':<18}" + "".join(f"{DATASETS[d]:>14}" for d in dcols))
    print("-" * (18 + 14 * len(dcols)))
    for model, (label, _p) in MODELS.items():
        if not any((model, d) in cells for d in dcols):
            continue
        row = f"{label:<18}"
        for d in dcols:
            v = cells.get((model, d))
            row += f"{v:>14.4f}" if v is not None else f"{'-':>14}"
            if v is not None:
                csv_rows.append([metric, f"{_p:.2f}", DATASETS[d], f"{v:.6f}"])
        print(row)

    # Verdict per domain + a cross-domain conclusion. An interior p only counts as a real
    # win if it beats BOTH endpoints by more than `eps` (deltas below eps are noise at 1
    # seed, ±~0.09 episode std); and remedy #4 is only "supported" if the SAME interior p
    # wins on EVERY domain. (Guards against calling a 0.001 fluctuation an interior optimum.)
    interior_winner = {}  # domain -> winning interior p, else None
    for d in dcols:
        pts = [(p, cells.get((m, d))) for m, (lab, p) in MODELS.items()]
        pts = [(p, v) for p, v in pts if v is not None]
        if len(pts) < 2:
            continue
        best_p, best_v = max(pts, key=lambda t: t[1])
        vmap = dict(pts)
        v0, v1 = vmap.get(0.0), vmap.get(1.0)
        endpoints = [v for p, v in pts if p in (0.0, 1.0)]
        endpoint_best = max(endpoints) if endpoints else None
        extra = f"  within(p=0)={v0:.4f} naive(p=1)={v1:.4f}" if v0 is not None and v1 is not None else ""
        is_interior = best_p not in (0.0, 1.0)
        margin = (best_v - endpoint_best) if endpoint_best is not None else 0.0
        real_win = is_interior and endpoint_best is not None and margin > eps
        interior_winner[d] = best_p if real_win else None
        if real_win:
            verdict = f"interior p={best_p:.2f} beats best endpoint by {margin:+.4f} (> {eps})"
        elif is_interior:
            verdict = (f"argmax interior p={best_p:.2f} but only {margin:+.4f} over best endpoint "
                       f"(<= {eps}: noise) -> no interior optimum")
        else:
            verdict = "within-source best" if best_p == 0.0 else "naive best"
        print(f"  {DATASETS[d]}: argmax p={best_p:.2f} ({best_v:.4f}){extra} -> {verdict}")

    domains = [d for d in dcols if d in interior_winner]
    if domains:
        wins = {interior_winner[d] for d in domains}
        if len(wins) == 1 and None not in wins:
            print(f"  => remedy #4 SUPPORTED on {metric}: p={wins.pop():.2f} beats endpoints by > {eps} on all domains.")
        else:
            print(f"  => remedy #4 NOT supported on {metric}: no single interior p beats within-source by > {eps} across domains.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default="log")
    ap.add_argument("--shots", default="3")
    ap.add_argument("--n-way", default="30")
    ap.add_argument("--metric", default="all", choices=["roc_auc", "accuracy", "f1", "all"])
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--interior-eps", type=float, default=0.005,
                    help="Min margin for an interior p to count as beating an endpoint (below = noise at 1 seed).")
    args = ap.parse_args()

    metrics = ["accuracy", "f1", "roc_auc"] if args.metric == "all" else [args.metric]
    csv_rows, found = [], False
    for metric in metrics:
        cells = collect(Path(args.log_root), args.shots, args.n_way, metric)
        if not cells:
            print(f"[warn] no eval dirs with test_{metric}")
            continue
        found = True
        report(cells, metric, csv_rows, args.interior_eps)
    if not found:
        raise SystemExit(f"No matching eval dirs under {args.log_root}")
    if args.out_csv and csv_rows:
        import csv as _csv
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f); w.writerow(["metric", "p", "test", "value"]); w.writerows(csv_rows)
        print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
