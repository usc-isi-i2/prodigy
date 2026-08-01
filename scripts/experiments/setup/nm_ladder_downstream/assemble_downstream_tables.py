#!/usr/bin/env python3
"""Assemble the ladder's downstream results into the deliverable tables.

Reads the three result sources produced by the sweep and expands the 21 distinct
encoders back into the 24 (order, rung) rows through ``row_map.csv``:

  node regression      analysis/nm_ladder_downstream/data/reg_probe/<dataset>__reg_probe.csv
  node classification  analysis/node_classification/data/node_classification.csv
  static link pred     analysis/nm_ladder_downstream/data/pair_lp/<dataset>__pair_lp.csv

Regression comes from the frozen-encoder probe, NOT from the shared
``node_regression.csv``. Those runner rows are void: the episodic ``task_name=regression``
path predicts through a ``regression_head`` that is in no checkpoint, loads with
``strict=False`` so it stays at random init, and ``--eval_only`` never takes an optimizer
step -- the reported number is a fixed random projection of the frozen embedding
(``setup/regression_probe_repair/README.md``). ``--reg-source runner`` still reads the old
path, for reproducing what the superseded 2026-07-27 figures showed; it is not a default
anyone should reach for.

Emits into ``analysis/nm_ladder_downstream/data/``:

  nm_ladder_downstream_long.csv   one row per (order, rung, task, dataset[, target]),
                                  carrying ``in_merge`` and ``rel_to_entry`` so the
                                  event-study figure can overlay the three orders on a
                                  common x-axis -- the same shape the NM table used.
  nm_ladder_downstream_<task>.csv wide: 24 rows x one column per test graph (per target
                                  for regression).

``rel_to_entry`` is the rung minus the rung at which that test graph joined the training
merge: < 0 is out-of-merge (zero-shot transfer), >= 0 is in-merge. It is defined for all
8 test graphs in every order, including the graphs a rung has not reached yet.

Static-LP rows are gated on the evaluator's own validity reads (leakage_edges == 0,
endpoint_sensitivity ~ 1, endpoint_permutation_auc ~ 0.5); a failing cell is dropped and
reported rather than averaged into a headline.

Usage:
    python3 assemble_downstream_tables.py
    python3 assemble_downstream_tables.py --allow-partial     # tolerate missing cells
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[3]
ANALYSIS = REPO_ROOT / "scripts" / "experiments" / "analysis"
sys.path.insert(0, str(HERE.parent / "nm_ladder_order_robustness-jul_23"))

from make_configs import ORDERS, SOURCES  # noqa: E402

DATASET_OF_KEY = {s[0]: s[1] for s in SOURCES}
KEY_OF_DATASET = {s[1]: s[0] for s in SOURCES}
CANON = [s[1] for s in SOURCES]          # dataset_key, in table-column order

# Primary metric per task. The others are carried in the long form for the notebook.
PRIMARY = {"reg": "spearman", "pl": "roc_auc", "slp": "auc"}
REG_METRICS = ("spearman", "r2", "rmse", "mae")
PL_METRICS = ("roc_auc", "accuracy", "f1")

SLP_NEGATIVE_KIND = "degree_matched"     # headline condition (see FINDINGS_rescore.md)
SLP_SCORER = "encoder_cosine"
HEURISTIC_SCORERS = ("common_neighbors", "adamic_adar",
                     "preferential_attachment", "jaccard", "raw_feature_cosine")

_TS_RE = re.compile(r"_(\d{2})_(\d{2})_(\d{4})_(\d{2})_(\d{2})_(\d{2})$")


def run_timestamp(run: str) -> tuple:
    """Sortable key from a log dir's trailing _DD_MM_YYYY_HH_MM_SS, so that when a
    config was evaluated more than once the newest row wins."""
    m = _TS_RE.search(run or "")
    if not m:
        return (0,) * 6
    dd, mm, yyyy, hh, mi, ss = (int(g) for g in m.groups())
    return (yyyy, mm, dd, hh, mi, ss)


def read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def latest_by(rows, key_fields):
    """Collapse duplicate evaluations of the same config, keeping the newest run."""
    best: dict[tuple, dict] = {}
    for row in rows:
        k = tuple(row.get(f, "") for f in key_fields)
        prev = best.get(k)
        if prev is None or run_timestamp(row.get("run", "")) >= run_timestamp(prev.get("run", "")):
            best[k] = row
    return best


def load_runner_task(csv_path: Path, models: set[str], shots: str, metrics):
    """(model, dataset, target) -> {metric: value} from a shared per-task CSV."""
    rows = [r for r in read_rows(csv_path)
            if r.get("model") in models and r.get("shots") == shots
            and r.get("split", "test") == "test"]
    best = latest_by(rows, ("model", "dataset", "target"))
    out = {}
    for (model, dataset, target), row in best.items():
        out[(model, dataset, target)] = {m: to_float(row.get(m)) for m in metrics}
    return out


def load_reg_probe(probe_dir: Path, models: set[str], shots: str):
    """(model, dataset, target) -> {metric: value} from the frozen-encoder probe,
    plus the per-(dataset, target) raw-feature floor.

    The sweep writes one CSV per dataset and one row per (model, target, alpha). There
    is no ``run`` column and no re-evaluation to collapse -- every arm was scored in a
    single pass against one shared episode set -- so ``latest_by`` is not needed here.
    A repeated (model, dataset, target) at more than one alpha is a caller error rather
    than a duplicate to resolve, and is reported instead of silently collapsed.
    """
    cells: dict[tuple, dict] = {}
    floors: dict[tuple, float] = {}
    dupes: list[str] = []
    for path in sorted(probe_dir.glob("*__reg_probe.csv")):
        for row in read_rows(path):
            if row.get("shots") != shots:
                continue
            dataset, model, target = row.get("dataset"), row.get("model"), row.get("target")
            rho = to_float(row.get("spearman"))
            if rho is None:
                continue
            if model == "__features_only__":
                floors[(dataset, target)] = rho
                continue
            if model not in models:
                continue
            key = (model, dataset, target)
            if key in cells:
                dupes.append(f"{model}@{dataset}/{target} (alpha={row.get('alpha')})")
                continue
            cells[key] = {m: to_float(row.get(m)) for m in REG_METRICS}
    if dupes:
        print(f"WARNING: {len(dupes)} duplicate probe cells ignored (keeping the first "
              f"-- pass a single --alpha to the sweep):", file=sys.stderr)
        for line in dupes[:10]:
            print(f"  {line}", file=sys.stderr)
    return cells, floors


def load_pair_lp(pair_dir: Path, models: set[str]):
    """(model, dataset) -> {metric: value}, plus per-dataset heuristic floors.

    Drops any model cell failing the evaluator's validity reads.
    """
    cells: dict[tuple, dict] = {}
    floors: dict[str, dict] = {}
    invalid: list[str] = []
    for path in sorted(pair_dir.glob("*__pair_lp.csv")):
        for row in read_rows(path):
            if row.get("negative_kind") != SLP_NEGATIVE_KIND:
                continue
            dataset, model, scorer = row.get("dataset"), row.get("model"), row.get("scorer")
            auc = to_float(row.get("auc"))
            if auc is None:
                continue
            if model == "__floor__":
                if scorer in HEURISTIC_SCORERS:
                    floors.setdefault(dataset, {})[scorer] = auc
                continue
            if model not in models or scorer != SLP_SCORER:
                continue
            leak = to_float(row.get("leakage_edges")) or 0.0
            sens = to_float(row.get("endpoint_sensitivity"))
            perm = to_float(row.get("endpoint_permutation_auc"))
            if leak > 0 or (sens is not None and sens < 0.99) \
                    or (perm is not None and abs(perm - 0.5) > 0.05):
                invalid.append(f"{model}@{dataset} (leak={leak}, sens={sens}, perm={perm})")
                continue
            cells[(model, dataset)] = {
                "auc": auc,
                "average_precision": to_float(row.get("average_precision")),
                "hits_at_50": to_float(row.get("hits_at_50")),
            }
    return cells, floors, invalid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-map", default=str(HERE / "row_map.csv"))
    ap.add_argument("--out-dir", default=str(ANALYSIS / "nm_ladder_downstream" / "data"))
    ap.add_argument("--pair-lp-dir",
                    default=str(ANALYSIS / "nm_ladder_downstream" / "data" / "pair_lp"))
    ap.add_argument("--reg-source", choices=["probe", "runner"], default="probe",
                    help="probe = the repaired frozen-encoder probe (default); "
                         "runner = the void episodic path, for reproducing the "
                         "superseded 2026-07-27 figures only.")
    ap.add_argument("--reg-probe-dir",
                    default=str(ANALYSIS / "nm_ladder_downstream" / "data" / "reg_probe"))
    ap.add_argument("--reg-csv",
                    default=str(ANALYSIS / "node_regression" / "data" / "node_regression.csv"))
    ap.add_argument("--pl-csv",
                    default=str(ANALYSIS / "node_classification" / "data" / "node_classification.csv"))
    ap.add_argument("--shots", default="10")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    rows = read_rows(Path(args.row_map))
    if len(rows) != 24:
        print(f"ERROR: row_map has {len(rows)} rows, expected 24 "
              f"(run make_model_list.py)", file=sys.stderr)
        return 2
    models = {r["model_key"] for r in rows}

    reg_floors: dict[tuple, float] = {}
    if args.reg_source == "probe":
        probe_dir = Path(args.reg_probe_dir)
        if not probe_dir.is_dir():
            print(f"ERROR: no probe results at {probe_dir}. Run "
                  f"run_reg_probe_sweep.sh, or pass --reg-source runner to rebuild "
                  f"the superseded figures from the void path.", file=sys.stderr)
            return 2
        reg, reg_floors = load_reg_probe(probe_dir, models, args.shots)
    else:
        print("WARNING: --reg-source runner reads the VOID episodic regression path "
              "(random projection of the frozen embedding). Results are not a measure "
              "of representation quality.", file=sys.stderr)
        reg = load_runner_task(Path(args.reg_csv), models, args.shots, REG_METRICS)
    pl = load_runner_task(Path(args.pl_csv), models, args.shots, PL_METRICS)
    slp, floors, invalid = load_pair_lp(Path(args.pair_lp_dir), models)

    reg_targets = sorted({t for (_, _, t) in reg})
    # Which (dataset[, target]) combinations each task actually covers. A cell is only
    # "missing" if the task covers that graph but this model has no row for it -- the
    # 8 test graphs are not all eligible for every task.
    reg_covered = {(d, t) for (_, d, t) in reg}
    pl_covered = {d for (_, d, _) in pl}
    slp_covered = {d for (_, d) in slp} | set(floors)
    print(f"loaded: reg={len(reg)} cells over targets {reg_targets}; "
          f"pl={len(pl)} cells; slp={len(slp)} cells")
    if invalid:
        print(f"WARNING: {len(invalid)} static-LP cells failed the validity reads "
              f"and were dropped:", file=sys.stderr)
        for line in invalid:
            print(f"  {line}", file=sys.stderr)

    long_rows = []
    missing = []
    for row in rows:
        order, rung = row["order"], int(row["rung"])
        model = row["model_key"]
        seq = ORDERS[order]
        for dataset in CANON:
            entry_rung = seq.index(KEY_OF_DATASET[dataset]) + 1
            base = dict(
                order=order, rung=rung, added=row["added"], n_sources=row["n_sources"],
                sources=row["sources"], model_key=model, dataset=dataset,
                entry_rung=entry_rung, rel_to_entry=rung - entry_rung,
                in_merge=int(rung >= entry_rung),
            )

            for target in reg_targets:
                if (dataset, target) not in reg_covered:
                    continue
                vals = reg.get((model, dataset, target))
                if vals is None:
                    missing.append(f"reg {model}/{dataset}/{target}")
                    continue
                for metric, value in vals.items():
                    if value is not None:
                        long_rows.append({**base, "task": "reg", "target": target,
                                          "metric": metric, "value": value,
                                          "primary": int(metric == PRIMARY["reg"])})

            if dataset in pl_covered:
                vals = pl.get((model, dataset, ""))
                if vals is None:
                    missing.append(f"pl {model}/{dataset}")
                for metric, value in (vals or {}).items():
                    if value is not None:
                        long_rows.append({**base, "task": "pl", "target": "",
                                          "metric": metric, "value": value,
                                          "primary": int(metric == PRIMARY["pl"])})

            if dataset in slp_covered:
                vals = slp.get((model, dataset))
                if vals is None:
                    missing.append(f"slp {model}/{dataset}")
                for metric, value in (vals or {}).items():
                    if value is not None:
                        long_rows.append({**base, "task": "slp", "target": "",
                                          "metric": metric, "value": value,
                                          "primary": int(metric == PRIMARY["slp"])})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = ["order", "rung", "added", "n_sources", "sources", "model_key", "task",
              "dataset", "target", "metric", "value", "primary", "entry_rung",
              "rel_to_entry", "in_merge"]
    long_path = out_dir / "nm_ladder_downstream_long.csv"
    with long_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in sorted(long_rows, key=lambda r: (r["task"], r["order"], r["rung"],
                                                  r["dataset"], r["target"], r["metric"])):
            w.writerow(r)
    print(f"wrote {long_path} ({len(long_rows)} cells)")

    # Wide per-task tables on the primary metric: 24 rows x test-graph columns.
    by_task = defaultdict(dict)
    for r in long_rows:
        if r["primary"]:
            col = f"{r['dataset']}__{r['target']}" if r["target"] else r["dataset"]
            by_task[r["task"]][(r["order"], r["rung"], col)] = r["value"]

    for task, cells in by_task.items():
        cols = sorted({c for (_, _, c) in cells})
        path = out_dir / f"nm_ladder_downstream_{task}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["order", "rung", "added", "n_sources", "model_key", *cols])
            for row in rows:
                order, rung = row["order"], int(row["rung"])
                w.writerow([order, rung, row["added"], row["n_sources"], row["model_key"],
                            *[cells.get((order, rung, c), "") for c in cols]])
        print(f"wrote {path} (24 rows x {len(cols)} cols, metric={PRIMARY[task]})")

    if floors:
        path = out_dir / "nm_ladder_downstream_slp_floors.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", *HEURISTIC_SCORERS])
            for dataset in CANON:
                if dataset in floors:
                    w.writerow([dataset, *[floors[dataset].get(s, "")
                                           for s in HEURISTIC_SCORERS]])
        print(f"wrote {path} ({len(floors)} datasets)")

    if reg_floors:
        # The raw-feature floor on the SAME episodes -- the regression analogue of the
        # static-LP heuristic floors, and the line an encoder has to clear to be
        # carrying anything the 768-d input features did not already carry.
        path = out_dir / "nm_ladder_downstream_reg_floors.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["dataset", "target", "features_only_spearman"])
            for (dataset, target) in sorted(reg_floors):
                w.writerow([dataset, target, reg_floors[(dataset, target)]])
        print(f"wrote {path} ({len(reg_floors)} cells)")

    if missing:
        print(f"\n{len(missing)} missing cells (first 20):", file=sys.stderr)
        for line in missing[:20]:
            print(f"  {line}", file=sys.stderr)
        if not args.allow_partial:
            print("re-run with --allow-partial to write anyway", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
