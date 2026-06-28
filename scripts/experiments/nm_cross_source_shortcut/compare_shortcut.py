#!/usr/bin/env python3
"""Compare NM test AUC across training regimes to test the cross-source-shortcut.

Pulls `test_roc_auc` from the eval log dirs (eval_<model>_to_<dataset>_nm_0shot_*)
for four models and prints a covid/ukr table plus the key verdict:

  single-source        (nm_matrix_ukr / nm_matrix_covid)   -- reference ceiling
  merged proportional  (nm_matrix_merged)                  -- baseline (shortcut on)
  merged within-source (nm_xsrc_within_source)             -- shortcut removed

If within-source >> proportional and approaches single-source on the cross-domain
cells, the cross-source-shortcut hypothesis is supported.

Stdlib only. Reuses eval logs already produced by the nm_transfer_matrix experiment
for the first three models, so only the within-source model needs fresh eval.

    python compare_shortcut.py --log-root /dataMeR1/phil/gfm/prodigy/log
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# model run-name -> display label
MODELS = {
    "nm_matrix_ukr": "single ukr",
    "nm_matrix_covid": "single covid",
    "nm_matrix_merged": "merged proportional",
    "nm_xsrc_within_source": "merged within-source",
}
DATASETS = {"ukr_rus_twitter": "test:ukr", "covid19_twitter": "test:covid"}
RUN_RE = re.compile(r"^eval_(?P<model>.+?)_to_(?P<dataset>.+?)_nm_(?P<shots>\d+)shot(?:_.*)?$")


def step_of(p: Path) -> int:
    m = re.search(r"_step(\d+)\.json$", p.name)
    return int(m.group(1)) if m else -1


def latest_auc(run_dir: Path):
    for p in sorted((run_dir / "data").glob("metrics_test*.json"), key=step_of, reverse=True):
        try:
            v = json.loads(p.read_text()).get("test_roc_auc")
        except (OSError, json.JSONDecodeError):
            continue
        if v is not None:
            return float(v)
    return None


def collect(log_root: Path):
    cells = {}
    for run_dir in sorted(log_root.glob("eval_*_to_*_nm_*shot*")):
        if not run_dir.is_dir():
            continue
        m = RUN_RE.match(run_dir.name)
        if not m or m["model"] not in MODELS or m["dataset"] not in DATASETS:
            continue
        auc = latest_auc(run_dir)
        if auc is not None:
            cells[(m["model"], m["dataset"])] = auc
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-root", default="log")
    args = ap.parse_args()

    cells = collect(Path(args.log_root))
    if not cells:
        raise SystemExit(f"No matching eval dirs under {args.log_root}")

    dcols = list(DATASETS)
    print(f"{'regime':<24}" + "".join(f"{DATASETS[d]:>14}" for d in dcols))
    print("-" * (24 + 14 * len(dcols)))
    for model, label in MODELS.items():
        row = f"{label:<24}"
        for d in dcols:
            v = cells.get((model, d))
            row += f"{v:>14.4f}" if v is not None else f"{'-':>14}"
        print(row)

    # Verdict: on each TEST domain, does within-source beat proportional and
    # approach the single-source ceiling? (cross-domain is the telling case)
    print()
    pairs = [("covid19_twitter", "nm_matrix_ukr", "test covid (single=ukr-trained)"),
             ("ukr_rus_twitter", "nm_matrix_covid", "test ukr (single=covid-trained)")]
    for dset, single_model, desc in pairs:
        prop = cells.get(("nm_matrix_merged", dset))
        within = cells.get(("nm_xsrc_within_source", dset))
        single = cells.get((single_model, dset))
        if prop is None or within is None:
            continue
        msg = f"{desc}: proportional={prop:.4f}  within-source={within:.4f}  (Δ={within - prop:+.4f})"
        if single is not None:
            msg += f"  single-ceiling={single:.4f}"
        verdict = "shortcut SUPPORTED" if within > prop else "no improvement"
        print(f"{msg} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
