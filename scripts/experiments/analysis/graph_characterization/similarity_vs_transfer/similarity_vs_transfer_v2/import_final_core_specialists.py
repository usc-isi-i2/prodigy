#!/usr/bin/env python3
"""Import the final-core three-seed specialist test summary into v2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
DIVERGENCE = ROOT / "scripts/experiments/analysis/graph_characterization/statistics/graph_divergence/data/graph_divergence_data.json"
TRANSFER = ROOT / "scripts/experiments/analysis/transfer/matrices/prodigy_nm/single_source/nm_single_source_matrix_facebook/data/nm_single_source_matrix_9x9_long.csv"
OUT = Path(__file__).resolve().parent / "data/final_core"
NAME_MAP = {
    "ss_ukr_rus": "ukr_rus_twitter", "ss_covid": "covid19_twitter",
    "ss_midterm": "midterm", "ss_covid_political": "covid_political",
    "ss_election2020": "election2020", "ss_ukr_rus_suspended": "ukr_rus_suspended",
    "ss_twibot20": "twibot20", "ss_cp_hk": "cp_hk_twitter",
    "ss_facebook_page_reference": "facebook_page_reference",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=OUT)
    args = p.parse_args()
    summary = pd.read_csv(args.summary, sep="\t")
    specialists = summary[summary.model_id.isin(NAME_MAP)].copy()
    specialists["graph"] = specialists.model_id.map(NAME_MAP)
    if len(specialists) != 9 or specialists.n_seeds.ne(3).any():
        raise ValueError("expected nine three-seed final-core specialists")

    divergence = json.loads(DIVERGENCE.read_text())
    transfer = pd.read_csv(TRANSFER)
    foreign = transfer[(transfer.metric == "roc_auc") & (transfer.train != transfer.test)]
    old_outflow = foreign.groupby("train").value.mean()
    specialists["old_mean_foreign_auc"] = specialists.graph.map(old_outflow)
    specialists = specialists.sort_values("test_score_mean", ascending=False)

    score = specialists.set_index("graph").test_score_mean.reindex(divergence["graphs"])
    rows = [{"predictor": "old_mean_foreign_auc",
             "spearman_with_final_core_test_score": float(spearmanr(
                 [old_outflow[g] for g in divergence["graphs"]], score).statistic)}]
    keys = set.intersection(*[set(divergence["per_graph"][g]) for g in divergence["graphs"]])
    for key in sorted(keys):
        values = [divergence["per_graph"][g].get(key) for g in divergence["graphs"]]
        if not all(isinstance(value, (int, float)) and value is not None for value in values):
            continue
        rows.append({"predictor": key,
                     "spearman_with_final_core_test_score": float(spearmanr(values, score).statistic)})
    correlations = pd.DataFrame(rows)
    correlations["absolute_spearman"] = correlations.spearman_with_final_core_test_score.abs()
    correlations = correlations.sort_values("absolute_spearman", ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    columns = ["model_id", "graph", "n_seeds", "test_score_mean", "test_score_sample_std",
               "selected_checkpoint_steps", "old_mean_foreign_auc"]
    specialists[columns].to_csv(args.out_dir / "specialist_test_summary.csv", index=False)
    correlations.to_csv(args.out_dir / "specialist_source_correlations.csv", index=False)
    provenance = {
        "source": str(args.summary),
        "protocol": "final-core: 2500 updates; validation-selected checkpoint; one 500-episode fixed static_test evaluation per seed",
        "important": "test_score is episodic NM score/accuracy, not ROC-AUC",
        "old_outflow_comparison": "mean off-diagonal ROC-AUC from the historical 9x9 transfer matrix",
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
