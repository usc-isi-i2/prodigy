"""Validate the complete identity audit and link to existing policy results.

Exploratory: no target ranking, p-values, model selection, or causal estimates.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.target_performance_mechanisms.audit_consumed_conflicts import normalized


HERE = Path(__file__).resolve().parent
POLICIES = ("lowest_sorted", "lowest_shuffled", "uniform_sorted", "uniform_shuffled")
TARGETS = ("covid_political", "facebook_page_reference", "twibot20", "election2020", "ukr_rus_suspended")
EXPOSURES = ("cross_class_duplicate_fraction", "query_also_wrong_class_support_fraction",
    "query_multilabel_query_fraction", "support_multilabel_support_fraction",
    "identity_only_query_accuracy_upper_bound", "identity_only_query_ce_lower_bound")


def validate(root):
    done = json.loads((root / "DONE.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    records = json.loads((root / "records.json").read_text())
    models, windows = (pd.read_csv(root / f"{name}.csv") for name in ("models", "windows"))
    expected = {("member", s, seed, p) for s in ("ukr_rus", "cp_hk") for seed in range(3) for p in POLICIES}
    expected |= {("readout", s, seed, "lowest_sorted") for s in ("ukr_rus", "cp_hk", "covid") for seed in range(3)}
    cols = ["campaign", "source", "seed", "policy"]
    if len(models) != 33 or set(models[cols].itertuples(index=False, name=None)) != expected or models.model_id.duplicated().any():
        raise ValueError("incomplete model grid")
    if len(windows) != 165 or windows.duplicated(["model_id", "window"]).any():
        raise ValueError("incomplete window grid")
    if done["reference_summaries_verified"] != 33 or not done["complete_streams"] or not protocol["actual_complete_consumed_streams"]:
        raise ValueError("missing validity gate")
    if len(records) != 33 or {r["model_id"] for r in records} != set(models.model_id):
        raise ValueError("record inventory mismatch")
    raw = [c for c in models.columns if c in ("episodes", "member_positions", "support_positions", "query_positions", "duplicate_positions")
           or c.endswith("_positions") or c.endswith("_sum") or c.startswith("episodes_with_") or c == "identity_only_query_oracle_correct"]
    by_record = {r["model_id"]: r for r in records}
    for _, m in models.iterrows():
        w = windows[windows.model_id == m.model_id]
        if set(w.window) != set(range(5)) or set(w.episodes) != {2000}:
            raise ValueError("missing chronological window")
        if (m.episodes, m.member_positions, m.support_positions, m.query_positions) != (10000, 2100000, 900000, 1200000):
            raise ValueError("wrong consumed counts")
        np.testing.assert_allclose(w[raw].sum().to_numpy(float), m[raw].to_numpy(float), rtol=2e-12, atol=1e-9)
        for frame in (m.to_dict(), *w.to_dict("records")):
            computed = normalized({k: frame[k] for k in raw})
            for k, v in computed.items():
                if v is not None:
                    np.testing.assert_allclose(frame[k], v, rtol=2e-12, atol=1e-12)
        record = by_record[m.model_id]
        if len(record["consumed_record_stream_sha256"]) != 64:
            raise ValueError("missing consumed record digest")
        ref = record["reference_summary_verified"]
        for k in ("episodes", "member_positions", "cross_class_duplicate_fraction"):
            np.testing.assert_allclose(m[k], ref[k], rtol=2e-12, atol=1e-12)
        keys = set()
        for case in record["witnesses"]:
            key = case["window"], case["kind"]
            if key in keys or not case["window"]*500 < case["step"] <= (case["window"]+1)*500:
                raise ValueError("duplicate/out-of-window witness")
            keys.add(key)
            occurrence = case["occurrences"]
            labels = [v["local_class"] for v in occurrence]
            roles = [v["role"] for v in occurrence]
            if len(set(labels)) != len(labels) or len(occurrence) < 2:
                raise ValueError("witness is not a cross-class conflict")
            if case["kind"] == "query_support" and not {"query", "support"} <= set(roles):
                raise ValueError("missing witness role")
            if case["kind"] == "query_query" and roles.count("query") < 2:
                raise ValueError("missing repeated query")
    # Six repeated streams are matched recordings, not six independent exposures.
    checked = []
    for source in ("ukr_rus", "cp_hk"):
        for seed in range(3):
            part = models[(models.source == source) & (models.seed == seed) & (models.policy == "lowest_sorted")]
            if len(part) != 2:
                raise ValueError("missing campaign control")
            a, b = part.to_dict("records")
            np.testing.assert_allclose([a[k] for k in raw], [b[k] for k in raw], rtol=0, atol=0)
            if by_record[a["model_id"]]["consumed_record_stream_sha256"] != by_record[b["model_id"]]["consumed_record_stream_sha256"]:
                raise ValueError("recorded controls differ")
            checked.append((source, seed))
    return models, windows, {"complete_models": 33, "windows": 165, "recorded_steps": 82500,
        "recorded_episode_occurrences": 330000, "matched_duplicate_stream_pairs": len(checked),
        "distinct_recorded_streams": 27, "full_context_identity_not_recorded": True,
        "all_window_and_rate_reductions_verified": True, "runtime_revision": protocol["revision"]}


def link_targets(models, targets, verified):
    f = targets[targets.decoder == "full_model"].copy()
    members = models[models.campaign == "member"].copy()
    expected = {(m, t, s) for m in members.model_id for t in TARGETS for s in ("original", "fresh")}
    if len(f) != 240 or set(f[["model_id", "dataset", "stream"]].itertuples(index=False, name=None)) != expected:
        raise ValueError("incomplete target grid")
    contracts = {r["model_id"]: r for r in verified}
    for row in f.to_dict("records"):
        r = contracts[row["model_id"]]
        if row["weights_sha256"] != r["final_sha256"] or row["checkpoint"] != r["checkpoint"]:
            raise ValueError("wrong target checkpoint")
        if (row["source"], row["seed"], row["policy"]) != (r["source"], r["seed"], r["policy"]):
            raise ValueError("wrong target arm")
    joined = f.merge(members[["model_id", *EXPOSURES]], on="model_id", validate="many_to_one")
    deltas = []
    keys = ["source", "seed", "dataset", "stream"]
    for key, group in joined.groupby(keys):
        baseline = group[group.policy == "lowest_sorted"].iloc[0]
        for _, row in group[group.policy != "lowest_sorted"].iterrows():
            deltas.append({**dict(zip(keys, key)), "policy": row.policy,
                **{f"delta_{k}": row[k]-baseline[k] for k in (*EXPOSURES, "roc_auc", "accuracy", "nll")}})
    return joined, pd.DataFrame(deltas)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, default=HERE/"data/consumed_conflicts")
    p.add_argument("--data", type=Path, default=HERE/"data")
    args = p.parse_args()
    models, windows, receipt = validate(args.input)
    joined, deltas = link_targets(models, pd.read_csv(args.data/"member_replay_cells.csv"),
        json.loads((args.data/"member_training_verified/arms.json").read_text()))
    joined.to_csv(args.data/"consumed_conflicts_target_cells.csv", index=False)
    deltas.to_csv(args.data/"consumed_conflicts_policy_deltas.csv", index=False)
    models.groupby(["campaign", "source", "policy"])[list(EXPOSURES)].agg(["mean", "min", "max"]).to_csv(
        args.data/"consumed_conflicts_source_summary.csv")
    receipt.update(target_cells=len(joined), policy_comparisons=len(deltas), exploratory=True,
        target_results_reused_not_new_replications=True)
    (args.data/"consumed_conflicts_validation.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))
    print(deltas[(deltas.source=="cp_hk") & (deltas.dataset=="covid_political") &
        (deltas.policy=="lowest_shuffled")][["seed", "stream", "delta_query_also_wrong_class_support_fraction",
        "delta_support_multilabel_support_fraction", "delta_roc_auc", "delta_nll"]].to_string(index=False))


if __name__ == "__main__":
    main()
