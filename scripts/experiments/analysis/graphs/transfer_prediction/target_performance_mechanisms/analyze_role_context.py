"""Validate the entire role factorial; report both rescues and newly caused errors."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from scripts.experiments.setup.target_performance_mechanisms.prepare_mixture_complementarity import TARGETS
from .analyze_mixture_complementarity import complete_grid, METRICS, STREAMS

VARIANTS = ("baseline", "features_query", "features_support", "features_both",
            "edges_query", "edges_support", "edges_both", "center_zero_query")
COHORTS = ("all", "no_context", "has_context", "incoming_dominant", "outgoing_dominant",
           "balanced_nonzero_degree", "raw_center_context_agree", "raw_center_context_disagree")
KEY = ["stream", "target", "model_id"]


def validate_paired_record(r):
    n, bc, ac, fixed, broken = [r[k] for k in ("queries", "baseline_correct", "altered_correct", "errors_fixed", "errors_introduced")]
    if any(not isinstance(v, int) or v < 0 for v in (n, bc, ac, fixed, broken)):
        raise ValueError("paired counts must be nonnegative integers")
    if max(bc, ac) > n or fixed > n-bc or broken > bc or ac-bc != fixed-broken:
        raise ValueError("invalid paired error accounting")
    if (n == 0 and r["delta_accuracy"] is not None) or (n and not np.isclose(r["delta_accuracy"], (fixed-broken)/n, rtol=0, atol=1e-12)):
        raise ValueError("paired accuracy delta disagrees")
    classes = r["by_global_class"]
    for k in ("queries", "baseline_correct", "altered_correct", "errors_fixed", "errors_introduced"):
        if sum(part[k] for part in classes.values()) != r[k]:
            raise ValueError("class decomposition does not sum to total")
    for part in classes.values():
        n1, bc1, ac1, f1, b1 = [part[k] for k in ("queries", "baseline_correct", "altered_correct", "errors_fixed", "errors_introduced")]
        if any(not isinstance(v, int) or v < 0 for v in (n1, bc1, ac1, f1, b1)) or max(bc1, ac1) > n1 or f1 > n1-bc1 or b1 > bc1 or ac1-bc1 != f1-b1:
            raise ValueError("invalid within-class paired counts")


def validate(root, reference_root):
    read = lambda name: json.loads((root / f"{name}.json").read_text())
    done, protocol = read("DONE"), read("protocol")
    expected = {"cells": 720, "models_target_stream": 90, "paired_cohort_cells": 5760,
                "baseline_bit_exact_batches": 2880, "all_weights_unchanged": True, "all_baseline_metrics_reproduced": True}
    if any(done.get(k) != v for k, v in expected.items()) or protocol["variants"] != list(VARIANTS) or protocol["new_training"] is not False:
        raise ValueError("complete declared factorial required")
    cells, inventory, inputs = [pd.DataFrame(read(n)) for n in ("metrics", "inventory", "input_inventory")]
    paired_records = read("paired_cohorts")
    paired = pd.DataFrame(paired_records)
    ids = ["ss_" + s for s in SOURCES]
    complete_grid(cells, KEY + ["variant"], set(product(STREAMS, TARGETS, ids, VARIANTS)), "complete 720-cell grid required")
    complete_grid(inventory, KEY, set(product(STREAMS, TARGETS, ids)), "complete 90 model/input audits required")
    complete_grid(inputs, KEY[:2], set(product(STREAMS, TARGETS)), "complete 10 input caches required")
    complete_grid(paired, KEY + ["variant", "cohort"], set(product(STREAMS, TARGETS, ids, VARIANTS, COHORTS)), "complete paired cohort grid required")
    if not np.isfinite(cells[list(METRICS)]).all().all() or not cells.nll.ge(0).all() or not cells.episodes.eq(128).all():
        raise ValueError("invalid metrics")
    if not cells[["accuracy", "roc_auc", "f1"]].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError("invalid bounded metric")
    if not inventory.baseline_bit_exact_batches.eq(32).all() or not inventory.direct_factorization_checks.eq(75).all() or not inventory.max_factorization_error.between(0, 1e-5).all():
        raise ValueError("missing or failed factorization checks")
    if done["max_factorization_error"] != inventory.max_factorization_error.max():
        raise ValueError("receipt and observed parity disagree")
    for model_id, group in inventory.groupby("model_id"):
        if group.weights_sha256.nunique() != 1 or group.checkpoint.nunique() != 1:
            raise ValueError("model changed across target inputs")
    for r in inputs.itertuples():
        if r.episodes != 128 or len(r.batch_sha256) != 32 or len(set(r.batch_sha256)) != 32:
            raise ValueError("invalid cached episode inventory")
    for target, group in inputs.groupby("target"):
        if group.episode_fingerprint.nunique() != 2 or group.graph_path.nunique() != 1:
            raise ValueError("two distinct streams on the same graph required")
    lookup = inventory.set_index(KEY)
    cache_lookup = inputs.set_index(KEY[:2])
    for r in cells.itertuples():
        prior = lookup.loc[(r.stream, r.target, r.model_id)]
        if r.source != r.model_id.removeprefix("ss_") or r.foreign_source != (r.source != r.target):
            raise ValueError("source ownership differs")
        if any(getattr(r, k) != prior[k] for k in ("checkpoint", "weights_sha256", "episode_fingerprint", "queries")):
            raise ValueError("cell model/input identity differs")
        if r.episode_fingerprint != cache_lookup.loc[(r.stream, r.target), "episode_fingerprint"]:
            raise ValueError("cell cache identity differs")
    metric_lookup = cells.set_index(KEY + ["variant"])
    for r in paired_records:
        validate_paired_record(r)
        if r["cohort"] == "all":
            cell = metric_lookup.loc[tuple(r[k] for k in KEY + ["variant"])]
            if r["queries"] != cell.queries or not np.isclose(r["altered_correct"] / r["queries"], cell.accuracy, rtol=0, atol=1e-12):
                raise ValueError("paired counts disagree with global accuracy")
    pidx = paired.set_index(KEY + ["variant", "cohort"])
    count_fields = ["queries", "baseline_correct", "altered_correct", "errors_fixed", "errors_introduced"]
    for key in product(STREAMS, TARGETS, ids, VARIANTS):
        for whole, parts in (("all", ("no_context", "has_context")),
                             ("has_context", ("raw_center_context_agree", "raw_center_context_disagree"))):
            total = pidx.loc[key + (whole,), count_fields].to_numpy(dtype=int)
            summed = sum(pidx.loc[key + (part,), count_fields].to_numpy(dtype=int) for part in parts)
            if not np.array_equal(total, summed):
                raise ValueError("cohort partition does not conserve counts")
    # A genuinely empty query context cannot be affected by changing its absent
    # neighbor features or background edges. Support changes remain permitted.
    empty_query_changes = paired[paired.cohort.eq("no_context") & paired.variant.isin(["features_query", "edges_query"])]
    if not empty_query_changes[["errors_fixed", "errors_introduced"]].eq(0).all().all():
        raise ValueError("empty query context changed under query-context intervention")
    refs = []
    for stream, name in (("original", "replay_cells.csv"), ("fresh", "fresh_replay_cells.csv")):
        ref = pd.read_csv(reference_root / name)
        ref = ref[ref.decoder.eq("full_model") & ref.model_id.isin(ids)].copy()
        ref["stream"] = stream
        ref["variant"] = ref.variant.map({"baseline": "baseline", "center_features_only": "features_both", "no_background_edges": "edges_both"})
        refs.append(ref.dropna(subset=["variant"]).rename(columns={"dataset": "target"}))
    ref = pd.concat(refs, ignore_index=True)
    matched = cells.merge(ref, on=KEY + ["variant"], suffixes=("", "_reference"), validate="one_to_one")
    if len(matched[matched.variant.eq("baseline")]) != 90:
        raise ValueError("all 90 independent saved baselines required")
    for field in ("checkpoint", "weights_sha256", "episode_fingerprint", "queries"):
        if not matched[field].eq(matched[field + "_reference"]).all():
            raise ValueError("independent saved reference identity differs")
    if any((matched[k]-matched[k + "_reference"]).abs().max() > 1e-6 for k in METRICS):
        raise ValueError("independent saved reference metrics differ")
    baseline = cells[cells.variant.eq("baseline")].set_index(KEY)
    for k in METRICS:
        cells["delta_" + k] = [r[k] - baseline.loc[tuple(r[v] for v in KEY), k] for r in cells.to_dict("records")]
    return cells, paired, {"cells": len(cells), "paired_cohorts": len(paired), "independent_reference_cells": len(matched),
        "baseline_bit_exact_batches": 2880, "direct_factorization_checks": int(inventory.direct_factorization_checks.sum()),
        "max_factorization_error": float(inventory.max_factorization_error.max())}


def interactions(cells):
    pivot = cells.pivot(index=KEY, columns="variant", values=list(METRICS))
    rows = []
    for key, row in pivot.iterrows():
        for family in ("features", "edges"):
            out = dict(zip(KEY, key)) | {"family": family}
            for metric in METRICS:
                base, q, s, both = [row[(metric, v)] for v in ("baseline", family + "_query", family + "_support", family + "_both")]
                out["query_effect_" + metric] = q-base
                out["support_effect_" + metric] = s-base
                out["joint_effect_" + metric] = both-base
                out["interaction_" + metric] = both-q-s+base
            rows.append(out)
    return pd.DataFrame(rows)


def summarize(cells):
    rows = []
    for (stream, target, variant), group in cells.groupby(["stream", "target", "variant"]):
        for scope, selected in (("all", group), ("foreign", group[group.foreign_source])):
            rows.append({"stream": stream, "target": target, "variant": variant, "scope": scope,
                         "models": len(selected), "models_with_auc_gain": int(selected.delta_roc_auc.gt(0).sum()),
                         **{"mean_delta_" + m: float(selected["delta_" + m].mean()) for m in METRICS}})
    primary = cells[cells.target.eq("covid_political") & cells.model_id.eq("ss_ukr_rus") & cells.variant.eq("features_query")]
    if len(primary) != 2 or set(primary.stream) != set(STREAMS):
        raise ValueError("both directional expectation cells required")
    result = {"ukraine_political_query_context_dependence_both_streams": bool(primary.delta_roc_auc.lt(0).all()),
              "directional_cells": primary[["stream", *["delta_" + m for m in METRICS]]].to_dict("records"),
              "independent_training_seeds": 1, "confirmatory_study": False}
    return pd.DataFrame(rows), result


def source_gaps(cells, paired):
    """Decompose the named political donor contrasts without selecting queries."""
    metric = cells.set_index(KEY + ["variant"])
    counts = paired.set_index(KEY + ["variant", "cohort"])
    rows = []
    for stream, better in product(STREAMS, ("ss_ukr_rus", "ss_covid")):
        a = (stream, "covid_political", better)
        b = (stream, "covid_political", "ss_cp_hk")
        gap = metric.loc[a + ("baseline",), "roc_auc"] - metric.loc[b + ("baseline",), "roc_auc"]
        altered_gap = metric.loc[a + ("edges_support",), "roc_auc"] - metric.loc[b + ("edges_support",), "roc_auc"]
        total_gap = counts.loc[a + ("baseline", "all"), "baseline_correct"] - counts.loc[b + ("baseline", "all"), "baseline_correct"]
        for cohort in ("all", "no_context", "has_context"):
            ca, cb = [counts.loc[k + ("baseline", cohort)] for k in (a, b)]
            if ca.queries != cb.queries:
                raise ValueError("source contrast must use identical query cohorts")
            excess = int(ca.baseline_correct - cb.baseline_correct)
            rows.append({"stream": stream, "target": "covid_political", "model_a": better, "model_b": "ss_cp_hk",
                         "cohort": cohort, "queries": int(ca.queries), "a_correct": int(ca.baseline_correct),
                         "b_correct": int(cb.baseline_correct), "excess_correct": excess,
                         "fraction_of_total_correct_gap": excess / total_gap if total_gap else None,
                         "baseline_auc_gap": float(gap), "support_edges_removed_auc_gap": float(altered_gap),
                         "fraction_auc_gap_reduced": float(1 - altered_gap/gap) if gap else None})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args()
    cells, paired, validation = validate(args.input, args.data)
    summary, expectation = summarize(cells)
    cells.to_csv(args.data / "role_context_cells.csv", index=False)
    summary.to_csv(args.data / "role_context_summary.csv", index=False)
    interactions(cells).to_csv(args.data / "role_context_interactions.csv", index=False)
    source_gaps(cells, paired).to_csv(args.data / "role_context_source_gaps.csv", index=False)
    flat = paired.drop(columns=["by_global_class"])
    flat.to_csv(args.data / "role_context_paired_cohorts.csv", index=False)
    class_rows = []
    for row in paired.to_dict("records"):
        for label, counts in row["by_global_class"].items():
            class_rows.append({k: row[k] for k in KEY + ["variant", "cohort"]} | {"global_label": int(label)} | counts)
    pd.DataFrame(class_rows).to_csv(args.data / "role_context_classwise_counts.csv", index=False)
    cohort_rows = []
    for row in paired.to_dict("records"):
        parts = list(row["by_global_class"].values())
        if not parts:
            continue
        cohort_rows.append({k: row[k] for k in KEY + ["variant", "cohort", "queries"]} |
            {"classes_observed": len(parts), "baseline_accuracy": row["baseline_correct"] / row["queries"],
             "altered_accuracy": row["altered_correct"] / row["queries"],
             "baseline_macro_class_recall": float(np.mean([r["baseline_correct"] / r["queries"] for r in parts])),
             "altered_macro_class_recall": float(np.mean([r["altered_correct"] / r["queries"] for r in parts])),
             "largest_class_fraction": max(r["queries"] for r in parts) / row["queries"]})
    pd.DataFrame(cohort_rows).to_csv(args.data / "role_context_cohort_metrics.csv", index=False)
    result = validation | expectation
    (args.data / "role_context_validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
