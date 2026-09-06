"""Evaluate the prespecified count-restoration prediction, retaining all controls."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from scripts.experiments.setup.target_performance_mechanisms.episode_cardinality_contract import VARIANTS
from scripts.experiments.setup.target_performance_mechanisms.prepare_mixture_complementarity import TARGETS
from .analyze_mixture_complementarity import complete_grid, METRICS, STREAMS

KEY = ["stream", "target", "model_id"]
ROLES = {"label": ("label_positive", "label_negative", "label_self"),
         "query": ("query_labels", "query_self"),
         "support": ("support_positive", "support_negative", "support_self")}


def validate(root):
    def read(name):
        return json.loads((root / f"{name}.json").read_text())
    done = read("DONE")
    expected_done = {"full_model_cells": 630, "model_target_stream_cells": 90,
        "baseline_suffix_bit_exact_batches": 2880, "all_weights_unchanged": True,
        "all_cached_inputs_verified": True, "all_baseline_metrics_reproduced": True}
    if done != expected_done:
        raise ValueError("incomplete count diagnostic receipt")
    protocol = read("protocol")
    if protocol["variants"] != list(VARIANTS) or protocol["query_outcomes_used_for_intervention"] is not False or protocol["new_distinct_negative_classes"] is not False:
        raise ValueError("prespecified intervention changed")
    cells, inventory, inputs = (pd.DataFrame(read(name)) for name in ("metrics", "inventory", "input_inventory"))
    ids = ["ss_" + source for source in SOURCES]
    expected = set(product(STREAMS, TARGETS, ids))
    complete_grid(cells, KEY + ["variant"], set(product(STREAMS, TARGETS, ids, VARIANTS)), "complete 630-cell grid required")
    complete_grid(inventory, KEY, expected, "90 checkpoint/prediction audits required")
    complete_grid(inputs, KEY[:2], set(product(STREAMS, TARGETS)), "10 input audits required")
    if not np.isfinite(cells[list(METRICS)]).all().all() or not cells.episodes.eq(128).all() or not cells.nll.ge(0).all():
        raise ValueError("invalid metrics or episode count")
    if not cells[["roc_auc", "accuracy", "f1"]].apply(lambda col: col.between(0, 1)).all().all():
        raise ValueError("invalid bounded metric")
    lookup = inventory.set_index(KEY)
    caches = inputs.set_index(KEY[:2])
    for model_id, group in inventory.groupby("model_id"):
        if group.weights_sha256.nunique() != 1 or group.checkpoint.nunique() != 1:
            raise ValueError("checkpoint changed across targets/streams")
    for row in inventory.itertuples():
        if row.baseline_suffix_bit_exact_batches != 32 or any(len(getattr(row, k)) != 64 for k in ("weights_sha256", "embedding_file_sha256", "prediction_file_sha256")):
            raise ValueError("missing numerical/identity audit")
    for row in cells.itertuples():
        inv, cache = lookup.loc[(row.stream, row.target, row.model_id)], caches.loc[(row.stream, row.target)]
        if row.source != row.model_id.removeprefix("ss_") or any(getattr(row, k) != inv[k] for k in ("checkpoint", "weights_sha256")):
            raise ValueError("cell model identity differs")
        if row.queries != cache.queries or row.episode_fingerprint != cache.episode_fingerprint or len(cache.batch_sha256) != 32:
            raise ValueError("cell input identity differs")
    for target in TARGETS:
        a, b = (caches.loc[(stream, target)] for stream in STREAMS)
        if a.episode_fingerprint == b.episode_fingerprint or a.batch_sha256 == b.batch_sha256 or a.graph_path != b.graph_path:
            raise ValueError("two distinct streams on the same target required")
    attentions = read("attention")
    complete_grid(pd.DataFrame(attentions), KEY + ["variant"], set(product(STREAMS, TARGETS, ids, VARIANTS)), "630 attention audits required")
    attention_rows = []
    for record in attentions:
        batch_records = record["batches"]
        cache = caches.loc[(record["stream"], record["target"])]
        if len(batch_records) != 32:
            raise ValueError("incomplete attention audit")
        for batch in batch_records:
            if any(batch[k] != v for k, v in {"eval_ways": 2, "eval_shots": 10, "train_ways": 30,
                "train_shots": 3, "label_nodes": 8, "support_nodes": 80}.items()) or batch["query_nodes"] != cache.queries / 32:
                raise ValueError("observed episode cardinality differs")
            for roles in ROLES.values():
                masses = np.asarray([batch["attention_mass_per_head"][r] for r in roles])
                if masses.shape != (len(roles), 8) or not np.isfinite(masses).all() or (masses < 0).any() or not np.allclose(masses.sum(0), 1, atol=1e-5, rtol=0):
                    raise ValueError("attention roles do not conserve per-destination mass")
        attention_rows.append({k: record[k] for k in KEY + ["variant"]} | {
            role: float(np.mean([b["attention_mass_per_head"][role] for b in batch_records]))
            for roles in ROLES.values() for role in roles})
    baseline = cells[cells.variant.eq("baseline")].set_index(KEY)
    for metric in METRICS:
        cells["delta_" + metric] = [row[metric] - baseline.loc[tuple(row[k] for k in KEY), metric] for row in cells.to_dict("records")]
    cells["foreign"] = cells.source.ne(cells.target)
    return cells, pd.DataFrame(attention_rows)


def summarize(cells):
    rows = []
    for (stream, target, variant), group in cells[~cells.variant.eq("baseline")].groupby(["stream", "target", "variant"]):
        for scope, subset in (("all", group), ("foreign", group[group.foreign])):
            rows.append({"stream": stream, "target": target, "variant": variant, "scope": scope,
                "models": len(subset), "positive_auc_models": int(subset.delta_roc_auc.gt(0).sum()),
                **{"mean_delta_" + metric: float(subset["delta_" + metric].mean()) for metric in METRICS}})
    summary = pd.DataFrame(rows)
    fixed = summary[summary.scope.eq("foreign")].set_index(["stream", "target", "variant"])
    primary = []
    for target, stream in product(("facebook_page_reference", "twibot20"), STREAMS):
        joint = fixed.loc[(stream, target, "count_joint_attention")]
        inverse = fixed.loc[(stream, target, "count_inverse_attention")]
        if joint.models != 8 or inverse.models != 8:
            raise ValueError("eight foreign donors required for primary")
        change, control = float(joint.mean_delta_roc_auc), float(inverse.mean_delta_roc_auc)
        primary.append({"stream": stream, "target": target, "mean_auc_change": change,
            "inverse_control_mean_auc_change": control, "joint_minus_inverse_auc": change - control,
            "positive_donors": int(joint.positive_auc_models), "passed": bool(change > 0 and change > control)})
    return summary, {"primary_prediction_passed": all(r["passed"] for r in primary), "primary_cells": primary,
        "independent_training_seeds": 1, "new_distinct_negative_classes": False,
        "inference": "fixed counterfactual attention mass, not actual 30-class evaluation or proof of a training cause"}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cells, attention = validate(args.results)
    summary, result = summarize(cells)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in (("cells", cells), ("summary", summary), ("attention", attention)):
        frame.to_csv(args.output / f"episode_cardinality_{name}.csv", index=False)
    result["input_sha256"] = {name: hashlib.sha256((args.results / name).read_bytes()).hexdigest()
        for name in ("protocol.json", "DONE.json", "metrics.json", "inventory.json", "input_inventory.json", "attention.json")}
    (args.output / "episode_cardinality_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
