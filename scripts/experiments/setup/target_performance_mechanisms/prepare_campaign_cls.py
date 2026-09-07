"""Reuse compatible completed NM campaign models for exploratory CLS replay."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

EXCLUDED_FLAGS = {"feature_standardized", "source_affine", "wide"}
ALLOWED_FLAGS = {"proportional", "blocked", "cross_graph", "degree_balanced", "low_degree",
                 "uniform_positive", "degree_hard", "one_hop", "grad_normalized",
                 "aux_reconstruction", "region_adaptive", "coverage_cycle", "per_source_budget"}
FORWARD = {"layers": "S,U,M", "emb_dim": 256, "input_dim": 768, "gnn_type": "sage",
           "attention_mask_scheme": "causal",
           "dropout": 0., "text_features_dropout": 0., "reset_after_layer": None,
           "skip_path": False, "has_final_back": False, "no_bn_metagraph": False,
           "no_bn_encoder": False, "meta_gnn_pos_only": False, "zero_shot": False,
           "zero_label_embeddings": False, "ignore_label_embeddings": True}


def select_records(records):
    endpoint = [r for r in records if len(r["selection"]["sources"]) == 8]
    if len(endpoint) != 18 or len({r["model_id"] for r in endpoint}) != 18:
        raise ValueError("requires the complete 18-model campaign endpoint inventory")
    included, excluded = [], []
    for record in endpoint:
        p, s = record["params"], record["selection"]
        flags = set(s["flags"])
        if flags & EXCLUDED_FLAGS:
            excluded.append({"model_id": record["model_id"], "flags": sorted(flags),
                             "reason": "requires different forward preprocessing, parameters, or width"})
            continue
        if flags - ALLOWED_FLAGS or any(p[k] != v for k, v in FORWARD.items()):
            raise ValueError(f"unsupported forward or unknown flags: {record['model_id']}")
        if s["status"] != "complete" or record["runtime"]["result"]["status"] != "complete" or p["seed"] != 0:
            raise ValueError("incomplete or different-seed campaign model")
        if "twibot20" in s["sources"] or p["batch_size"] != 1 or p["learning_rate"] != .001:
            raise ValueError("unexpected source holdout or training protocol")
        if (p["n_way"], p["n_shots"], p["n_query"]) != (30, 3, 4) or p["edge_view"] != "static_train":
            raise ValueError("unexpected NM task or background relation")
        if any(float(s["exposure"].get(t, 0)) != 0 for t in ["twibot20"]):
            raise ValueError("unseen target received training exposure")
        included.append(record)
    if len(included) != 15 or len(excluded) != 3:
        raise ValueError("compatible endpoint subset differs")
    if len({tuple(r["selection"]["sources"]) for r in included}) != 1:
        raise ValueError("training source compositions differ")
    common_step = min(r["selection"]["training_steps"] for r in included)
    if common_step != 6000:
        raise ValueError("maximum common checkpoint changed; review before outcomes")
    return included, excluded, common_step


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.records.read_text())
    included, excluded, common_step = select_records(records)
    args.output.mkdir(parents=True, exist_ok=False)
    rows, details = [], []
    for record in included:
        s, p = record["selection"], record["params"]
        for role, step in (("common6000", common_step), ("source_val_selected", s["best_step"])):
            checkpoint = str(Path(s["checkpoint"]).with_name(f"state_dict_{step}.ckpt"))
            model_id = f"{record['model_id']}__{role}"
            rows.append({"model_id": model_id, "checkpoint": checkpoint, "sources": ",".join(s["sources"])})
            details.append({"model_id": model_id, "original_model_id": record["model_id"],
                "checkpoint_role": role, "checkpoint": checkpoint, "step": step,
                "training_steps": s["training_steps"], "flags": s["flags"], "sources": s["sources"],
                "seed": p["seed"], "training_revision": p["campaign_revision"],
                "source_validation_auc": s["best_val"] if role == "source_val_selected" else None,
                "actual_training_batch_size": 1, "actual_learning_rate": .001,
                "forward_params": {k: p[k] for k in FORWARD},
                "classification_adaptation": {"task_name": "classification", "ignore_label_embeddings": False}})
    with (args.output / "model_list.tsv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_id", "checkpoint", "sources"], delimiter="\t")
        writer.writeheader(); writer.writerows(rows)
    (args.output / "manifest.json").write_text(json.dumps({"models": details, "excluded": excluded,
        "input_records": str(args.records.resolve()), "input_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
        "common_steps_and_episodes": common_step, "training_seeds": [0], "new_training": False,
        "selection_uses_downstream_cls": False, "all_training_sources": 8,
        "held_out_source": "twibot20", "default_target_episode_contract": "existing 128-episode 2-way 10-shot CLS"}, indent=2) + "\n")
    print(json.dumps({"models": len(rows), "underlying_training_runs": len(included),
                      "excluded": excluded, "common_step": common_step}))


if __name__ == "__main__":
    main()
