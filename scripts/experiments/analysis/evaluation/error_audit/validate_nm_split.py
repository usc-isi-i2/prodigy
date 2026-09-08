#!/usr/bin/env python3
"""Reconcile public aggregate evidence with canonical-split inference records."""
import argparse
import json
from pathlib import Path


def validate(root):
    protocol = json.loads((root/"protocol_summary.json").read_text())
    assert protocol["episodes_per_target_split"] == 512
    assert (protocol["n_way"],protocol["n_shots"],protocol["n_query"]) == (30,3,4)
    comparisons = []
    for source, canonical in [("ukr_rus","ukraine"),("cp_hk","hongkong")]:
        bio = json.loads((root/f"nm_bio_clusters_{canonical}.json").read_text())
        weights = json.loads((root/f"nm_query_weighting_{canonical}.json").read_text())
        for split in ["val","test"]:
            cell = protocol["targets"][source]["splits"][split]
            counts = cell["edge_audit"]["membership_counts"]
            assert counts[split] == cell["edge_audit"]["unique_anchor_member_pairs"]
            assert all(v == 0 for k,v in counts.items() if k != split)
            if split == "test":
                assert cell["published_plan_match"] and cell["published_identity_match"]
            b = bio["baseline"][split]
            assert b["n"] == 61440 and b["distinct_episodes"] == 512 and b["queries_per_episode"] == [120]
            assert abs(sum(b[k] for k in ["both_correct","both_wrong","ukr_only","hk_only"])-1) < 1e-12
            for model,short in [("ukr_rus","ukr"),("cp_hk","hk")]:
                m = cell["models"][model]
                assert m["identical_input_replay"] and m["query_occurrences"] == b["n"]
                assert abs(m["accuracy"]-b[f"{short}_accuracy"]) < 1e-8
            clusters = bio["runs"]["8"]["clusters"]
            assert sum(c["streams"][split]["n"] for c in clusters) == b["n"]
            for model in ["ukr","hk"]:
                weighted = sum(c["streams"][split]["n"]*c["streams"][split][f"{model}_accuracy"] for c in clusters)/b["n"]
                assert abs(weighted-b[f"{model}_accuracy"]) < 1e-12
            w = weights["splits"][split]
            assert sum(g["n"] for g in w["frequency_groups"].values()) == b["n"]
            assert sum(g["unique_queries"] for g in w["frequency_groups"].values()) == b["unique_queries"]
            assert w["validation_reused_occurrences"] + w["novel_vs_val"]["n"] == b["n"]
            comparisons.append({"target":canonical,"split":split,"ukr":b["ukr_accuracy"],"hk":b["hk_accuracy"],
                                "train_pair_overlap":counts["train"],"queries":b["n"],"episodes":b["distinct_episodes"]})
    print(json.dumps({"validated":True,"comparisons":comparisons},indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir",type=Path,required=True)
    validate(p.parse_args().data_dir)
