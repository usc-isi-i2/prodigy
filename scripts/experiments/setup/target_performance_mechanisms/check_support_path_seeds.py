"""Check the political support-edge finding on six pre-existing three-seed arms.

This does not train and does not regard CPU execution repeats as new seeds.
It uses the production-sort arms of the completed member-policy factorial.
"""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .replay import batch_hash, clone_batch, trace_stages
from .role_context import intervene_roles, query_mask
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def inspect_path(model, batch):
    if model.layer_list[2].num_gnn_layers != 1 or model.layer_list[2].gnn_layers_back is not None:
        raise ValueError("expected one metagraph layer without final reverse layer")
    pre, logits = {}, {}
    for condition in ("baseline", "edges_support"):
        altered = intervene_roles(batch, condition)
        with trace_stages(model, altered[0]) as traces:
            _, logits[condition], _ = model(*altered)
        pre[condition] = traces["U1_pre_meta"]
    q = query_mask(batch)
    torch.testing.assert_close(pre["baseline"][q], pre["edges_support"][q], rtol=0, atol=0)
    label_input = model.initial_label_mlp(batch[1])
    post = {v: model.forward_metagraph(model.layer_list[2], z, label_input, *batch[3:9]) for v, z in pre.items()}
    torch.testing.assert_close(post["baseline"][0][q], post["edges_support"][0][q], rtol=0, atol=0)
    a, b = [post[v][1] for v in ("baseline", "edges_support")]
    # Reconstruct the measured change using the unchanged query side and altered
    # label side; production decoding must reproduce the whole-forward result.
    recomposed = model.decode(post["baseline"][0], b, batch[3]).reshape(batch[2].shape)[q]
    torch.testing.assert_close(recomposed, logits["edges_support"], rtol=0, atol=0)
    return logits, {"query_pre_and_post_bit_exact": True, "label_only_recomposition_bit_exact": True,
                    "label_max_abs_change": float((a-b).abs().max()),
                    "mean_label_cosine": float(F.cosine_similarity(a, b).mean()),
                    "changed_label_vectors": int((a != b).any(1).sum()), "label_vectors": len(a)}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--arms", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--fresh-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("new output and hidden GPUs required")
    torch.set_num_threads(4)
    arms = [a for a in json.loads(args.arms.read_text()) if a["policy"] == "lowest_sorted"]
    if len(arms) != 6 or {(a["source"], a["seed"]) for a in arms} != {(s, n) for s in ("ukr_rus", "cp_hk") for n in range(3)}:
        raise ValueError("complete six-arm, three-seed control required")
    ref = pd.read_csv(args.references)
    ref = ref[ref.dataset.eq("covid_political") & ref.decoder.eq("full_model") & ref.policy.eq("lowest_sorted")]
    if len(ref) != 12:
        raise ValueError("12 complete saved reference cells required")
    if args.dry_run:
        print("Six existing models, both fixed streams, two conditions: 24 cells; 384 label-path checks; no training.")
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "arms": str(args.arms), "references": str(args.references), "new_training": False,
        "selection": "All original-sort controls, both sources, all three seeds; no checkpoint selection.",
        "expectation": "Check whether the historical seed-0 Hong Kong benefit from support-edge removal recurs across all three independent initialization seeds and both streams.",
        "runtime_variability_caveat": "These earlier CPU-trained arms include execution variability; this test changes only fixed-model inputs.",
        "intervention": "Remove only the support subgraphs' background edges; retain all query input and all sampled members/features."})
    rows, audits = [], []
    with torch.no_grad():
        for stream, root in (("original", args.original_root), ("fresh", args.fresh_root)):
            directory = root / "covid_political"
            protocol = json.loads((root / "protocol.json").read_text())
            labels = input_labels(directory, "covid_political")
            batches = [torch.load(directory / "batches" / f"batch_{i:03d}.pt", map_location="cpu", weights_only=False) for i in range(32)]
            for i, b in enumerate(batches):
                if batch_hash(b) != labels["cache"]["batch_sha256"][i]:
                    raise ValueError("input changed")
            for arm in arms:
                r = ref[(ref.stream == stream) & (ref.model_id == arm["model_id"])].iloc[0]
                if r.episode_fingerprint != labels["cache"]["episode_fingerprint"] or r.checkpoint != arm["checkpoint"]:
                    raise ValueError("reference model/input differs")
                state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
                if model_digest(state) != arm["final_sha256"]:
                    raise ValueError("checkpoint differs from verified training receipt")
                model = make_model(protocol, "covid_political", labels["cache"]["graph_path"], batches[0][0].x.shape[1], state)
                collected = {v: [] for v in ("baseline", "edges_support")}
                for bi, batch in enumerate(batches):
                    logits, audit = inspect_path(model, batch)
                    for v, z in logits.items():
                        collected[v].append(z)
                    audits.append({"stream": stream, "model_id": arm["model_id"], "source": arm["source"], "seed": arm["seed"],
                                   "batch": bi, "batch_sha256": labels["cache"]["batch_sha256"][bi], **audit})
                predictions = {v: torch.cat(z) for v, z in collected.items()}
                for condition, z in predictions.items():
                    scores = evaluate_logits(z, labels)
                    if condition == "baseline" and max(abs(scores[k] - r[k]) for k in METRICS) > 1e-6:
                        raise ValueError("baseline differs from pre-existing evaluation")
                    rows.append({"stream": stream, "target": "covid_political", "model_id": arm["model_id"],
                                 "source": arm["source"], "seed": arm["seed"], "condition": condition,
                                 "checkpoint": arm["checkpoint"], "weights_sha256": arm["final_sha256"],
                                 "episode_fingerprint": labels["cache"]["episode_fingerprint"], **scores})
                if model_digest(model.state_dict()) != arm["final_sha256"]:
                    raise ValueError("model mutated")
                torch.save({"predictions": predictions, "labels": labels}, args.output / f"{stream}_{arm['model_id']}.pt")
                print(json.dumps(rows[-2:]), flush=True)
    if len(rows) != 24 or len(audits) != 384:
        raise ValueError("incomplete result grid")
    write_json(args.output / "metrics.json", rows)
    write_json(args.output / "label_path_audit.json", audits)
    write_json(args.output / "DONE.json", {"complete": True, "cells": 24, "label_path_checks": 384,
        "all_query_embeddings_unchanged": True, "all_logits_recomposed_from_label_change": True,
        "all_baseline_metrics_match_saved": True, "all_weights_unchanged": True})


if __name__ == "__main__":
    main()
