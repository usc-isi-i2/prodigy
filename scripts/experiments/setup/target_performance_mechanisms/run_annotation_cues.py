"""Outcome-independent published-cue strata, reusing complete fixed-model logits.

Raw profiles/account IDs stay on Tucker. This does NOT create new political
labels, remove cues, retrain models, resample supports or reconstruct news labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from scripts.social_llm.generate_graph import read_user_data_csv
from experiments.run_shared_graph import write_json
from .annotation_cues import SOURCE_URL, PUBLISHED_CUES, DETECTORS, SUBSETS, cue_flags, join_original_profiles, subset_masks
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .replay import batch_hash, episode_probe
from .role_context import query_mask


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(logits, labels, mask, ids):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    count = int(mask.sum())
    kept_ids = ids[mask]
    row = {"queries": count, "unique_queries": int(kept_ids.unique().numel()),
           "episodes": int(labels["episode_ids"][mask].unique().numel())}
    selected = {**labels, **{k: labels[k][mask] for k in ("local_y", "mapping", "episode_ids")}}
    y = selected["mapping"][torch.arange(count), selected["local_y"]]
    row.update({"class_0": int((y == 0).sum()), "class_1": int((y == 1).sum())})
    if count == 0 or y.unique().numel() != 2:
        return {**row, **{k: None for k in METRICS}, "unique_account_auc": None}
    row.update(evaluate_logits(logits[mask], selected))
    # Descriptive identity-balanced sensitivity: average each account's scores
    # over its occurrences BEFORE AUC. This is not a bootstrap/independence claim.
    prob = logits[mask].softmax(1)
    p1 = prob[torch.arange(count), (selected["mapping"] == 1).long().argmax(1)].double()
    unique_ids, inverse = kept_ids.unique(return_inverse=True)
    sums, counts = torch.zeros(len(unique_ids), dtype=torch.float64), torch.zeros(len(unique_ids), dtype=torch.float64)
    sums.scatter_add_(0, inverse, p1)
    counts.scatter_add_(0, inverse, torch.ones(count, dtype=torch.float64))
    unique_y = torch.zeros(len(unique_ids), dtype=torch.long)
    unique_y[inverse] = y
    if not torch.equal(unique_y[inverse], y):
        raise ValueError("Repeated account has inconsistent ground truth")
    row["unique_account_auc"] = float(roc_auc_score(unique_y.numpy(), (sums/counts).numpy()))
    return row


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--topology", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 4:
        raise ValueError("New output, hidden GPUs, bounded CPU required")
    torch.set_num_threads(args.threads)
    prior = json.loads((args.topology/"protocol.json").read_text())
    done = json.loads((args.topology/"DONE.json").read_text())
    if done["smoke_only"] or done["cells"] != 1140 or not done["all_weights_unchanged"]:
        raise ValueError("Complete role-topology panel required")
    conditions = [("intact", "intact", 0), ("intact", "removed", 0),
                  ("removed", "intact", 0), ("removed", "removed", 0)]
    conditions += [("intact", "rewired", d) for d in range(3)]
    manifest = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_url": SOURCE_URL, "source_table": 3, "published_cues": PUBLISHED_CUES,
        "detectors": DETECTORS, "subsets": SUBSETS, "target": "covid_political",
        "streams": ["original", "fresh"], "models": prior["models"], "conditions": conditions,
        "topology": str(args.topology), "threads": args.threads,
        "design": "Fixed published vocabulary and all declared strata before examining outcome associations. Original hashtag detector and broader original-or-cleaned whole-token detector with digit-deleted aliases. No political polarity assigned. Raw prototype uses unchanged episode supports.",
        "limitation": "Cue absence is not independently labeled, leakage-free, or news-heuristic-free. Original-profile match is provenance, not label-generator reconstruction. All input absent includes selected query subgraph plus all support subgraphs, not other queries. Strata overlap, are observational and reuse three seeds/two streams. No significance or fresh-target claim.",
        "new_forward_pass": False, "new_training": False, "private_raw_export": False}
    if args.dry_run:
        print(json.dumps(manifest, indent=2)); return
    args.output.mkdir(parents=True)
    write_json(args.output/"protocol.json", manifest)
    start = time.monotonic()
    source = args.data_root/"social_llm_data/covid"
    table = read_user_data_csv(str(source/"user_data.csv")).fillna("")
    full = read_user_data_csv(str(source/"full_user_data.csv")).fillna("")
    raw = join_original_profiles(table, full)
    flags = torch.tensor([cue_flags(r, c) for r, c in zip(raw, table.profile)], dtype=torch.bool)
    artifact = args.data_root/"covid_political/graphs/retweet_graph.pt"
    graph = torch.load(artifact, map_location="cpu", weights_only=False)
    if list(map(int, graph["user_ids"])) != list(range(len(table))):
        raise ValueError("Graph identities are not aligned CSV rows")
    torch.testing.assert_close(graph["y"].long(), torch.tensor(table.label_conservative.to_numpy()).long(), atol=0, rtol=0)
    graph_counts = []
    for di, detector in enumerate(DETECTORS):
        for cls in (0, 1):
            selected = graph["y"] == cls
            graph_counts.append({"detector": detector, "class": cls, "rows": int(selected.sum()),
                                 "cue_rows": int(flags[selected, di].sum())})
    provenance = {"graph_rows": len(table), "unique_complete_row_matches": len(raw),
        "all_labels_match": True, "cleaned_profiles_with_hash_sign": int(table.profile.str.contains("#", regex=False).sum()),
        "sha256": {str(x): sha(x) for x in (artifact, source/"user_data.csv", source/"full_user_data.csv")},
        "graph_counts": graph_counts}
    ref_rows = json.loads((args.topology/"metrics.json").read_text())
    rows, receipts, contexts = [], [], []
    for stream in manifest["streams"]:
        directory = Path(prior["cache_roots"]["covid_political/"+stream])/"covid_political"
        labels = input_labels(directory, "covid_political")
        ids_all, center_all, subgraph_all, support_all, raw_logits = [], [], [], [], []
        for bi, fingerprint in enumerate(labels["cache"]["batch_sha256"]):
            batch = torch.load(directory/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(batch) != fingerprint: raise ValueError("Cached batch changed")
            g = batch[0]
            real = g.global_node_ids >= 0
            torch.testing.assert_close(g.x[real], graph["x"][g.global_node_ids[real]], atol=0, rtol=0)
            centers = g.ptr[:-1]
            center_ids = g.global_node_ids[centers]
            y = batch[2].argmax(1)
            torch.testing.assert_close(graph["y"][center_ids].long(), g.task_label_map[g.task_id_per_sample, y].long(), atol=0, rtol=0)
            q = query_mask(batch)
            center = flags[center_ids]
            subgraph = torch.zeros((len(centers), len(DETECTORS)), dtype=torch.long)
            subgraph.index_add_(0, g.batch[real], flags[g.global_node_ids[real]].long())
            subgraph = subgraph > 0
            support = torch.zeros_like(subgraph)
            for ep in range(4):
                belongs = g.task_id_per_sample == ep
                support[belongs] = subgraph[belongs & ~q].any(0)
                for di, detector in enumerate(DETECTORS):
                    contexts.append({"stream": stream, "batch": bi, "episode": 4*bi+ep, "detector": detector,
                        "support_centers": int((belongs & ~q).sum()),
                        "cue_support_centers": int(center[belongs & ~q, di].sum()),
                        "cue_support_subgraphs": int(subgraph[belongs & ~q, di].sum())})
            ids_all.append(center_ids[q]); center_all.append(center[q]); subgraph_all.append(subgraph[q]); support_all.append(support[q])
            raw_logits.append(episode_probe(g.x[centers], batch, "prototype"))
        ids, center, subgraph, support = [torch.cat(x) for x in (ids_all, center_all, subgraph_all, support_all)]
        masks = {d: subset_masks(center[:, i], subgraph[:, i], support[:, i]) for i, d in enumerate(DETECTORS)}
        # Saved logits already carry the complete input hashes; check every label,
        # mapping and episode ID, then independently reproduce ALL reused metrics.
        for mid in manifest["models"]:
            path = args.topology/"predictions/covid_political"/stream/(mid+".pt")
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for k in ("local_y", "mapping", "episode_ids"):
                torch.testing.assert_close(saved["labels"][k], labels[k], atol=0, rtol=0)
            if saved["labels"]["cache"]["batch_sha256"] != labels["cache"]["batch_sha256"]:
                raise ValueError("Saved prediction inputs differ")
            error = 0.0
            for c in conditions:
                logits = saved["logits"][c]
                reference = [r for r in ref_rows if (r["target"],r["stream"],r["model_id"],r["query_condition"],r["support_condition"],r["draw"]) == ("covid_political",stream,mid,*c)]
                if len(reference) != 1: raise ValueError("Missing unique reference")
                reference = reference[0]
                got = evaluate_logits(logits, labels)
                error = max(error, max(abs(got[k]-reference[k]) for k in METRICS))
                if error > 1e-6: raise ValueError("Whole-stream historical metric mismatch")
                for detector, subsets in masks.items():
                    for subset, mask in subsets.items():
                        rows.append({"stream":stream,"model_id":mid,"source":reference["source"],"seed":reference["seed"],
                            "query_condition":c[0],"support_condition":c[1],"draw":c[2],"detector":detector,"subset":subset,
                            **metrics(logits, labels, mask, ids)})
            receipts.append({"stream":stream,"model_id":mid,"prediction_sha256":sha(path),"batch_count":32,
                             "all_input_hashes_and_features_match":True,"metric_max_abs_error":error})
        for detector, subsets in masks.items():
            for subset, mask in subsets.items():
                rows.append({"stream":stream,"model_id":"raw_prototype","source":"none","seed":None,
                    "query_condition":"intact","support_condition":"intact","draw":0,"detector":detector,"subset":subset,
                    **metrics(torch.cat(raw_logits),labels,mask,ids)})
        print(json.dumps({"stream":stream,"cells":len(rows),"elapsed_seconds":round(time.monotonic()-start,1)}),flush=True)
    expected = 2*(6*7+1)*len(DETECTORS)*len(SUBSETS)
    if len(rows) != expected or len(receipts) != 12: raise ValueError("Incomplete audit grid")
    for name, value in (("metrics",rows),("receipts",receipts),("provenance",provenance),("support_counts",contexts)):
        write_json(args.output/(name+".json"),value)
    write_json(args.output/"DONE.json",{"cells":len(rows),"receipts":len(receipts),"provenance_join_complete":True,
        "all_cached_features_and_labels_match":True,"historical_metric_parity":True,"new_model_forwards":0,
        "elapsed_seconds":time.monotonic()-start,"output_contains_raw_text_or_account_ids":False})


if __name__ == "__main__": main()
