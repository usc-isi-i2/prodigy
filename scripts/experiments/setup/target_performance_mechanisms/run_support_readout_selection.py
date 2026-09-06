"""One fixed support-only readout test using existing model and embedding caches."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, evaluate_logits
from .natural_support import frozen_contract, meta_logits
from .replay import batch_hash
from .role_context import query_mask
from .run_episode_cardinality import make_model
from .run_natural_support import query_labels_prefix
from .support_readout_selection import fold_pairs, prototype_oof, full_oof, select_predictions
from .verify_member_training import model_digest


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path("scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data"))
    p.add_argument("--matched", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8:
        raise ValueError("new CPU-only output and bounded threads required")
    torch.set_num_threads(args.threads)
    done = json.loads((args.matched / "DONE.json").read_text())
    if not done["complete"] or done["smoke"] or done["metric_cells"] != 290:
        raise ValueError("complete matched-family replay required")
    inputs = json.loads((args.data / "role_context_replay/input_inventory.json").read_text())
    sources = sorted(pd.read_csv(args.matched / "cells.csv").query('source != "none"').source.unique())
    if len(inputs) != 10 or len(sources) != 9:
        raise ValueError("all existing source/target/stream cells required")
    if args.smoke:
        inputs = [r for r in inputs if r["target"] == "covid_political" and r["stream"] == "original"]
        sources = ["cp_hk"]
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", {"smoke": args.smoke,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "selection": "Maximum AUC on 20 balanced leave-pair-out support margins; ties raw, prototype, full.",
        "scale": "Positive RMS of the candidate's 20 OOF support margins; zero RMS uses one; no intercept.",
        "candidates": {"prodigy": ["raw", "prototype", "full"], "samgpt": ["raw", "prototype"]},
        "primary": "Foreign-source mean AUC across political/Facebook/TwiBot: selected_scaled exceeds every fixed_scaled rule in each family and each stream.",
        "query_labels_used_for_selection": False, "additional_labeled_accounts": 0,
        "new_pretraining": False, "new_encoder_passes": 0, "inherited_unlabeled_transductive_context": True,
        "novel_selection_algorithm_claim": False, "matched_replay": str(args.matched),
        "original_scores_and_unscaled_selection_retained": True})
    metrics, decisions, audits = [], [], []
    with torch.no_grad():
        for cache in inputs:
            stream, target = cache["stream"], cache["target"]
            root = Path(cache["root"])
            labels = input_labels(root, target)
            matched = torch.load(args.matched / "predictions" / stream / f"{target}.pt", map_location="cpu", weights_only=False)
            if labels["cache"]["episode_fingerprint"] != matched["labels"]["cache"]["episode_fingerprint"]:
                raise ValueError("query cache differs from the matched comparison")
            protocol = json.loads((root.parent / "protocol.json").read_text())
            limit = 1 if args.smoke else 32
            batches = [torch.load(root / "batches" / f"batch_{i:03d}.pt", map_location="cpu", weights_only=False) for i in range(limit)]
            raw_oof, pairs, qcount = [], [], 0
            for i, b in enumerate(batches):
                if batch_hash(b) != cache["batch_sha256"][i]:
                    raise ValueError("cached graph input changed")
                q, g = query_mask(b), b[0]
                y, tasks = b[2].argmax(1)[~q], g.task_id_per_sample[~q]
                pair = fold_pairs(y, tasks, 4)
                centers = g.global_node_ids[g.ptr[:-1]]
                torch.testing.assert_close(centers[q], matched["query_node_ids"][qcount:qcount+int(q.sum())], rtol=0, atol=0)
                for ep in range(4):
                    if len(centers[(g.task_id_per_sample == ep) & ~q].unique()) != 20:
                        raise ValueError("duplicate support center within episode")
                pairs.append(pair)
                raw_oof.append(prototype_oof(g.x[g.ptr[:-1]][~q], y, tasks, pair))
                qcount += int(q.sum())
            local_labels = query_labels_prefix(labels, qcount)
            prior_rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            for source in sources:
                prior = next(r for r in prior_rows if r["model_id"] == f"ss_{source}" and r["variant"] == "baseline" and r["decoder"] == "full_model")
                state = torch.load(prior["checkpoint"], map_location="cpu", weights_only=True)["model"]
                if model_digest(state) != prior["weights_sha256"]:
                    raise ValueError("PRODIGY checkpoint differs from cached embeddings")
                model = make_model(protocol, target, cache["graph_path"], batches[0][0].x.shape[1], state)
                frozen_contract(model)
                saved = torch.load(root / f"ss_{source}__baseline.pt", map_location="cpu", weights_only=False)
                native = torch.load(args.matched / "native_embeddings" / target / f"{source}.pt", map_location="cpu", weights_only=False)
                collected = {family: [] for family in ("prodigy", "samgpt")}
                offset = 0
                for bi, b in enumerate(batches):
                    q, g = query_mask(b), b[0]
                    y, tasks, query_tasks = b[2].argmax(1)[~q], g.task_id_per_sample[~q], g.task_id_per_sample[q]
                    if saved[bi]["batch_sha256"] != cache["batch_sha256"][bi]:
                        raise ValueError("cached embedding identity differs")
                    pre = saved[bi]["embeddings"]["U1_pre_meta"]
                    if bi == 0:
                        _, full = meta_logits(model, pre, b[1], b[2].argmax(1), q, g.task_id_per_sample)
                        torch.testing.assert_close(full[q], saved[bi]["logits"]["full_model"], rtol=0, atol=0)
                    pro_oof = {"raw": raw_oof[bi], "prototype": prototype_oof(pre[~q], y, tasks, pairs[bi]),
                        "full": full_oof(model, pre[~q], b[1], y, tasks, pairs[bi])}
                    centers = g.global_node_ids[g.ptr[:-1]][~q]
                    sam_oof = {"raw": raw_oof[bi], "prototype": prototype_oof(native["embeddings"][centers], y, tasks, pairs[bi])}
                    count = int(q.sum())
                    for family, oof in (("prodigy", pro_oof), ("samgpt", sam_oof)):
                        candidates = {"raw": matched["predictions"][("none", "raw_768_prototype")][offset:offset+count],
                            "prototype": matched["predictions"][(source, family + "_prototype")][offset:offset+count]}
                        if family == "prodigy":
                            candidates["full"] = matched["predictions"][(source, "prodigy_full")][offset:offset+count]
                        result = select_predictions(candidates, oof, y, tasks, query_tasks)
                        collected[family].append({**result, "oof": oof})
                        for ep in range(4):
                            chosen = list(candidates)[int(result["choice"][ep])]
                            for ci, candidate in enumerate(candidates):
                                decisions.append({"stream": stream, "target": target, "source": source, "family": family,
                                    "episode": 4*bi+ep, "candidate": candidate, "selected": chosen == candidate,
                                    "support_cv_auc": float(result["cv_auc"][ci, ep]), "support_margin_rms": float(result["cv_rms"][ci, ep])})
                    offset += count
                for family, records in collected.items():
                    names = ["raw", "prototype"] + (["full"] if family == "prodigy" else [])
                    outputs = {}
                    for scale in ("original", "scaled"):
                        z = torch.cat([r["fixed_"+scale] for r in records], dim=1)
                        outputs.update({f"{name}_{scale}": z[i] for i, name in enumerate(names)})
                        outputs["selected_"+scale] = torch.cat([r["selected_"+scale] for r in records])
                    common = {"stream": stream, "target": target, "source": source, "family": family,
                        "episode_fingerprint": cache["episode_fingerprint"]}
                    for condition, logits in outputs.items():
                        metrics.append({**common, "condition": condition, **evaluate_logits(logits, local_labels)})
                    dest = args.output / "predictions" / stream / target
                    dest.mkdir(parents=True, exist_ok=True)
                    torch.save({"outputs": outputs, "cv": records, "labels": local_labels,
                        "query_ids": matched["query_node_ids"][:qcount], "candidate_order": names,
                        "prodigy_checkpoint": prior["checkpoint"], "samgpt_receipt": native["receipt"], **common},
                        dest / f"{family}_{source}.pt")
                    audits.append({**common, "batches": limit, "support_labels_per_episode": 20,
                        "heldout_supports_per_episode": 20, "prodigy_suffix_first_batch_bit_exact": True,
                        "candidate_query_predictions_reused_from_matched_replay": True})
                if model_digest(model.state_dict()) != prior["weights_sha256"]:
                    raise ValueError("model state changed")
                pd.DataFrame(metrics).to_csv(args.output / "cells.csv", index=False)
                pd.DataFrame(decisions).to_csv(args.output / "decisions.csv", index=False)
                write_json(args.output / "audits.json", audits)
                print(f"Support readout selection {stream}/{target}/{source}: {len(audits)} family cells", flush=True)
                del saved, native, model, state
    if len(metrics) != (14 if args.smoke else 1260) or len(audits) != (2 if args.smoke else 180):
        raise ValueError("incomplete selection comparison")
    write_json(args.output / "DONE.json", {"complete": True, "smoke": args.smoke,
        "metric_cells": len(metrics), "family_source_target_stream_cells": len(audits),
        "support_cv_candidate_episode_rows": len(decisions), "new_encoder_passes": 0, "additional_labeled_accounts": 0})


if __name__ == "__main__":
    main()
