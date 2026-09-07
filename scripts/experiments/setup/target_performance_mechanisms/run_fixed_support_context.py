"""Cache natural same-account support contexts, then replay the frozen source controls."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import pandas as pd
import torch
from torch_geometric.data import Batch

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.icl_arch_matrix.common_protocol import build_classification_dataset, classification_targets
from .analyze_mixture_predictions import input_labels, METRICS
from .episode_cardinality import cached_meta_forward
from .fixed_support_context import (draw_seed, graph_hash, support_draw, replace_support_contexts,
    verify_replacement, context_comparison, seal_plan, score_context_draws)
from .natural_support import frozen_contract
from .prepare_mixture_complementarity import TARGETS
from .replay import batch_hash, clone_batch
from .role_context import query_mask, changed_embedding
from .run_episode_cardinality import make_model
from .run_natural_support import query_labels_prefix
from .verify_member_training import model_digest


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def check_graph(graph, dataset, sorted_edges):
    ids = graph.global_node_ids
    real = ids >= 0
    torch.testing.assert_close(graph.x[real], dataset.graph.x[ids[real]], rtol=0, atol=0)
    if graph.x[~real].count_nonzero():
        raise ValueError("nonzero synthetic pooling feature")
    edges = ids[graph.edge_index]
    if edges.numel():
        if (edges < 0).any():
            raise ValueError("background edge touches synthetic node")
        encoded = edges[0]*dataset.graph.num_nodes+edges[1]
        pos = torch.searchsorted(sorted_edges, encoded)
        if (pos == len(sorted_edges)).any() or not torch.equal(sorted_edges[pos], encoded):
            raise ValueError("cached/sampled directed edge absent from target graph")


def context_cache(args, inputs):
    plans = []
    for target in sorted(set(r["target"] for r in inputs)):
        caches = [r for r in inputs if r["target"] == target]
        parent_protocol = json.loads((Path(caches[0]["root"]).parent/"protocol.json").read_text())
        target_config = classification_targets(parent_protocol["catalog"], include_facebook=True)[target]
        dataset, _, graph_path = build_classification_dataset(dataset_name=target,
            data_root=parent_protocol["data_root"], target=target_config)
        if any(str(graph_path) != r["graph_path"] for r in caches):
            raise ValueError("target graph path changed")
        source_digest = file_hash(graph_path)
        sorted_edges = (dataset.graph.edge_index[0]*dataset.graph.num_nodes+dataset.graph.edge_index[1]).sort().values
        for cache in caches:
            root, stream = Path(cache["root"]), cache["stream"]
            if graph_path.stat().st_mtime_ns > (root/"cache.json").stat().st_mtime_ns:
                raise ValueError("target graph artifact newer than original cache")
            dest = args.output/stream/target
            dest.mkdir(parents=True)
            rows, contexts = [], []
            for bi in range(1 if args.smoke else 32):
                batch = torch.load(root/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                if batch_hash(batch) != cache["batch_sha256"][bi]:
                    raise ValueError("original input hash changed")
                q, g = query_mask(batch), batch[0]
                y, tasks = batch[2].argmax(1), g.task_id_per_sample
                for ep in range(4):
                    for c in range(2):
                        if int(((tasks == ep) & (~q) & (y == c)).sum()) != 10:
                            raise ValueError("not ordinary two-way 10-shot inputs")
                check_graph(g, dataset, sorted_edges)
                graphs = g.to_data_list()
                original = Batch.from_data_list([graphs[i] for i in torch.where(~q)[0].tolist()])
                restored = replace_support_contexts(batch, original)
                if batch_hash(restored) != cache["batch_sha256"][bi]:
                    raise ValueError("identity reconstruction changes full input")
                del restored
                draws = [original]
                for draw in range(2 if args.smoke else 8):
                    seed = None if draw == 0 else draw_seed(stream, target, bi, draw)
                    if draw:
                        draws.append(support_draw(dataset, batch, seed))
                    sampled = draws[-1]
                    check_graph(sampled, dataset, sorted_edges)
                    changed = replace_support_contexts(batch, sampled)
                    audit = verify_replacement(batch, changed)
                    comparisons = context_comparison(original, sampled)
                    contexts.extend({"batch": bi, "draw": draw, "support_slot": i, **r} for i, r in enumerate(comparisons))
                    rows.append({"batch": bi, "draw": draw, "seed": seed, "input_sha256": cache["batch_sha256"][bi],
                        "support_graph_sha256": graph_hash(sampled), "support_centers": sampled.global_node_ids[sampled.ptr[:-1]].tolist(),
                        **audit})
                    if bi == 0 and draw == 1:
                        tampered = clone_batch(batch)
                        tampered[2][q] = tampered[2][q].flip(1)
                        tampered[0].x[q[tampered[0].batch]] = 777
                        other = support_draw(dataset, tampered, seed)
                        if graph_hash(other) != graph_hash(sampled):
                            raise ValueError("query truth/features changed support sample")
                        del tampered, other
                    del changed
                torch.save(draws, dest/f"supports_{bi:03d}.pt")
                if batch_hash(batch) != cache["batch_sha256"][bi]:
                    raise ValueError("original batch mutated")
                print(f"Cached {stream}/{target} batch {bi}", flush=True)
                del draws, batch, graphs, g
            plan = {"stream": stream, "target": target, "root": str(dest), "original": cache,
                "graph_file_sha256": source_digest, "graph_features_sha256": model_digest({"x": dataset.graph.x}),
                "graph_edges_sha256": model_digest({"edges": dataset.graph.edge_index}),
                "graph_file_predates_original_cache": True, "every_cached_and_new_feature_and_edge_verified": True,
                "same_sampler": {"hops": dataset.neighbor_sampler.num_hops, "hop_sizes": dataset.neighbor_sampler.hop_sizes,
                    "node_limit": dataset.neighbor_sampler.limit, "bidirectional": dataset.bidirectional},
                "query_truth_and_feature_tamper_passed": True, "rows": rows}
            plan["sha256"] = seal_plan(plan)
            write_json(dest/"plan.json", plan)
            pd.DataFrame(contexts).to_csv(dest/"contexts.csv", index=False)
            plans.append(plan)
            write_json(args.output/"plans.json", plans)
        if file_hash(graph_path) != source_digest:
            raise ValueError("target graph artifact changed during export")
        del dataset, sorted_edges
    write_json(args.output/"DONE.json", {"complete": True, "phase": "cache", "smoke": args.smoke,
        "plans": len(plans), "support_draw_batches": sum(len(p["rows"]) for p in plans),
        "support_positions": 80*sum(len(p["rows"]) for p in plans), "no_models_or_target_predictions_used": True})


def evaluate(args, arms):
    done = json.loads((args.cache/"DONE.json").read_text())
    if not done["complete"] or done["phase"] != "cache" or done["smoke"] != args.smoke:
        raise ValueError("complete matching-scope context cache required")
    plans = json.loads((args.cache/"plans.json").read_text())
    if len(plans) != (1 if args.smoke else 10):
        raise ValueError("incomplete input grid")
    ref = pd.read_csv(args.data/"member_replay_cells.csv")
    ref = ref[(ref.policy == "lowest_sorted") & (ref.decoder == "full_model")]
    previous_done = json.loads((args.reference/"DONE.json").read_text())
    if not previous_done["complete"] or previous_done["smoke"] or previous_done["model_target_stream_cells"] != 60:
        raise ValueError("complete prior natural-support baselines required")
    metrics, episodes, cohorts, audits = [], [], [], []
    with torch.no_grad():
        for plan in plans:
            if seal_plan({k: v for k, v in plan.items() if k != "sha256"}) != plan["sha256"]:
                raise ValueError("context plan hash changed")
            stream, target = plan["stream"], plan["target"]
            original = Path(plan["original"]["root"])
            protocol = json.loads((original.parent/"protocol.json").read_text())
            labels = input_labels(original, target)
            if labels["cache"]["episode_fingerprint"] != plan["original"]["episode_fingerprint"]:
                raise ValueError("original episode identity changed")
            batches = [torch.load(original/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
                for bi in range(1 if args.smoke else 32)]
            context = torch.cat([(b[0].ptr[1:]-b[0].ptr[:-1]-2)[query_mask(b)] for b in batches])
            labels = query_labels_prefix(labels, len(context))
            spec = {(r["batch"], r["draw"]): r for r in plan["rows"]}
            for arm in arms:
                model_id = arm["model_id"]
                prior = ref[(ref.stream == stream) & (ref.dataset == target) & (ref.model_id == model_id)]
                if len(prior) != 1 or prior.iloc[0].weights_sha256 != arm["final_sha256"] or prior.iloc[0].checkpoint != arm["checkpoint"]:
                    raise ValueError("prior model identity mismatch")
                saved = torch.load(args.reference/"predictions"/stream/target/f"{model_id}.pt", map_location="cpu", weights_only=False)
                if saved["weights_sha256"] != arm["final_sha256"] or saved["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]:
                    raise ValueError("saved baseline identity mismatch")
                state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
                if model_digest(state) != arm["final_sha256"]:
                    raise ValueError("source checkpoint changed")
                model = make_model(protocol, target, plan["original"]["graph_path"], batches[0][0].x.shape[1], state)
                frozen_contract(model)
                parts = [[] for _ in range(2 if args.smoke else 8)]
                direct_errors, invariant, offset = [], 0, 0
                for bi, batch in enumerate(batches):
                    if batch_hash(batch) != plan["original"]["batch_sha256"][bi]:
                        raise ValueError("cached original input changed")
                    q = query_mask(batch)
                    draws = torch.load(Path(plan["root"])/f"supports_{bi:03d}.pt", map_location="cpu", weights_only=False)
                    pre, baseline = changed_embedding(model, batch, "baseline")
                    torch.testing.assert_close(baseline, saved["baseline_logits"][offset:offset+int(q.sum())], rtol=0, atol=0)
                    base_x, base_z = cached_meta_forward(model, batch, pre)
                    torch.testing.assert_close(base_z, baseline, rtol=0, atol=0)
                    if len(draws) != len(parts):
                        raise ValueError("missing context draws")
                    for draw, supports in enumerate(draws):
                        if graph_hash(supports) != spec[bi, draw]["support_graph_sha256"]:
                            raise ValueError("saved support graph changed")
                        changed = replace_support_contexts(batch, supports)
                        verify_replacement(batch, changed)
                        if draw == 0:
                            if batch_hash(changed) != batch_hash(batch):
                                raise ValueError("original draw not identical")
                            new_pre, direct = pre, baseline
                        else:
                            new_pre, direct = changed_embedding(model, changed, "baseline")
                        torch.testing.assert_close(new_pre[q], pre[q], rtol=0, atol=0)
                        x, z = cached_meta_forward(model, batch, new_pre)
                        torch.testing.assert_close(x[q], base_x[q], rtol=0, atol=0)
                        torch.testing.assert_close(z, direct, rtol=0, atol=0)
                        direct_errors.append(float((z-direct).abs().max()))
                        parts[draw].append(z)
                        invariant += 1
                        del changed
                    offset += int(q.sum())
                    del draws
                predictions = torch.stack([torch.cat(p) for p in parts])
                common = {"stream": stream, "target": target, "model_id": model_id, "source": arm["source"], "seed": arm["seed"],
                    "weights_sha256": arm["final_sha256"], "checkpoint": arm["checkpoint"],
                    "episode_fingerprint": labels["cache"]["episode_fingerprint"], "plan_sha256": plan["sha256"], "queries": offset}
                rows, ep_rows, cohort_rows = score_context_draws(predictions, labels, context, common)
                if not args.smoke and any(abs(rows[0][m]-prior.iloc[0][m]) > 1e-6 for m in METRICS):
                    raise ValueError("baseline metrics mismatch")
                metrics.extend(rows)
                episodes.extend(ep_rows)
                cohorts.extend(cohort_rows)
                if model_digest(model.state_dict()) != arm["final_sha256"]:
                    raise ValueError("model weights/buffers mutated")
                dest = args.output/"predictions"/stream/target
                dest.mkdir(parents=True, exist_ok=True)
                torch.save({**common, "predictions": predictions, "labels": labels, "context_nodes": context}, dest/f"{model_id}.pt")
                audits.append({**common, "baseline_batches_bit_exact": len(batches), "query_invariance_batches": invariant,
                    "direct_suffix_checks": len(direct_errors), "maximum_direct_error": max(direct_errors),
                    "all_support_ids_and_labels_unchanged": True, "all_weights_unchanged": True})
                write_json(args.output/"metrics.json", metrics)
                write_json(args.output/"cohorts.json", cohorts)
                write_json(args.output/"audits.json", audits)
                pd.DataFrame(episodes).drop(columns=["checkpoint", "weights_sha256", "episode_fingerprint", "plan_sha256"]).to_csv(args.output/"episode_scores.csv", index=False)
                print(f"Completed {stream}/{target}/{model_id}: {len(audits)} cells", flush=True)
            del batches
    expected = 2 if args.smoke else 60
    draws, nb = (2, 1) if args.smoke else (8, 32)
    if len(audits) != expected or len(metrics) != expected*(draws+1) or len(episodes) != expected*nb*4*draws:
        raise ValueError("incomplete model/target/stream grid")
    write_json(args.output/"DONE.json", {"complete": True, "phase": "evaluate", "smoke": args.smoke,
        "cells": len(audits), "metric_cells": len(metrics), "episode_draw_cells": len(episodes),
        "direct_suffix_checks": sum(a["direct_suffix_checks"] for a in audits),
        "all_query_inputs_and_vectors_bit_exact": True, "same_labeled_support_accounts": True,
        "no_optimizer_updates": True, "no_additional_labeled_accounts": True})


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--phase", choices=("cache", "evaluate"), required=True)
    p.add_argument("--data", type=Path, default=Path("scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data"))
    p.add_argument("--reference", type=Path, default=Path("/dataMeR1/phil/gfm/prodigy-mechanisms-role/log/natural_support_full_20260906"))
    p.add_argument("--cache", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.threads <= 8:
        raise ValueError("new output and bounded threads required")
    torch.set_num_threads(a.threads)
    if torch.cuda.is_available():
        raise ValueError("CPU-only diagnostic")
    arms = [r for r in json.loads((a.data/"member_training_verified/arms.json").read_text()) if r["policy"] == "lowest_sorted"]
    inputs = json.loads((a.data/"role_context_replay/input_inventory.json").read_text())
    if len(arms) != 6 or {(r["source"], r["seed"]) for r in arms} != {(s, i) for s in ("ukr_rus", "cp_hk") for i in range(3)}:
        raise ValueError("incomplete source/seed grid")
    if len(inputs) != 10 or {(r["stream"], r["target"]) for r in inputs} != {(s, t) for s in ("original", "fresh") for t in TARGETS}:
        raise ValueError("incomplete target/stream grid")
    if a.smoke:
        arms = [r for r in arms if r["seed"] == 0]
        inputs = [r for r in inputs if (r["stream"], r["target"]) == ("original", "covid_political")]
    if a.dry_run:
        print(json.dumps({"phase": a.phase, "models": len(arms), "caches": len(inputs), "draws": 2 if a.smoke else 8,
            "batches_per_cache": 1 if a.smoke else 32, "threads": a.threads, "new_labeled_accounts": 0}))
        return
    a.output.mkdir(parents=True)
    write_json(a.output/"protocol.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "phase": a.phase, "smoke": a.smoke, "cache": str(a.cache), "reference": str(a.reference), "threads": a.threads,
        "draws": 2 if a.smoke else 8, "sampling": "Original support context plus seven independent same-center draws; no query labels/features used",
        "primary": "Hong Kong political query-probability variance exceeds Ukraine in every seed/stream under same-account support-context draws",
        "limits": "Previously studied targets, no additional labeled centers, multiple-pass extra inference compute, ensemble Jensen gain not an empirical success criterion"})
    try:
        context_cache(a, inputs) if a.phase == "cache" else evaluate(a, arms)
    except BaseException:
        write_json(a.output/"FAILED.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
