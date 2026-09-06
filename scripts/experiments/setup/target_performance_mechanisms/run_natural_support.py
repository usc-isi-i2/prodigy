"""Finite natural-support replay with frozen, query-blind selection."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd
import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from .analyze_mixture_predictions import input_labels, evaluate_logits, ensemble_scores, METRICS
from .natural_support import frozen_contract, meta_inputs, meta_logits, support_cv_loss, support_plan, replace_graphs
from .prepare_mixture_complementarity import TARGETS
from .replay import batch_hash
from .role_context import query_mask, changed_embedding
from .run_episode_cardinality import make_model
from .verify_member_training import model_digest


def query_labels_prefix(labels, n):
    return {**labels, **{k: labels[k][:n] for k in ("local_y", "mapping", "episode_ids")}}


def score_draws(predictions, cv, labels, context, common):
    """Selection depends only on cv, never any query label or query feature."""
    selected_draw = cv.argmin(0)
    ep, truth = labels["episode_ids"], labels["local_y"]
    selected = predictions[selected_draw[ep], torch.arange(len(ep))]
    rows = [{**common, "condition": "draw", "draw": draw, **evaluate_logits(z, labels)}
            for draw, z in enumerate(predictions)]
    rows.append({**common, "condition": "support_cv_selected", "draw": -1, **evaluate_logits(selected, labels)})
    ensembles = ensemble_scores(predictions, labels)
    rows.append({**common, "condition": "probability_ensemble", "draw": -1, **ensembles["probability"][0]})
    loss = F.cross_entropy(predictions.reshape(-1, 2), truth.repeat(len(predictions)), reduction="none").reshape(len(predictions), -1)
    correct = predictions.argmax(2).eq(truth[None])
    episode_rows = []
    for episode in ep.unique(sorted=True).tolist():
        idx = ep.eq(episode)
        for draw in range(len(predictions)):
            episode_rows.append({**common, "episode": episode, "draw": draw, "queries": int(idx.sum()),
                                 "support_cv_nll": float(cv[draw, episode]), "query_nll": float(loss[draw, idx].mean()),
                                 "query_correct": int(correct[draw, idx].sum()),
                                 "selected": bool(selected_draw[episode] == draw)})
    # Local-label orientation is fixed across draws for every query, so its
    # probability variance is invariant to a global binary label permutation.
    probs = predictions.softmax(2)[:, :, 1]
    cohort_rows = []
    for name, mask in (("all", torch.ones_like(context, dtype=torch.bool)), ("no_context", context.eq(0)), ("has_context", context.gt(0))):
        if not mask.any():
            continue
        count_right = correct[:, mask].sum(0)
        cohort_rows.append({**common, "cohort": name, "queries": int(mask.sum()),
                           "mean_probability_variance": float(probs[:, mask].var(0, unbiased=False).mean()),
                           "correctness_flips": int(((count_right > 0) & (count_right < len(predictions))).sum()),
                           "always_correct": int(count_right.eq(len(predictions)).sum()), "always_wrong": int(count_right.eq(0).sum())})
    return rows, episode_rows, cohort_rows, selected_draw


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path("scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("new output and hidden GPUs required")
    torch.set_num_threads(4)
    arms = [a for a in json.loads((args.data / "member_training_verified/arms.json").read_text()) if a["policy"] == "lowest_sorted"]
    if len(arms) != 6 or {(a["source"], a["seed"]) for a in arms} != {(s, n) for s in ("ukr_rus", "cp_hk") for n in range(3)}:
        raise ValueError("all six original-policy source/seed controls required")
    inputs = json.loads((args.data / "role_context_replay/input_inventory.json").read_text())
    if len(inputs) != 10 or {(r["stream"], r["target"]) for r in inputs} != {(s, t) for s in ("original", "fresh") for t in TARGETS}:
        raise ValueError("complete saved two-stream/five-target input grid required")
    ref = pd.read_csv(args.data / "member_replay_cells.csv")
    ref = ref[ref.policy.eq("lowest_sorted") & ref.decoder.eq("full_model")]
    if len(ref) != 60:
        raise ValueError("all 60 independent saved model baselines required")
    if args.smoke:
        arms = [a for a in arms if a["source"] == "cp_hk" and a["seed"] == 0]
        inputs = [r for r in inputs if r["stream"] == "original" and r["target"] == "covid_political"]
    draws = 2 if args.smoke else 8
    if args.dry_run:
        print(json.dumps(dict(models=len(arms), caches=len(inputs), draws=draws, cpu_threads=4, new_training=False,
                              recipient_batches=1 if args.smoke else 32, cv_folds=10)))
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        smoke=args.smoke, draws=draws, models=[a["model_id"] for a in arms], data=str(args.data), new_training=False,
        selection="Minimum ten-balanced-fold support-only held-out NLL; fixed ties; no target query input or label used.",
        primary="Hong Kong political support sensitivity exceeds Ukraine in every seed/stream; support-CV selection improves Hong Kong political NLL against mean random-draw NLL in every seed/stream.",
        sampling="Uniform eligible unique support center identity, then cached occurrence; no recipient-episode occurrences or recipient-query identities; no replacement within class.",
        limits="Larger labeled candidate pool; not a benchmark-fair 10-shot remedy; existing studied streams, not independent unseen domains.",
        direct_parity_atol=1e-5, threads=4))
    metrics, episodes, cohorts, audits, plan_inventory = [], [], [], [], []
    with torch.no_grad():
        for cache in sorted(inputs, key=lambda r: (r["stream"], r["target"])):
            stream, target = cache["stream"], cache["target"]
            directory = Path(cache["root"])
            labels = input_labels(directory, target)
            if any(labels["cache"][k] != cache[k] for k in ("episode_fingerprint", "graph_path", "batch_sha256")):
                raise ValueError("saved cache receipt differs")
            protocol = json.loads((directory.parent / "protocol.json").read_text())
            batches = [torch.load(directory / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False) for bi in range(32)]
            for bi, b in enumerate(batches):
                if batch_hash(b) != cache["batch_sha256"][bi]:
                    raise ValueError("input changed")
                for actual, expected in zip(meta_inputs(b[2].argmax(1), query_mask(b), b[0].task_id_per_sample, len(b[1])), b[3:6]):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            seed = 9102026 + (100003 if stream == "fresh" else 0) + sum((i+1)*ord(c) for i, c in enumerate(target))
            mappings, plan = support_plan(batches, draws, seed)
            plans = args.output / "plans"
            plans.mkdir(exist_ok=True)
            write_json(plans / f"{stream}_{target}.json", plan)
            plan_inventory.append(dict(stream=stream, target=target, plan_sha256=plan["sha256"],
                                       pool_unique_identities=plan["pool_unique_identities"], pool_occurrences=plan["pool_occurrences"]))
            all_graphs = [g for batch in batches for g in batch[0].to_data_list()]
            batch_limit = 1 if args.smoke else 32
            context = torch.cat([torch.bincount(b[0].batch, weights=(b[0].global_node_ids >= 0).float())[query_mask(b)].long() - 1 for b in batches[:batch_limit]])
            for arm in arms:
                prior = ref[(ref.stream == stream) & (ref.dataset == target) & (ref.model_id == arm["model_id"])].iloc[0]
                if (prior.checkpoint != arm["checkpoint"] or prior.weights_sha256 != arm["final_sha256"]
                        or prior.source != arm["source"] or prior.seed != arm["seed"]
                        or prior.episode_fingerprint != cache["episode_fingerprint"]):
                    raise ValueError("independent baseline identity differs")
                state = torch.load(arm["checkpoint"], map_location="cpu", weights_only=True)["model"]
                if model_digest(state) != arm["final_sha256"]:
                    raise ValueError("verified checkpoint changed")
                model = make_model(protocol, target, cache["graph_path"], batches[0][0].x.shape[1], state)
                frozen_contract(model)
                pre_parts, baseline = [], []
                for b in batches:
                    pre, z = changed_embedding(model, b, "baseline")
                    pre_parts.append(pre)
                    baseline.append(z)
                baseline_logits = torch.cat(baseline)
                if max(abs(evaluate_logits(baseline_logits, labels)[m] - prior[m]) for m in METRICS) > 1e-6:
                    raise ValueError("baseline metrics differ from pre-existing reference")
                pre_pool = torch.cat(pre_parts)
                predictions, scores = [[] for _ in range(draws)], [[] for _ in range(draws)]
                direct_errors = []
                offset = 0
                for bi, b in enumerate(batches[:batch_limit]):
                    q, y, task = query_mask(b), b[2].argmax(1), b[0].task_id_per_sample
                    base_x, base_z = meta_logits(model, pre_parts[bi], b[1], y, q, task)
                    torch.testing.assert_close(base_z[q], baseline[bi], rtol=0, atol=0)
                    for draw, mapping in enumerate(mappings[bi]):
                        pre = pre_pool[mapping]
                        torch.testing.assert_close(pre[q], pre_parts[bi][q], rtol=0, atol=0)
                        x, z = meta_logits(model, pre, b[1], y, q, task)
                        torch.testing.assert_close(x[q], base_x[q], rtol=0, atol=0)
                        cv = support_cv_loss(model, pre, b[1], y, q, task)
                        if not torch.isfinite(z).all() or not torch.isfinite(cv).all():
                            raise ValueError("nonfinite predictions")
                        if bi == 0 and draw == 0:
                            _, direct, _ = model(*replace_graphs(b, all_graphs, mapping))
                            torch.testing.assert_close(direct, z[q], rtol=0, atol=1e-5)
                            direct_errors.append(float((direct - z[q]).abs().max()))
                            # Real-input selector leakage guard: neither target query
                            # vectors nor its labels may affect support-only CV scores.
                            tampered = pre.clone()
                            tampered[q] = 777
                            changed_y = y.clone()
                            changed_y[q] = 1 - changed_y[q]
                            other = support_cv_loss(model, tampered, b[1], changed_y, q, task)
                            torch.testing.assert_close(cv, other, rtol=0, atol=0)
                        predictions[draw].append(z[q])
                        scores[draw].append(cv)
                    offset += int(q.sum())
                prediction_tensor = torch.stack([torch.cat(parts) for parts in predictions])
                cv_tensor = torch.stack([torch.cat(parts) for parts in scores])
                local_labels = query_labels_prefix(labels, offset)
                common = dict(stream=stream, target=target, model_id=arm["model_id"], source=arm["source"], seed=arm["seed"],
                              weights_sha256=arm["final_sha256"], checkpoint=arm["checkpoint"], episode_fingerprint=cache["episode_fingerprint"],
                              plan_sha256=plan["sha256"], queries=offset)
                rows, ep_rows, cohort_rows, selected = score_draws(prediction_tensor, cv_tensor, local_labels, context, common)
                metrics.extend(rows)
                metrics.append({**common, "condition": "baseline", "draw": -1, **evaluate_logits(baseline_logits[:offset], local_labels)})
                episodes.extend(ep_rows)
                cohorts.extend(cohort_rows)
                if model_digest(model.state_dict()) != arm["final_sha256"]:
                    raise ValueError("model mutated")
                outputs = args.output / "predictions" / stream / target
                outputs.mkdir(parents=True, exist_ok=True)
                torch.save(dict(predictions=prediction_tensor, cv_nll=cv_tensor, selected_draw=selected, labels=local_labels,
                                context_nodes=context, baseline_logits=baseline_logits[:offset], **common), outputs / (arm["model_id"] + ".pt"))
                audits.append({**common, "baseline_batches_bit_exact": batch_limit, "query_invariance_batches": batch_limit * draws,
                               "support_cv_query_blind": True, "direct_checks": len(direct_errors), "max_direct_error": max(direct_errors),
                               "all_weights_unchanged": True})
                write_json(args.output / "metrics.json", metrics)
                write_json(args.output / "audits.json", audits)
                write_json(args.output / "cohorts.json", cohorts)
                print(json.dumps({"stream": stream, "target": target, "model": arm["model_id"], "completed_cells": len(audits),
                                  "max_direct_error": max(direct_errors)}), flush=True)
            # Repeated model provenance is in audits/metrics; keep the episode
            # table compact without dropping any query loss or selected draw.
            ep_table = pd.DataFrame(episodes).drop(columns=["checkpoint", "weights_sha256", "episode_fingerprint", "plan_sha256"])
            ep_table.to_csv(args.output / "episode_scores.csv", index=False)
            write_json(args.output / "plan_inventory.json", plan_inventory)
    expected = 1 if args.smoke else 60
    if len(audits) != expected or len(metrics) != expected * (draws + 3) or len(episodes) != expected * (4 if args.smoke else 128) * draws:
        raise ValueError("incomplete natural-support grid")
    write_json(args.output / "DONE.json", dict(complete=True, smoke=args.smoke, model_target_stream_cells=len(audits),
        metric_cells=len(metrics), episode_draw_cells=len(episodes), plans=len(plan_inventory), draws=draws,
        direct_checks=sum(a["direct_checks"] for a in audits), maximum_direct_error=max(a["max_direct_error"] for a in audits),
        all_query_vectors_unchanged=True, all_selectors_query_blind=True, all_weights_unchanged=True))


if __name__ == "__main__":
    main()
