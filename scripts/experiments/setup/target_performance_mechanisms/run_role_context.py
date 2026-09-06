"""Complete query/support context factorial on all historical singleton CLS cells."""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from experiments.run_shared_graph import write_json
from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_mixture_predictions import input_labels, evaluate_logits, METRICS
from .inspect_examples import localized_intervention
from .prepare_mixture_complementarity import TARGETS
from .probe_cached_inputs import input_summaries
from .replay import batch_hash
from .role_context import VARIANTS, role_predictions, query_mask
from .run_episode_cardinality import choose_root, make_model
from .verify_member_training import model_digest


def query_metadata(batch, record):
    g = batch[0]
    q = query_mask(batch)
    _, topology, text = input_summaries(batch)
    centers = g.ptr[:-1]
    nodes = (g.ptr[1:] - g.ptr[:-1] - 2).long()
    incoming = torch.bincount(g.edge_index[1], minlength=len(g.x))[centers]
    outgoing = torch.bincount(g.edge_index[0], minlength=len(g.x))[centers]
    return {"global_node_id": g.global_node_ids[centers][q], "context_nodes": nodes[q],
            "incoming": incoming[q], "outgoing": outgoing[q], "center_context_cosine": text[q, 2],
            "raw_center_prediction": record["logits"]["raw_center/ridge"].argmax(1),
            "raw_context_prediction": record["logits"]["raw_context/ridge"].argmax(1)}


def cohorts(meta):
    has = meta["context_nodes"] > 0
    inc, out = meta["incoming"], meta["outgoing"]
    agree = meta["raw_center_prediction"] == meta["raw_context_prediction"]
    return {"all": torch.ones_like(has), "no_context": ~has, "has_context": has,
            "incoming_dominant": inc > out, "outgoing_dominant": out > inc,
            "balanced_nonzero_degree": (inc == out) & (inc > 0),
            "raw_center_context_agree": has & agree, "raw_center_context_disagree": has & ~agree}


def paired_counts(baseline, changed, y, mask, global_labels=None):
    n = int(mask.sum())
    bc, cc = baseline.argmax(1) == y, changed.argmax(1) == y
    fixed = int((~bc & cc & mask).sum())
    broken = int((bc & ~cc & mask).sum())
    base_correct = int((bc & mask).sum())
    altered_correct = int((cc & mask).sum())
    if altered_correct - base_correct != fixed - broken:
        raise ValueError("paired correctness accounting failed")
    delta = changed.softmax(1).gather(1, y[:, None])[:, 0] - baseline.softmax(1).gather(1, y[:, None])[:, 0]
    result = {"queries": n, "baseline_correct": base_correct, "altered_correct": altered_correct,
            "errors_fixed": fixed, "errors_introduced": broken,
            "delta_accuracy": (fixed-broken)/n if n else None,
            "fraction_baseline_errors_fixed": fixed/(n-base_correct) if n > base_correct else None,
            "fraction_baseline_correct_broken": broken/base_correct if base_correct else None,
            "mean_delta_label_probability": float(delta[mask].mean()) if n else None}
    if global_labels is not None:
        result["by_global_class"] = {}
        for label in global_labels[mask].unique(sorted=True).tolist():
            selected = mask & (global_labels == label)
            result["by_global_class"][str(label)] = {
                "queries": int(selected.sum()), "baseline_correct": int((selected & bc).sum()),
                "altered_correct": int((selected & cc).sum()),
                "errors_fixed": int((selected & ~bc & cc).sum()),
                "errors_introduced": int((selected & bc & ~cc).sum())}
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--original-roots", type=Path, nargs="+", required=True)
    p.add_argument("--fresh-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available() or not 1 <= args.threads <= 8:
        raise ValueError("new output, hidden GPUs, bounded CPU threads required")
    torch.set_num_threads(args.threads)
    ids = {"ss_" + s for s in SOURCES}
    choices = {(stream, target): choose_root(getattr(args, stream + "_roots"), target, ids)
               for stream in ("original", "fresh") for target in sorted(TARGETS)}
    if args.dry_run:
        print(json.dumps({"models": len(ids), "targets": len(TARGETS), "streams": 2,
                          "variants": VARIANTS, "full_model_cells": 720, "new_training": False}))
        return
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "original_roots": list(map(str, args.original_roots)), "fresh_roots": list(map(str, args.fresh_roots)),
        "models": sorted(ids), "targets": sorted(TARGETS), "variants": VARIANTS,
        "motivation": "Exploratory extension of the first 24 outcome-selected qualitative cases to all queries and all singleton sources.",
        "fixed_in_advance": "Report all targets/models/conditions, both streams; no best-condition selection. Decompose errors fixed and introduced, not fixes alone.",
        "interventions": "2x2 query/support context-feature replacement and 2x2 query/support background-edge removal, plus query center zeroing.",
        "prediction": "Ukraine COVID-Political dependence on query context should recur on both streams; other source/target benefits are exploratory.",
        "cohort_definitions": "context zero/positive; center incoming versus outgoing count; raw center/context ridge argmax agreement among nonempty contexts. No labels define cohorts.",
        "seeds": 1, "new_training": False, "query_input_unchanged_in_support_only_conditions": True,
        "uncertainty": "Descriptive paired counts and two episode streams; no independence assumption across recurring nodes, episodes, source models, or streams.",
        "parity_atol": 1e-5, "threads": args.threads})
    rows, paired, inventory, input_inventory = [], [], [], []
    verified_batches = 0
    with torch.no_grad():
        for (stream, target), (root, full) in choices.items():
            directory = root / target
            protocol = json.loads((root / "protocol.json").read_text())
            if protocol["eval_episode_seed_offset"] != (0 if stream == "original" else 100003):
                raise ValueError("wrong episode stream")
            labels = input_labels(directory, target)
            global_labels = labels["mapping"][torch.arange(len(labels["local_y"])), labels["local_y"]]
            batches = [torch.load(directory / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False) for bi in range(32)]
            for bi, batch in enumerate(batches):
                if batch_hash(batch) != labels["cache"]["batch_sha256"][bi]:
                    raise ValueError("input changed after validation")
            input_inventory.append({"stream": stream, "target": target, "root": str(directory), **labels["cache"]})
            reference_raw = None
            for model_id, prior in sorted(full.items()):
                if prior["sources"] != [model_id.removeprefix("ss_")] or Path(prior["checkpoint"]).name != "state_dict_2500.ckpt":
                    raise ValueError("wrong source or checkpoint step")
                state = torch.load(prior["checkpoint"], map_location="cpu", weights_only=True)["model"]
                digest = model_digest(state)
                if digest != prior["weights_sha256"]:
                    raise ValueError("checkpoint/embedding identity mismatch")
                model = make_model(protocol, target, labels["cache"]["graph_path"], batches[0][0].x.shape[1], state)
                saved = torch.load(directory / f"{model_id}__baseline.pt", map_location="cpu", weights_only=False)
                if len(saved) != 32 or prior["episode_fingerprint"] != labels["cache"]["episode_fingerprint"]:
                    raise ValueError("incomplete/mismatched saved cell")
                predictions = {v: [] for v in VARIANTS}
                parity = []
                metas = []
                for bi, (batch, record) in enumerate(zip(batches, saved)):
                    if record["batch"] != bi or record["batch_sha256"] != labels["cache"]["batch_sha256"][bi]:
                        raise ValueError("cached prediction input/order mismatch")
                    values, checks = role_predictions(model, batch, record["embeddings"]["U1_pre_meta"], verify_direct=bi == 0)
                    torch.testing.assert_close(values["baseline"], record["logits"]["full_model"], rtol=0, atol=0)
                    verified_batches += 1
                    for v in VARIANTS:
                        if not torch.isfinite(values[v]).all():
                            raise ValueError("nonfinite output")
                        predictions[v].append(values[v])
                    if bi == 0:
                        qidx = query_mask(batch).nonzero().flatten().tolist()
                        for variant, local in (("features_query", "query_context_replaced_by_center"),
                                               ("edges_query", "query_edges_removed"), ("center_zero_query", "query_center_zeroed")):
                            for qi in (0, len(qidx)-1):
                                _, one, _ = model(*localized_intervention(batch, qidx[qi], local))
                                torch.testing.assert_close(one[qi], values[variant][qi], rtol=0, atol=1e-5)
                                checks[f"independent_query/{variant}/{qi}"] = float((one[qi]-values[variant][qi]).abs().max())
                    parity.append(checks)
                    if reference_raw is None:
                        metas.append(query_metadata(batch, record))
                if reference_raw is None:
                    meta = {k: torch.cat([part[k] for part in metas]) for k in metas[0]}
                    reference_raw = {decoder: torch.cat([record["logits"][decoder] for record in saved])
                                     for decoder in ("raw_center/ridge", "raw_context/ridge")}
                else:
                    for decoder, values in reference_raw.items():
                        torch.testing.assert_close(values, torch.cat([r["logits"][decoder] for r in saved]), rtol=0, atol=0)
                outputs = {v: torch.cat(predictions[v]) for v in VARIANTS}
                common = {"stream": stream, "target": target, "model_id": model_id, "source": prior["sources"][0],
                          "foreign_source": prior["sources"][0] != target,
                          "checkpoint": prior["checkpoint"], "weights_sha256": digest,
                          "episode_fingerprint": prior["episode_fingerprint"], "queries": len(labels["local_y"]), "episodes": 128}
                for variant, values in outputs.items():
                    scores = evaluate_logits(values, labels)
                    if variant == "baseline" and max(abs(scores[k]-prior[k]) for k in METRICS) > 1e-6:
                        raise ValueError("baseline metric parity failed")
                    rows.append({**common, "variant": variant, **scores})
                    for cohort, mask in cohorts(meta).items():
                        paired.append({"stream": stream, "target": target, "model_id": model_id,
                                       "variant": variant, "cohort": cohort,
                                       **paired_counts(outputs["baseline"], values, labels["local_y"], mask, global_labels)})
                if model_digest(model.state_dict()) != digest:
                    raise ValueError("model parameters or buffers mutated")
                out = args.output / "predictions" / stream / target
                out.mkdir(parents=True, exist_ok=True)
                torch.save({"logits": outputs, "labels": labels, "query_metadata": meta,
                            "weights_sha256": digest, "parity_checks": parity}, out / f"{model_id}.pt")
                inventory.append({**common, "baseline_bit_exact_batches": 32,
                                  "direct_factorization_checks": sum(len(r) for r in parity),
                                  "max_factorization_error": max(max(r.values()) for r in parity),
                                  "prediction_file": str(out / f"{model_id}.pt")})
                write_json(args.output / "metrics.json", rows)
                write_json(args.output / "paired_cohorts.json", paired)
                write_json(args.output / "inventory.json", inventory)
                print(json.dumps({"stream": stream, "target": target, "model": model_id,
                                  "full_cells": len(rows), "max_parity_error": inventory[-1]["max_factorization_error"]}), flush=True)
            write_json(args.output / "input_inventory.json", input_inventory)
    if len(rows) != 720 or len(inventory) != 90 or verified_batches != 2880 or len(paired) != 5760:
        raise ValueError("incomplete full grid")
    write_json(args.output / "DONE.json", {"cells": len(rows), "models_target_stream": len(inventory),
        "paired_cohort_cells": len(paired), "baseline_bit_exact_batches": verified_batches,
        "all_weights_unchanged": True, "all_baseline_metrics_reproduced": True,
        "max_factorization_error": max(r["max_factorization_error"] for r in inventory)})


if __name__ == "__main__":
    main()
