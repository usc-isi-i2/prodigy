"""Direct repeated CPU/GPU forwards to audit a failed historical AUC parity cell."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import wandb

from experiments.run_shared_graph import write_json
from experiments.trainer import TrainerFS
from scripts.experiments.setup.icl_arch_matrix.common_protocol import build_classification_dataset, classification_targets
from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import resolved_params
from .analyze_mixture_predictions import input_labels, load_predictions, evaluate_logits
from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest


def score_and_truth(logits, labels):
    probs = F.softmax(logits, dim=1)
    yt, mapping = labels["local_y"], labels["mapping"]
    rows = torch.arange(len(yt))
    if labels["use_global"]:
        return probs[rows, (mapping == 1).long().argmax(1)], mapping[rows, yt]
    return probs[:, 1], yt


def compare_rank_pairs(cpu, other, labels):
    """Attribute AUC differences exactly to positive/negative score-order changes."""
    a, truth = score_and_truth(cpu, labels)
    b, truth_b = score_and_truth(other, labels)
    if not torch.equal(truth, truth_b) or set(truth.tolist()) != {0, 1}:
        raise ValueError("binary aligned labels required")
    pos, neg = torch.where(truth == 1)[0], torch.where(truth == 0)[0]
    ca = torch.sign(a[pos, None] - a[neg][None, :])
    cb = torch.sign(b[pos, None] - b[neg][None, :])
    changed = torch.nonzero(ca != cb)
    delta = float(((cb.double() - ca.double()) / 2).mean())
    actual = roc_auc_score(truth, b) - roc_auc_score(truth, a)
    if abs(delta - actual) > 1e-12:
        raise ValueError("cross-label pair changes do not reproduce AUC difference")
    rows = [{"positive_query": int(pos[i]), "negative_query": int(neg[j]),
             "cpu_scores": [float(a[pos[i]]), float(a[neg[j]])],
             "other_scores": [float(b[pos[i]]), float(b[neg[j]])],
             "cpu_order": int(ca[i, j]), "other_order": int(cb[i, j])}
            for i, j in changed.tolist()]
    return {"positive_queries": len(pos), "negative_queries": len(neg),
            "auc_half_pair_quantum": .5 / (len(pos) * len(neg)), "auc_other_minus_cpu": delta,
            "changed_cross_label_pairs": rows, "max_abs_logit_difference": float((cpu - other).abs().max()),
            "max_abs_probability_difference": float((a - b).abs().max()),
            "changed_decisions": int((cpu.argmax(1) != other.argmax(1)).sum())}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--cpu-replay", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("new output and exactly one explicitly selected GPU required")
    torch.set_num_threads(4)
    args.output.mkdir(parents=True)
    rows = json.loads(args.inventory.read_text())["models"]
    model = next(r for r in rows if r["model_id"] == args.model_id)
    state = torch.load(model["checkpoint"], map_location="cpu", weights_only=True)["model"]
    if model_digest(state) != model["weights_sha256"]:
        raise ValueError("checkpoint differs from verified historical inventory")
    directory = args.cpu_replay / args.target
    labels = input_labels(directory, args.target)
    metrics_rows = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
    cpu, cpu_metrics, cpu_audit = load_predictions(directory, model, labels, metrics_rows)
    reference = pd.read_csv(args.reference, sep="\t")
    prior = reference[(reference.model_id == args.model_id) & (reference.dataset == args.target)]
    if len(prior) != 1 or prior.iloc[0].episode_fingerprint != labels["cache"]["episode_fingerprint"]:
        raise ValueError("official reference cell not uniquely matched")
    protocol = json.loads((args.cpu_replay / "protocol.json").read_text())
    settings = SimpleNamespace(**protocol)
    settings.device = 0
    settings.state_root = settings.eval_state_root = str(args.output / "state")
    settings.log_root = str(args.output / "trainer_logs")
    settings.run_stamp = "mixture_numerical_audit"
    target = classification_targets(settings.catalog, include_facebook=True)[args.target]
    dataset, _, graph_path = build_classification_dataset(dataset_name=args.target, data_root=settings.data_root, target=target)
    params = resolved_params(settings, args.target, target, graph_path, None, "numerical_audit", 2500)
    torch.manual_seed(0)
    trainer = TrainerFS(dataset, params)
    trainer.model.load_state_dict(state, strict=True)
    trainer.model.eval()
    outputs, audits = {}, []
    # Direct production forward only: no observation hooks or stage probes.
    for device, repeats in (("cuda:0", 3), ("cpu", 1)):
        trainer.model.to(device)
        for repeat in range(repeats):
            values = []
            with torch.no_grad():
                for index in range(32):
                    batch = torch.load(directory / "batches" / f"batch_{index:03d}.pt", map_location="cpu", weights_only=False)
                    if batch_hash(batch) != labels["cache"]["batch_sha256"][index]:
                        raise ValueError("test input changed")
                    _, logits, _ = trainer.model(*clone_batch(batch, device))
                    values.append(logits.cpu())
            logits = torch.cat(values)
            if not torch.isfinite(logits).all() or logits.shape != cpu.shape:
                raise ValueError("invalid direct-forward outputs")
            name = f"{device.replace(':', '_')}_repeat{repeat}"
            outputs[name] = logits
            metrics = evaluate_logits(logits, labels)
            audits.append({"device": device, "repeat": repeat, **compare_rank_pairs(cpu, logits, labels), "metrics": metrics,
                "official_auc_abs_error": abs(metrics["roc_auc"] - prior.iloc[0].roc_auc),
                "official_decision_metric_max_abs_error": max(abs(metrics[m] - prior.iloc[0][m]) for m in ("accuracy", "f1"))})
    if model_digest(trainer.model.state_dict()) != model["weights_sha256"]:
        raise ValueError("model tensors changed during the numerical audit")
    if not torch.equal(outputs["cpu_repeat0"], cpu):
        raise ValueError("unhooked direct CPU forward did not exactly reproduce saved CPU logits")
    torch.save(outputs, args.output / "direct_logits.pt")
    write_json(args.output / "audit.json", {"model": model, "target": args.target,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "cpu_prediction_audit": cpu_audit, "cpu_metrics": cpu_metrics,
        "official_metrics": {m: float(prior.iloc[0][m]) for m in ("roc_auc", "accuracy", "f1")},
        "input_cache": labels["cache"], "direct_forwards": audits, "all_weights_unchanged": True,
        "cpu_direct_logits_bit_exact": True, "fitted_or_selected_outputs": False,
        "gpu_name": torch.cuda.get_device_name(0)})
    write_json(args.output / "DONE.json", {"direct_cpu_passes": 1, "direct_gpu_passes": 3,
        "all_inputs_and_weights_identical": True, "cpu_direct_logits_bit_exact": True})
    print(json.dumps(audits, indent=2), flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
