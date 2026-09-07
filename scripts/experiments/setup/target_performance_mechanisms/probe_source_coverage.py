"""Asymmetric target-query coverage by audited training-member bio features.

This measures a simulated center-feature exposure distribution, not the exact
historical stream, full graph overlap, or label-conditional task alignment.
Reference budgets and RNG are fixed without target labels. No weights are fit.
"""
import argparse
import json
from pathlib import Path
import subprocess

import torch
import torch.nn.functional as F

from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash
from scripts.experiments.setup.icl_arch_matrix.common_protocol import classification_targets


def select_references(features, counts, size, mode, seed):
    if len(features) < size or len(features) != len(counts) or torch.any(counts <= 0):
        raise ValueError("insufficient or invalid audited reference pool")
    rng = torch.Generator().manual_seed(seed)
    if mode == "uniform_unique":
        index = torch.randperm(len(features), generator=rng)[:size]
    elif mode == "exposure_weighted":
        index = torch.multinomial(counts.float(), size, replacement=True, generator=rng)
    else:
        raise ValueError(mode)
    # Repeated references cannot alter the maximum, but their count is recorded.
    unique = index.unique()
    return F.normalize(features[unique].float(), dim=1), unique


def nearest_cosine(query, reference, chunk=512):
    query = F.normalize(query.float(), dim=1)
    return torch.cat([(part @ reference.T).max(dim=1).values for part in query.split(chunk)])


def correlation(x, y):
    x, y = x.double().reshape(-1), y.double().reshape(-1)
    x, y = x - x.mean(), y - y.mean()
    denominator = x.norm() * y.norm()
    return float((x @ y) / denominator) if denominator > 0 else None


def double_center(x):
    return x - x.mean(0, keepdim=True) - x.mean(1, keepdim=True) + x.mean()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--audit-root", type=Path, required=True)
    p.add_argument("--replay-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-size", type=int, default=4096)
    p.add_argument("--seed", type=int, default=48391)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if not (args.audit_root / "DONE").exists():
        raise ValueError("source audit incomplete")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    protocol = json.loads((args.audit_root / "protocol.json").read_text())
    sources = protocol["source_names"]
    settings = {**vars(args), "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "historical_realized_stream": False, "center_features_only": True,
                "reference_sampling_uses_target_labels": False, "source_names": sources}
    (args.output / "protocol.json").write_text(json.dumps(settings, indent=2, default=str))
    metrics, associations = [], []
    with torch.no_grad():
        for target in classification_targets("docs/graph_catalog.json", include_facebook=True):
            roots = [root / target for root in args.replay_roots if (root / target / "random_init__baseline.pt").exists()]
            root = roots[-1]
            cache = json.loads((root / "cache.json").read_text())
            features, ids, labels = [], [], []
            for i, digest in enumerate(cache["batch_sha256"]):
                batch = torch.load(root / "batches" / f"batch_{i:03d}.pt", weights_only=False)
                if batch_hash(batch) != digest:
                    raise RuntimeError("replay cache changed")
                query = batch[5].reshape(-1, 2)[:, 0].bool()
                centers = batch[0].ptr[:-1][query]
                features.append(batch[0].x[centers])
                ids.append(batch[0].global_node_ids[centers])
                labels.append(batch[2][query].argmax(1))
            features, ids, labels = torch.cat(features), torch.cat(ids), torch.cat(labels)
            present = features.norm(dim=1) > 1e-8
            if not present.any():
                raise RuntimeError("no nonzero target features")
            # Correctness depends only on local class alignment, not global names.
            correct, losses = [], []
            for source in sources:
                export = torch.load(root / f"ss_{source}__baseline.pt", weights_only=False)
                if [x["batch_sha256"] for x in export] != cache["batch_sha256"]:
                    raise RuntimeError("model/input mismatch")
                logits = torch.cat([x["logits"]["full_model"] for x in export])
                correct.append((logits.argmax(1) == labels).float())
                losses.append(F.cross_entropy(logits, labels, reduction="none"))
            correct, losses = torch.stack(correct), torch.stack(losses)
            target_export = {"query_ids": ids, "features_present": present,
                             "source_names": sources, "correct": correct, "nll": losses,
                             "episode_fingerprint": cache["episode_fingerprint"], "coverage": {}}
            foreign = torch.tensor([s != target for s in sources])
            for policy in ("production_sorted", "uniform_walk_endpoints"):
                for mode in ("uniform_unique", "exposure_weighted"):
                    values, reference_ids = [], {}
                    for index, source in enumerate(sources):
                        audit = torch.load(args.audit_root / f"{source}__{policy}.pt", weights_only=False)
                        reference, selected = select_references(audit["features"], audit["counts"], args.reference_size, mode, args.seed + index)
                        sim = nearest_cosine(features, reference)
                        values.append(sim)
                        reference_ids[source] = audit["unique_member_ids"][selected]
                        row = {"dataset": target, "source": source, "policy": policy, "reference_mode": mode,
                               "reference_draws": args.reference_size, "unique_references": len(selected),
                               "queries": len(labels), "nonzero_queries": int(present.sum()),
                               "mean_nearest_cosine_nonzero": float(sim[present].mean()),
                               "near_identical_feature_fraction_nonzero": float((sim[present] >= 1 - 1e-6).float().mean()),
                               "correctness_correlation_nonzero": correlation(sim[present], correct[index, present]),
                               "nll_correlation_nonzero": correlation(sim[present], losses[index, present]),
                               "source_is_target": source == target, "episode_fingerprint": cache["episode_fingerprint"]}
                        metrics.append(row)
                    matrix = torch.stack(values)
                    # Remove each query's common difficulty and each source's
                    # mean quality. This is descriptive, not a causal estimate.
                    resid = double_center(matrix[foreign][:, present])
                    result = {"dataset": target, "policy": policy, "reference_mode": mode,
                              "foreign_sources": int(foreign.sum()), "nonzero_queries": int(present.sum()),
                              "double_centered_similarity_correctness_correlation": correlation(resid, double_center(correct[foreign][:, present])),
                              "double_centered_similarity_nll_correlation": correlation(resid, double_center(losses[foreign][:, present]))}
                    associations.append(result)
                    print(json.dumps(result), flush=True)
                    target_export["coverage"][f"{policy}/{mode}"] = {"similarity": matrix, "reference_ids": reference_ids}
            torch.save(target_export, args.output / f"{target}.pt")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output / "associations.json").write_text(json.dumps(associations, indent=2))
    (args.output / "DONE").write_text("Complete asymmetric audited-center coverage; descriptive only.\n")


if __name__ == "__main__":
    main()
