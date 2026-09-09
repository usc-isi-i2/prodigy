"""Relate native/foreign NM outcomes to cached pre-metagraph source coverage.

This is an outcome-blind geometry construction followed by an outcome audit.
Every checkpoint defines its own representation space. Within that space, each
query is compared with balanced deterministic reference banks from HK and
Ukraine. Exact query identities are excluded from same-graph reference banks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGETS = ("cp_hk", "ukr_rus")
MODELS = ("hk", "ukr")
NATIVE = {"cp_hk": "hk", "ukr_rus": "ukr"}
OTHER = {"cp_hk": "ukr_rus", "ukr_rus": "cp_hk"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def deterministic_rank(node_id, salt):
    raw = f"{salt}:{int(node_id)}".encode()
    return hashlib.sha256(raw).digest()


def load_target(root, target):
    rows = pd.read_csv(root / target / "stage_predictions_private.csv")
    rows = rows[["episode", "sample", "query", "model", "native_correct",
                 "native_prediction", "encoded_correct", "encoded_prediction"]]
    rows["episode"] = rows.episode.astype(str)
    rows["model"] = rows.model.astype(str)
    if len(rows) != 122880 or rows.duplicated(["episode", "sample", "model"]).any():
        raise ValueError(f"unexpected stage rows for {target}")

    extracted = []
    file_hashes = []
    for batch_index in range(16):
        path = root / target / f"batch_{batch_index:03d}_embeddings_private.pt"
        file_hashes.append({"file": path.name, "sha256": digest(path)})
        packed = torch.load(path, map_location="cpu", weights_only=False)
        for model in MODELS:
            for episode_index, episode_data in packed["models"][model].items():
                episode_index = int(episode_index)
                label = f"{batch_index}:{episode_index}"
                slots = np.asarray([slot for slot in range(210) if slot % 7 >= 3])
                samples = episode_index * 210 + slots
                episode_tensor = episode_data["inputs"][0]
                values = episode_tensor[torch.as_tensor(slots)].float().numpy()
                extracted.append(pd.DataFrame({
                    "episode": label,
                    "sample": samples,
                    "model": model,
                    "embedding": list(values),
                }))
    embeddings = pd.concat(extracted, ignore_index=True)
    if len(embeddings) != 122880 or embeddings.duplicated(["episode", "sample", "model"]).any():
        raise ValueError(f"unexpected embedding rows for {target}")
    joined = rows.merge(embeddings, on=["episode", "sample", "model"], validate="one_to_one")
    if len(joined) != 122880:
        raise ValueError(f"stage/embedding join failed for {target}")
    return joined, file_hashes


def reference_bank(frame, model, size, salt):
    subset = frame[frame.model.eq(model)][["query", "embedding"]].copy()
    subset = subset.drop_duplicates("query", keep="first")
    subset["rank"] = subset["query"].map(lambda value: deterministic_rank(value, salt))
    subset = subset.sort_values("rank").head(size)
    if len(subset) != size:
        raise ValueError(f"reference bank only has {len(subset)} nodes")
    matrix = np.stack(subset.embedding).astype(np.float32)
    matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return subset["query"].to_numpy(dtype=np.int64), matrix


def topk_similarity(queries, query_ids, reference_ids, reference, k, chunk_size,
                    exclude_matching_ids):
    result = np.empty(len(queries), dtype=np.float32)
    for start in range(0, len(queries), chunk_size):
        stop = min(start + chunk_size, len(queries))
        q = queries[start:stop].astype(np.float32, copy=True)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        scores = q @ reference.T
        if exclude_matching_ids:
            same = query_ids[start:stop, None] == reference_ids[None, :]
        else:
            same = np.zeros(scores.shape, dtype=bool)
        scores[same] = -np.inf
        available = (~same).sum(axis=1)
        if (available < k).any():
            raise ValueError("too few reference nodes after identity exclusion")
        selected = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        result[start:stop] = selected.mean(axis=1)
    return result


def auc_binary(y, score):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if not n1 or not n0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def weighted_auc_binary(y, score, weight):
    """Weighted probability that a positive score exceeds a negative score."""
    table = pd.DataFrame({"y": np.asarray(y, dtype=bool),
                          "score": np.asarray(score, dtype=float),
                          "weight": np.asarray(weight, dtype=float)})
    total_positive = table.loc[table.y, "weight"].sum()
    total_negative = table.loc[~table.y, "weight"].sum()
    if total_positive <= 0 or total_negative <= 0:
        return None
    favorable = 0.0
    negative_below = 0.0
    for _, group in table.groupby("score", sort=True):
        positive = group.loc[group.y, "weight"].sum()
        negative = group.loc[~group.y, "weight"].sum()
        favorable += positive * (negative_below + 0.5 * negative)
        negative_below += negative
    return float(favorable / (total_positive * total_negative))


def summaries(frame):
    metrics = ["target_similarity", "other_similarity", "coverage_gap"]
    result = {"rows": len(frame), "unique_queries": int(frame["query"].nunique()), "cohorts": {}}
    for cohort, subset in frame.groupby("cohort", sort=True):
        item = {"rows": len(subset), "unique_queries": int(subset["query"].nunique())}
        for metric in metrics:
            item[metric] = {
                "occurrence_mean": float(subset[metric].mean()),
                "node_weighted_mean": float(subset.groupby("query")[metric].mean().mean()),
            }
        result["cohorts"][cohort] = item
    result["correctness_auc"] = {
        metric: auc_binary(frame.native_correct, frame[metric]) for metric in metrics
    }
    node_weight = 1.0 / frame.groupby("query")["query"].transform("size")
    result["node_balanced_correctness_auc"] = {
        metric: weighted_auc_binary(frame.native_correct, frame[metric], node_weight)
        for metric in metrics
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--reference-replicate", type=int, default=0)
    args = parser.parse_args()
    if args.reference_size < args.top_k + 1 or args.top_k < 1:
        raise ValueError("reference-size must exceed top-k")
    args.output_root.mkdir(parents=True, exist_ok=False)

    data, hashes = {}, {}
    for target in TARGETS:
        data[target], hashes[target] = load_target(args.input_root, target)

    banks = {}
    for model in MODELS:
        for source in TARGETS:
            banks[(model, source)] = reference_bank(
                data[source], model, args.reference_size,
                f"nm-coverage-v2:{args.reference_replicate}:{model}:{source}"
            )

    output_rows = []
    report = {
        "complete": True,
        "protocol": "nm_pre_metagraph_source_coverage_v2",
        "reference_size": args.reference_size,
        "top_k": args.top_k,
        "chunk_size": args.chunk_size,
        "reference_replicate": args.reference_replicate,
        "geometry_uses_outcomes": False,
        "identity_exclusion": "Exact query id excluded from same-source reference bank.",
        "reference_selection": "SHA256 ordering of unique query ids; first occurrence embedding; balanced banks.",
        "input_hashes": hashes,
        "targets": {},
    }
    for target in TARGETS:
        wide = data[target].pivot(index=["episode", "sample", "query"], columns="model",
                                  values=["native_correct", "native_prediction"])
        native, foreign = NATIVE[target], NATIVE[OTHER[target]]
        native_ok = wide[("native_correct", native)].astype(bool)
        foreign_ok = wide[("native_correct", foreign)].astype(bool)
        cohort = np.select(
            [native_ok & foreign_ok, native_ok & ~foreign_ok, ~native_ok & foreign_ok],
            ["both_correct", "native_only", "foreign_only"], default="both_wrong"
        )
        cohort_map = pd.Series(cohort, index=wide.index)
        report["targets"][target] = {"native_model": native, "foreign_model": foreign, "models": {}}
        for model in MODELS:
            frame = data[target][data[target].model.eq(model)].copy()
            queries = np.stack(frame.embedding).astype(np.float32)
            ids = frame["query"].to_numpy(dtype=np.int64)
            target_ids, target_bank = banks[(model, target)]
            other_ids, other_bank = banks[(model, OTHER[target])]
            frame["target_similarity"] = topk_similarity(
                queries, ids, target_ids, target_bank, args.top_k, args.chunk_size, True
            )
            frame["other_similarity"] = topk_similarity(
                queries, ids, other_ids, other_bank, args.top_k, args.chunk_size, False
            )
            frame["coverage_gap"] = frame.target_similarity - frame.other_similarity
            keys = pd.MultiIndex.from_frame(frame[["episode", "sample", "query"]])
            frame["cohort"] = cohort_map.loc[keys].to_numpy()
            frame["target"] = target
            output_rows.append(frame[["target", "episode", "sample", "query", "model", "native_correct",
                                      "native_prediction", "cohort", "target_similarity",
                                      "other_similarity", "coverage_gap"]])
            report["targets"][target]["models"][model] = summaries(frame)

    private = pd.concat(output_rows, ignore_index=True)
    private_path = args.output_root / "coverage_rows_private.csv.gz"
    private.to_csv(private_path, index=False, compression="gzip")
    report["private_rows"] = len(private)
    report["private_rows_sha256"] = digest(private_path)
    (args.output_root / "receipt.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
