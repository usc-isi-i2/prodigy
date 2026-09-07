"""Extract and hydrate exact matched-query examples selected before text review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def feature_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def locate_occurrence(batch_counts, occurrence):
    offset = 0
    for batch_index, count in enumerate(batch_counts):
        if occurrence < offset + count:
            return batch_index, occurrence - offset
        offset += count
    raise IndexError(occurrence)


def extract(args):
    import torch
    import torch.nn.functional as functional
    from scripts.experiments.setup.target_performance_mechanisms.replay import batch_hash

    if args.output.exists():
        raise ValueError("choose a new output file")
    export = torch.load(args.export / "original.pt", map_location="cpu", weights_only=False)
    graph = torch.load(export["receipts"]["graph_path"], map_location="cpu", weights_only=False)
    batch_counts = list(map(int, export["labels"]["batch_counts"]))
    candidates = list(csv.DictReader(args.candidates.open()))
    selected, counts = [], {}
    for row in candidates:
        group = row["selection"]
        if counts.get(group, 0) >= args.per_group:
            continue
        counts[group] = counts.get(group, 0) + 1
        selected.append(row)

    nodes = {}
    cases = []

    def node_record(row):
        row = int(row)
        key = str(row)
        if key not in nodes:
            label = int(graph["y"][row])
            nodes[key] = {
                "graph_row": row,
                "source_id": str(graph["user_ids"][row]),
                "dataset_label_id": label,
                "dataset_label": graph["label_names"][label] if label >= 0 else "unlabeled",
                "feature_sha256": feature_hash(graph["x"][row]),
            }
        return key

    loaded_batches = {}
    for row in selected:
        occurrence = int(row["occurrence_index"])
        batch_index, query_index = locate_occurrence(batch_counts, occurrence)
        if batch_index not in loaded_batches:
            batch_path = Path(export["receipts"]["cache_root"]) / "batches" / f"batch_{batch_index:03d}.pt"
            batch = torch.load(batch_path, map_location="cpu", weights_only=False)
            if batch_hash(batch) != export["receipts"]["batch_sha256"][batch_index]:
                raise ValueError("cached batch hash mismatch")
            loaded_batches[batch_index] = batch
        batch = loaded_batches[batch_index]
        sampled = batch[0]
        query_mask = batch[5].reshape(-1, batch[2].shape[1])[:, 0].bool()
        sample_index = int(query_mask.nonzero().flatten()[query_index])
        centers = sampled.global_node_ids[sampled.ptr[:-1]]
        center = int(centers[sample_index])
        if center != int(row["global_node_id"]):
            raise ValueError("candidate/query identity mismatch")
        task = int(sampled.task_id_per_sample[sample_index])
        mapping = list(map(int, sampled.task_label_map[task].tolist()))
        local_label = int(batch[2][sample_index].argmax())
        if int(graph["y"][center]) != mapping[local_label]:
            raise ValueError("query label mapping mismatch")
        start, end = int(sampled.ptr[sample_index]), int(sampled.ptr[sample_index + 1])
        context = sampled.global_node_ids[start + 1:end]
        context = context[context >= 0]
        edge_mask = sampled.batch[sampled.edge_index[0]] == sample_index
        edges = sampled.global_node_ids[sampled.edge_index[:, edge_mask]].T.tolist()
        support_indices = ((sampled.task_id_per_sample == task) & ~query_mask).nonzero().flatten()
        similarities = functional.cosine_similarity(
            graph["x"][center][None], graph["x"][centers[support_indices]], dim=1,
        )
        supports = []
        for support_index, similarity in zip(support_indices.tolist(), similarities.tolist()):
            support_center = int(centers[support_index])
            support_local_label = mapping.index(int(graph["y"][support_center]))
            supports.append({
                "node": node_record(support_center), "local_label": support_local_label,
                "cosine_to_query": float(similarity),
            })
        supports.sort(key=lambda item: item["cosine_to_query"], reverse=True)
        cases.append({
            "selection": row["selection"], "occurrence_index": occurrence,
            "episode_id": int(row["episode_id"]), "batch": batch_index,
            "query_index": query_index, "sample_index": sample_index,
            "node": node_record(center), "local_label": local_label,
            "dataset_label_mapping": [graph["label_names"][label] for label in mapping],
            "outcome": row["outcome"],
            "predictions": {
                name: {
                    "decoders": {
                        decoder: {
                            "predicted_local_label": int(logits[occurrence].argmax()),
                            "correct": int(logits[occurrence].argmax()) == local_label,
                            "logits": list(map(float, logits[occurrence].tolist())),
                        }
                        for decoder, logits in export["models"][name]["logits"].items()
                    },
                    "calibrated_true_loss": float(row[f"loss_{suffix}_cal"]),
                    "calibrated_prob1": float(row[f"prob1_{suffix}_cal"]),
                }
                for name, suffix in zip(export["models"], ("b", "c"))
            },
            "query_context": [node_record(item) for item in context.tolist()],
            "sampled_edges_graph_rows": edges,
            "supports_by_similarity": supports,
        })
    output = {
        "target": export["target"], "models": list(export["models"]),
        "selection": "loss-extreme distinct query IDs selected from original stream before text inspection",
        "per_group": args.per_group, "cases": cases, "nodes": nodes,
        "graph_path": export["receipts"]["graph_path"],
        "cached_inputs_verified": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "nodes": len(nodes), "groups": counts}))


def hydrate(args):
    import numpy as np
    import pyarrow.parquet as parquet

    if args.output.exists():
        raise ValueError("choose a new output file")
    document = json.loads(args.input.read_text())
    graph_root = args.data_root / document["target"]
    embedding_root = graph_root / "bio_embeddings/gte-multilingual-base/version=v001"
    observations = parquet.read_table(
        embedding_root / "user_bio_observations.parquet", columns=["userid", "bio_hash"]
    ).to_pylist()
    user_hash = {}
    for row in observations:
        if row["userid"] in user_hash and user_hash[row["userid"]] != row["bio_hash"]:
            raise ValueError("ambiguous bio for one source ID")
        user_hash[row["userid"]] = row["bio_hash"]
    texts = {row["bio_hash"]: row["normalized_bio_text"] for row in parquet.read_table(
        embedding_root / "bio_texts.parquet", columns=["bio_hash", "normalized_bio_text"]
    ).to_pylist()}
    locations = {row["bio_hash"]: row for row in parquet.read_table(
        embedding_root / "bio_embedding_index.parquet"
    ).to_pylist()}
    profile_columns = [
        "account_id", "account_name", "account_handle", "page_description", "page_category",
        "page_admin_top_country", "page_created_date", "verified", "subscriber_count", "source_post_count",
    ]
    profiles = {str(row["account_id"]): row for row in parquet.read_table(
        graph_root / "tables/2022-02-24_to_2022-06-29_page_reference_v1/page_profiles.parquet",
        columns=profile_columns,
    ).to_pylist()}
    shards = {}
    verified = 0
    for node in document["nodes"].values():
        source_id = node["source_id"]
        bio_hash = user_hash.get(source_id)
        location = locations.get(bio_hash)
        if location is None:
            vector = np.zeros(768, dtype=np.float32)
            text = None
        else:
            shard = location["embedding_shard"]
            if shard not in shards:
                shards[shard] = np.load(embedding_root / shard, mmap_mode="r")
            vector = np.asarray(shards[shard][location["embedding_row"]], dtype=np.float32)
            text = texts[bio_hash]
        if hashlib.sha256(vector.tobytes()).hexdigest() != node["feature_sha256"]:
            raise ValueError(f"embedding identity mismatch for {source_id}")
        profile = profiles[source_id]
        node.update({name: profile[name] for name in profile_columns if name != "account_id"})
        node["normalized_description"] = text
        node["bio_hash"] = bio_hash
        node["text_embedding_identity_verified"] = True
        verified += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2, default=str) + "\n")
    print(json.dumps({"cases": len(document["cases"]), "nodes": len(document["nodes"]),
                      "embedding_identities_verified": verified}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    extract_parser = subparsers.add_parser("extract")
    extract_parser.add_argument("--export", type=Path, required=True)
    extract_parser.add_argument("--candidates", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--per-group", type=int, default=6)
    hydrate_parser = subparsers.add_parser("hydrate")
    hydrate_parser.add_argument("--input", type=Path, required=True)
    hydrate_parser.add_argument("--output", type=Path, required=True)
    hydrate_parser.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    args = parser.parse_args()
    {"extract": extract, "hydrate": hydrate}[args.mode](args)


if __name__ == "__main__":
    main()
