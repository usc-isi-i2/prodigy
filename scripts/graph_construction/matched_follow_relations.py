"""Build a same-node follow graph after verifying converted relation semantics.

This is an induced matched-node diagnostic, not the untouched public benchmark.
The public original sample must verify friend=following and follow=follower.
"""
import argparse
import collections
import copy
import hashlib
import json
from pathlib import Path


def canonical_pair(source, relation, target):
    if relation == "friend":
        return source, target
    if relation == "follow":
        return target, source
    raise ValueError("Expected a verified user-user relation")


def verify_sample(sample, observed):
    checked = collections.Counter()
    for user in sample:
        neighbors = user.get("neighbor")
        if neighbors is None:
            continue
        uid = "u" + str(user["ID"])
        for original, converted in (("following", "friend"), ("follower", "follow")):
            expected = {"u"+str(x) for x in neighbors[original]}
            if observed.get((uid, converted), set()) != expected:
                raise ValueError("Converted relation differs from original sample")
            checked[converted] += len(expected)
    if any(checked[r] == 0 for r in ("friend", "follow")):
        raise ValueError("Both directions require nonempty original evidence")
    return dict(checked)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import torch
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--edges", type=Path, required=True)
    p.add_argument("--original-sample", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--name", default="twibot20_follow_matched")
    args = p.parse_args()
    if torch.cuda.is_available() or args.output.exists():
        raise ValueError("Hidden GPUs and unused output required")
    torch.set_num_threads(2)
    catalog = json.loads(args.catalog.read_text())
    relative = args.output.resolve().relative_to(Path(catalog["data_root"]).resolve())
    graph = torch.load(args.reference, map_location="cpu", weights_only=False)
    sample = json.loads(args.original_sample.read_text())
    sample_ids = {"u"+str(r["ID"]) for r in sample if r.get("neighbor") is not None}
    ids = graph["user_ids"]
    u2i = {u:i for i,u in enumerate(ids)}
    if len(u2i) != len(ids):
        raise ValueError("Duplicate reference identities")
    observed = collections.defaultdict(set)
    edges, counts, retained = set(), collections.Counter(), collections.Counter()
    with args.edges.open() as f:
        if f.readline().strip() != "source_id,relation,target_id":
            raise ValueError("Unexpected relation schema")
        for line in f:
            if ",post," in line:
                counts["post"] += 1
                continue
            a, relation, b = line.strip().split(",")
            counts[relation] += 1
            if a in sample_ids:
                observed[a,relation].add(b)
            a, b = canonical_pair(a, relation, b)
            if a in u2i and b in u2i:
                retained[relation] += 1
                edges.add((u2i[a], u2i[b]))
    checked = verify_sample(sample, observed)
    edge_index = torch.tensor(sorted(edges), dtype=torch.long).T.contiguous()
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("No retained edges")
    edge_attr = torch.ones((len(edges), 1), dtype=torch.float32)
    result = dict(graph)
    for key in ("edge_index_views", "target_edge_index_views", "edge_attr_views",
                "edge_attr_feature_names_views", "static_split_stats", "benchmark_target_stats"):
        result.pop(key, None)
    result.update(edge_index=edge_index, edge_attr=edge_attr,
                  edge_attr_feature_names=["follow_relation"])
    result["data"] = copy.copy(graph["data"])
    result["data"].edge_index = edge_index
    result["data"].edge_attr = edge_attr
    # The wrapper's tensor fields and the reference payload must remain unchanged.
    if not torch.equal(graph["data"].edge_index, graph["edge_index"]):
        raise ValueError("Reference wrapper mutated")
    for field in ("x", "y"):
        if not torch.equal(result[field], graph[field]):
            raise ValueError("Reference features/labels changed")
    provenance = dict(reference=str(args.reference), reference_sha256=sha(args.reference),
        edges_sha256=sha(args.edges), original_sample_sha256=sha(args.original_sample),
        original_sample_url="https://raw.githubusercontent.com/BunsenFeng/TwiBot-20/main/TwiBot-20_sample.json",
        verified_original_sample_edges=checked, input_relation_counts=dict(counts),
        retained_relation_rows=dict(retained), nodes=len(ids), edges=len(edges),
        direction="follower_to_followed; friend as written, follow reversed",
        same_node_order_features_labels_splits=True,
        scope="induced matched-node diagnostic; not unmodified public benchmark")
    result["matched_relation_provenance"] = provenance
    registry_key = "graphs"
    if registry_key not in catalog or not isinstance(catalog[registry_key], list):
        raise ValueError("Unexpected graph catalog registry")
    if any(r.get("canonical_name") == args.name for r in catalog[registry_key]):
        raise ValueError("Catalog name already exists")
    entry = dict(canonical_name=args.name, dataset_key=args.name, kind="diagnostic",
        relative_path=str(relative), metadata_path=str(relative)+".meta.json",
        statistics={"nodes":len(ids), "edges":len(edges)},
        features={"node_feature_dim":int(graph["x"].shape[1]), "node_features":"Exact reference tensors"},
        tasks={"supported":["node_classification"], "unavailable":["neighbor_matching", "static_link_prediction"]},
        construction=provenance, default_eval=False)
    # Register before producing the graph. This catalog edit is local to this worktree.
    catalog[registry_key].append(entry)
    args.catalog.write_text(json.dumps(catalog, indent=2)+"\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    provenance["artifact_sha256"] = sha(args.output)
    Path(str(args.output)+".meta.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
