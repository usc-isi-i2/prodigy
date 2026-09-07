"""Reconstruct selected successes/errors from verified, immutable CLS episodes.

Extract under prodigy; hydrate under bio-embeddings-v001. Selection precedes text
inspection. This is a qualitative, outcome-conditioned audit, not a benchmark.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import subprocess


MODELS = ("ss_ukr_rus", "ss_cp_hk", "ss_covid")
STRATA = ("ukr_right_hk_wrong", "hk_right_ukr_wrong", "both_wrong_raw_right", "both_right_raw_wrong")
CONDITIONS = ("query_context_replaced_by_center", "query_center_zeroed", "query_edges_removed",
              "support_context_replaced_by_center", "support_edges_removed")


def stratum(y, ukr, hk, raw):
    if ukr == y and hk != y:
        return STRATA[0]
    if hk == y and ukr != y:
        return STRATA[1]
    if ukr != y and hk != y and raw == y:
        return STRATA[2]
    if ukr == y and hk == y and raw != y:
        return STRATA[3]
    return None


def localized_intervention(batch, sample, condition):
    """Change only specified query or same-episode supports, never truth labels."""
    import torch
    from .replay import clone_batch
    if condition not in CONDITIONS:
        raise ValueError(condition)
    b = clone_batch(batch)
    g = b[0]
    query = b[5].reshape(-1, b[2].shape[1])[:, 0].bool()
    if not query[sample]:
        raise ValueError("selected center must be a query")
    selected = torch.zeros_like(query)
    if condition.startswith("support_"):
        selected = (g.task_id_per_sample == g.task_id_per_sample[sample]) & ~query
    else:
        selected[sample] = True
    centers = g.ptr[:-1]
    if condition.endswith("context_replaced_by_center"):
        mask = selected[g.batch] & (g.global_node_ids >= 0)
        mask[centers] = False
        g.x[mask] = g.x[centers[g.batch[mask]]]
    elif condition == "query_center_zeroed":
        g.x[centers[sample]] = 0
    else:
        keep = ~selected[g.batch[g.edge_index[0]]]
        g.edge_index = g.edge_index[:, keep]
        if g.edge_attr is not None:
            g.edge_attr = g.edge_attr[keep]
    return b


def extract(args):
    import torch
    import torch.nn.functional as F
    from scripts.social_llm.generate_graph import read_user_data_csv
    from .replay import batch_hash, clone_batch
    from .run_episode_cardinality import make_model
    from .probe_cached_inputs import input_summaries, standardized_probe
    torch.set_num_threads(4)
    if torch.cuda.is_available() or args.output.exists():
        raise ValueError("hide GPUs and choose a new output")
    args.output.mkdir(parents=True)
    for target in args.targets:
        candidates = [r / target for r in args.roots if (r / target / "ss_cp_hk__baseline.pt").is_file()]
        if not candidates:
            raise FileNotFoundError(target)
        root = candidates[-1]
        cache = json.loads((root / "cache.json").read_text())
        if cache["episodes"] != 128 or len(cache["batch_sha256"]) != 32:
            raise ValueError("incomplete input stream")
        a = torch.load(cache["graph_path"], map_location="cpu", weights_only=False)
        names = a["label_names"]
        table = None
        if target == "covid_political":
            table = read_user_data_csv(str(args.data_root / "social_llm_data/covid/user_data.csv"))
            if list(map(int, a["user_ids"])) != list(range(len(table))):
                raise ValueError("CSV rows do not match graph identities")
            torch.testing.assert_close(torch.tensor(table.label_conservative.to_numpy()).long(), a["y"].long(), rtol=0, atol=0)
        records = {m: torch.load(root / f"{m}__baseline.pt", map_location="cpu", weights_only=False) for m in MODELS}
        metrics = [json.loads(s) for s in (root / "metrics.jsonl").read_text().splitlines()]
        checkpoints = {r["model_id"]: r["checkpoint"] for r in metrics if r["variant"] == "baseline" and r["decoder"] == "full_model"}
        nodes, cases, counts = {}, [], Counter()
        selected_episodes, selected_nodes = defaultdict(set), defaultdict(set)
        batches = {}

        def node(n):
            n = int(n)
            key = str(n)
            if key not in nodes:
                y = int(a["y"][n])
                nodes[key] = {"graph_row": n, "source_id": str(a["user_ids"][n]), "label_id": y,
                              "dataset_label": names[y] if y >= 0 else "unlabeled",
                              "feature_sha256": hashlib.sha256(a["x"][n].contiguous().numpy().tobytes()).hexdigest()}
                if table is not None:
                    nodes[key]["text"] = str(table.iloc[n].profile)
            return key

        for bi, digest in enumerate(cache["batch_sha256"]):
            batch = torch.load(root / "batches" / f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(batch) != digest or any(records[m][bi]["batch_sha256"] != digest for m in MODELS):
                raise ValueError("input/prediction hash mismatch")
            g = batch[0]
            real = g.global_node_ids >= 0
            torch.testing.assert_close(a["x"][g.global_node_ids[real]], g.x[real], rtol=0, atol=0)
            centers = g.ptr[:-1]
            ids = g.global_node_ids[centers]
            labels = batch[2].argmax(1)
            query = batch[5].reshape(-1, 2)[:, 0].bool()
            torch.testing.assert_close(a["y"][ids].long(), g.task_label_map[g.task_id_per_sample, labels].long(), rtol=0, atol=0)
            raw, topology, _ = input_summaries(batch)
            degree_logits = standardized_probe(topology[:, 3:4], batch)
            for qi, s in enumerate(query.nonzero().flatten().tolist()):
                y = int(labels[s])
                ukr, hk = [int(records[m][bi]["logits"]["full_model"][qi].argmax()) for m in MODELS[:2]]
                r = int(records[MODELS[0]][bi]["logits"]["raw_center/ridge"][qi].argmax())
                kind = stratum(y, ukr, hk, r)
                if kind is None:
                    continue
                counts[kind] += 1
                episode = bi * 4 + int(g.task_id_per_sample[s])
                if len(selected_episodes[kind]) >= args.per_stratum or episode in selected_episodes[kind] or int(ids[s]) in selected_nodes[kind]:
                    continue
                selected_episodes[kind].add(episode)
                selected_nodes[kind].add(int(ids[s]))
                batches[bi] = batch
                task = int(g.task_id_per_sample[s])
                mapping = g.task_label_map[task].tolist()
                supports = ((g.task_id_per_sample == task) & ~query).nonzero().flatten()

                def geometry(si):
                    start, end = int(g.ptr[si]), int(g.ptr[si + 1])
                    context = g.global_node_ids[start + 1:end]
                    context = context[context >= 0]
                    edges = g.edge_index[:, g.batch[g.edge_index[0]] == si]
                    return {"neighbors": [node(n) for n in context],
                            "edges_graph_rows": g.global_node_ids[edges].T.tolist(),
                            "center_indegree": int((edges[1] == start).sum()),
                            "center_outdegree": int((edges[0] == start).sum())}

                def score(z):
                    return {"predicted_label": names[mapping[int(z.argmax())]],
                            "correct": int(z.argmax()) == y, "logits": z.tolist(),
                            "true_minus_other_score": float(z[y] - z[1-y])}

                predictions = {}
                for m in MODELS:
                    predictions[m] = {d: score(z[qi]) for d, z in records[m][bi]["logits"].items()
                                      if d in ("full_model", "raw_center/ridge", "raw_context/ridge", "U1_pre_meta/ridge")}
                    predictions[m]["full_model"]["p_dataset_label"] = float(records[m][bi]["logits"]["full_model"][qi].softmax(0)[y])
                predictions["sampled_center_indegree/ridge"] = score(degree_logits[qi])
                cosine = F.cosine_similarity(raw["raw_center"][s][None], raw["raw_center"][supports])
                support_records = [{"node": node(ids[si]), "sample_index": si, "cosine_to_query": float(co), **geometry(si)}
                                   for si, co in zip(supports.tolist(), cosine.tolist())]
                cases.append({"case_id": f"{target}_b{bi:02d}_q{qi:02d}", "stratum": kind, "episode": episode,
                              "batch": bi, "query_index": qi, "sample_index": s, "local_label": y,
                              "label_mapping": mapping, "node": node(ids[s]), "query_geometry": geometry(s),
                              "supports": support_records, "predictions": predictions})
        protocol = json.loads((root.parent / "protocol.json").read_text())
        parities = {}
        with torch.no_grad():
            for m in MODELS:
                state = torch.load(checkpoints[m], map_location="cpu", weights_only=True)["model"]
                model = make_model(protocol, target, cache["graph_path"], a["x"].shape[1], state)
                for bi, batch in batches.items():
                    _, z, _ = model(*clone_batch(batch))
                    ref = records[m][bi]["logits"]["full_model"]
                    torch.testing.assert_close(z, ref, rtol=0, atol=1e-5)
                    parities[f"{m}/{bi}"] = float((z-ref).abs().max())
                for case in cases:
                    bi, qi, s, y = [case[k] for k in ("batch", "query_index", "sample_index", "local_label")]
                    result = {}
                    for condition in CONDITIONS:
                        altered = localized_intervention(batches[bi], s, condition)
                        _, logits, _ = model(*altered)
                        z = logits[qi]
                        result[condition] = {"p_dataset_label": float(z.softmax(0)[y]), "logits": z.tolist(),
                                             "predicted_label": names[case["label_mapping"][int(z.argmax())]], "correct": int(z.argmax()) == y}
                    case.setdefault("localized_interventions", {})[m] = result
        result = {"target": target, "root": str(root), "cache": cache, "checkpoints": checkpoints,
                  "selection": "first per stratum in fixed batch/query order, distinct episodes and query identities within stratum, before text reading",
                  "per_stratum": args.per_stratum, "stratum_query_counts": dict(counts), "cases": cases, "nodes": nodes,
                  "baseline_parity_atol": 1e-5, "baseline_logit_max_errors": parities,
                  "verified_all_cached_features_against_graph": True, "verified_all_cached_center_labels": True,
                  "intervention_scope": "one query subgraph OR all supports in its episode; all other query subgraphs unchanged; no training; no label changes",
                  "label_caveat": "Dataset annotations, not independently verified personal attributes. Supports' neighbor labels are audit-only and not classifier inputs.",
                  "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
        (args.output / f"{target}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({"target": target, "cases": len(cases), "counts": counts, "unique_nodes": len(nodes), "max_parity_error": max(parities.values())}), flush=True)
    (args.output / "DONE").write_text("Verified example extraction and localized interventions completed.\n")


def hydrate(args):
    """Use exact stored normalized texts and verify their embedding shard hashes."""
    import numpy as np
    import pyarrow.parquet as pq
    if args.output.exists():
        raise ValueError("choose a new output")
    args.output.mkdir(parents=True)
    for path in sorted(args.input.glob("*.json")):
        d = json.loads(path.read_text())
        target = d["target"]
        if target != "covid_political":
            root = args.data_root / target / "bio_embeddings/gte-multilingual-base/version=v001"
            obs = pq.read_table(root / "user_bio_observations.parquet", columns=["userid", "bio_hash"]).to_pylist()
            user_hash = {}
            for r in obs:
                if r["userid"] in user_hash and user_hash[r["userid"]] != r["bio_hash"]:
                    raise ValueError("ambiguous bio per user")
                user_hash[r["userid"]] = r["bio_hash"]
            texts = {r["bio_hash"]: r["normalized_bio_text"] for r in pq.read_table(root / "bio_texts.parquet", columns=["bio_hash", "normalized_bio_text"]).to_pylist()}
            index = {r["bio_hash"]: r for r in pq.read_table(root / "bio_embedding_index.parquet").to_pylist()}
            shards = {}
            for n in d["nodes"].values():
                h = user_hash.get(n["source_id"])
                loc = index.get(h)
                if loc is None:
                    vector = np.zeros(768, dtype=np.float32)
                    n["text"] = None
                else:
                    shard = loc["embedding_shard"]
                    if shard not in shards:
                        shards[shard] = np.load(root / shard, mmap_mode="r")
                    vector = np.asarray(shards[shard][loc["embedding_row"]], dtype=np.float32)
                    n["text"] = texts[h]
                if hashlib.sha256(vector.tobytes()).hexdigest() != n["feature_sha256"]:
                    raise ValueError(f"text embedding does not match graph: {n['source_id']}")
                n["bio_hash"] = h
                n["text_embedding_identity_verified"] = True
            if target == "facebook_page_reference":
                profiles = {r["account_id"]: r for r in pq.read_table(args.data_root / target / "tables/2022-02-24_to_2022-06-29_page_reference_v1/page_profiles.parquet",
                            columns=["account_id", "account_name", "page_category"]).to_pylist()}
                for n in d["nodes"].values():
                    p = profiles[n["source_id"]]
                    n["name"] = p["account_name"]
                    if n["label_id"] >= 0 and p["page_category"] != n["dataset_label"]:
                        raise ValueError("page category does not match graph")
        (args.output / path.name).write_text(json.dumps(d, ensure_ascii=False, indent=2))
        print(json.dumps({"target": target, "nodes_with_text": sum(n.get("text") is not None for n in d["nodes"].values()), "cases": len(d["cases"])}), flush=True)
    (args.output / "DONE").write_text("Joined raw/normalized text identities; embedding hashes verified where available.\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("mode", choices=["extract", "hydrate"])
    p.add_argument("--roots", type=Path, nargs="+", default=[])
    p.add_argument("--targets", nargs="+", default=["covid_political", "facebook_page_reference", "twibot20"])
    p.add_argument("--per-stratum", type=int, default=2)
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    args = p.parse_args()
    {"extract": extract, "hydrate": hydrate}[args.mode](args)
