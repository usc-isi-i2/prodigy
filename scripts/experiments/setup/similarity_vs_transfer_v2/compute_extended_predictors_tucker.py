#!/usr/bin/env python3
"""Extract v2 user and message-passing-feature predictors on Tucker.

Graphs are loaded serially. Feature samples remain in memory; full user-ID
vectors are stored as sorted uint64 hashes in a temporary directory so overlap
does not require nine large Python sets at once.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import skew, wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[4]
GD = ROOT / "scripts/experiments/analysis/graph_divergence"
PFC = ROOT / "scripts/experiments/analysis/path_feature_coupling"
sys.path[:0] = [str(GD), str(PFC)]

from compute_graph_divergence import (  # noqa: E402
    DEFAULT_GRAPHS, as_numpy, load_graph, proxy_a_distance, rbf_mmd2,
)
from analyze_neighbor_augmented_features import (  # noqa: E402
    build_undirected_csr, sample_nonmissing_node_ids, sampled_neighbor_means, spaces,
)


def finite(value):
    return None if not np.isfinite(value) else float(value)


def skew_summary(rows: np.ndarray) -> dict:
    coeff = np.asarray(skew(rows, axis=0, bias=False, nan_policy="omit"), float)
    coeff = coeff[np.isfinite(coeff)]
    return {
        "n_dimensions": int(len(coeff)),
        "mean_skew": finite(np.mean(coeff)),
        "median_skew": finite(np.median(coeff)),
        "mean_absolute_skew": finite(np.mean(np.abs(coeff))),
        "left_skew_fraction": finite(np.mean(coeff < -0.5)),
        "right_skew_fraction": finite(np.mean(coeff > 0.5)),
        "coefficients": coeff.tolist(),
    }


def projected_frechet(a: np.ndarray, b: np.ndarray, seed: int, dims: int) -> float:
    n_dims = min(dims, a.shape[1], len(a) + len(b) - 1)
    z = PCA(n_components=n_dims, svd_solver="randomized", random_state=seed).fit_transform(np.vstack([a, b]))
    x, y = z[:len(a)], z[len(a):]
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    cross = sqrtm(cx @ cy)
    if np.iscomplexobj(cross):
        cross = cross.real
    return float((mx - my) @ (mx - my) + np.trace(cx + cy - 2 * cross))


def centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.mean(0), b.mean(0)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(1 - x @ y / denom) if denom else np.nan


def local_structure_signatures(adjacency, nodes: np.ndarray, fanout: int, rng) -> np.ndarray:
    """Degree-binned one-hop computation-tree signatures for sampled centers."""
    degrees = np.diff(adjacency.indptr)
    bins = np.asarray([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, np.inf])
    result = np.zeros((len(nodes), 6 + len(bins) - 1), dtype=np.float32)
    for row, node in enumerate(nodes):
        nbrs = adjacency.indices[adjacency.indptr[node]:adjacency.indptr[node + 1]]
        if len(nbrs) > fanout:
            nbrs = rng.choice(nbrs, fanout, replace=False)
        nbr_degree = degrees[nbrs].astype(float)
        log_degree = np.log1p(nbr_degree)
        result[row, :6] = [np.log1p(degrees[node]), len(nbrs),
                           log_degree.mean() if len(nbrs) else 0,
                           log_degree.std() if len(nbrs) else 0,
                           np.max(log_degree) if len(nbrs) else 0,
                           np.mean(nbr_degree == 1) if len(nbrs) else 0]
        if len(nbrs):
            result[row, 6:] = np.histogram(nbr_degree, bins=bins)[0] / len(nbrs)
    return result


def shared_projection(samples: dict[str, np.ndarray], dims: int, seed: int):
    names = list(samples)
    lengths = [len(samples[name]) for name in names]
    joined = np.vstack([samples[name] for name in names])
    n_dims = min(dims, joined.shape[1], len(joined) - 1)
    projected = PCA(n_components=n_dims, svd_solver="randomized", random_state=seed).fit_transform(joined)
    cuts = np.cumsum([0] + lengths)
    return {name: projected[cuts[i]:cuts[i + 1]].astype(np.float32) for i, name in enumerate(names)}


def hash_ids(ids) -> np.ndarray:
    def mix64(x):
        x = np.asarray(x, dtype=np.uint64).copy()
        x ^= x >> np.uint64(30); x *= np.uint64(0xbf58476d1ce4e5b9)
        x ^= x >> np.uint64(27); x *= np.uint64(0x94d049bb133111eb)
        x ^= x >> np.uint64(31)
        return x

    if hasattr(ids, "detach"):
        ids = ids.detach().cpu().numpy()
    arr = np.asarray(ids)
    if np.issubdtype(arr.dtype, np.integer):
        return np.unique(mix64(arr))
    out = np.empty(len(ids), np.uint64)
    for i, value in enumerate(ids):
        text = str(value).strip()
        if text.isdigit():
            out[i] = mix64([int(text)])[0]
        else:
            out[i] = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8, person=b"gfm-user").digest(), "little")
    return np.unique(out)


def user_ids(graph):
    for key in ("user_ids", "node_ids", "ids"):
        if key in graph and graph[key] is not None:
            return graph[key], key
    if "u2i" in graph and graph["u2i"] is not None:
        return list(graph["u2i"].keys()), "u2i.keys"
    return None, None


def matrix(n: int):
    return [[None for _ in range(n)] for _ in range(n)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample-nodes", type=int, default=2_000)
    p.add_argument("--fanout", type=int, default=100)
    p.add_argument("--frechet-dims", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260807)
    args = p.parse_args()
    names = list(DEFAULT_GRAPHS)
    samples, summaries, availability, structure_samples = {}, {}, {}, {}
    with tempfile.TemporaryDirectory(prefix="gfm-transfer-users-") as temp:
        temp = Path(temp)
        user_files = {}
        for gi, name in enumerate(names):
            path = args.data_root / DEFAULT_GRAPHS[name]
            print(f"[{gi+1}/{len(names)}] {name}: {path}", flush=True)
            graph = load_graph(path)
            x, edge_index = graph["x"], graph["edge_index"]
            rng = np.random.default_rng(args.seed + gi)
            nodes, raw = sample_nonmissing_node_ids(x, int(x.shape[0]), args.sample_nodes, rng)
            adjacency = build_undirected_csr(as_numpy(edge_index), int(x.shape[0]))
            neighbor, degree, sampled = sampled_neighbor_means(x, adjacency, nodes, args.fanout, rng)
            samples[name] = spaces(raw, neighbor)
            structure_samples[name] = local_structure_signatures(adjacency, nodes, args.fanout, rng)
            summaries[name] = {
                "n_sampled_centers": int(len(nodes)), "fanout": args.fanout,
                "mean_center_degree": finite(np.mean(degree)),
                "mean_sampled_neighbors": finite(np.mean(sampled)),
                "spaces": {space: skew_summary(rows) for space, rows in samples[name].items()},
            }
            ids, key = user_ids(graph)
            if name == "facebook_page_reference":
                availability[name] = {"available": False, "reason": "different platform/identity namespace"}
            elif ids is None:
                availability[name] = {"available": False, "reason": "no user ID field in graph artifact"}
            else:
                hashes = hash_ids(ids)
                target = temp / f"{name}.npy"; np.save(target, hashes)
                user_files[name] = target
                availability[name] = {"available": True, "source_key": key, "n_unique": int(len(hashes)), "hash": "blake2b-64 or splitmix64 for integer IDs"}
            del graph, x, edge_index, adjacency
            gc.collect()

        # A shared projection makes all pairwise metrics comparable and keeps
        # proxy-A/MMD/Frechet tractable in the 1536-D concatenated space.
        projected = {}
        for si, space in enumerate(next(iter(samples.values()))):
            projected[space] = shared_projection(
                {name: samples[name][space] for name in names}, args.frechet_dims,
                args.seed + 10_000 + si,
            )
        projected["local_structure"] = structure_samples

        n = len(names)
        pairwise = {}
        for space, space_samples in projected.items():
            for metric in ("centroid_cosdist", "mmd2", "proxy_a_distance", "projected_frechet", "skew_l1", "skew_wasserstein"):
                if space == "local_structure" and metric.startswith("skew_"):
                    continue
                pairwise[f"{space}_{metric}"] = matrix(n)
            for i, a_name in enumerate(names):
                a = space_samples[a_name]
                a_skew = (np.asarray(summaries[a_name]["spaces"][space]["coefficients"])
                          if space != "local_structure" else None)
                for j, b_name in enumerate(names):
                    if j < i:
                        continue
                    b = space_samples[b_name]
                    b_skew = (np.asarray(summaries[b_name]["spaces"][space]["coefficients"])
                              if space != "local_structure" else None)
                    values = {
                        "centroid_cosdist": centroid_cosine(a, b),
                        "mmd2": 0.0 if i == j else rbf_mmd2(a, b, np.random.default_rng(args.seed + 1000*i + j)),
                        "proxy_a_distance": 0.0 if i == j else proxy_a_distance(a, b, np.random.default_rng(args.seed + 1000*i + j)),
                        "projected_frechet": 0.0 if i == j else projected_frechet(a, b, args.seed + 1000*i + j, min(args.frechet_dims, a.shape[1])),
                    }
                    if a_skew is not None:
                        values["skew_l1"] = float(np.mean(np.abs(a_skew - b_skew)))
                        values["skew_wasserstein"] = float(wasserstein_distance(a_skew, b_skew))
                    for metric, value in values.items():
                        pairwise[f"{space}_{metric}"][i][j] = finite(value)
                        pairwise[f"{space}_{metric}"][j][i] = finite(value)

        # Embedding-derived topic composition: a shared unsupervised vocabulary
        # over raw node features, compared by Jensen-Shannon distance.
        raw_projected = projected["raw_center"]
        joined = np.vstack([raw_projected[name] for name in names])
        topic_model = MiniBatchKMeans(n_clusters=64, batch_size=2048,
                                      random_state=args.seed, n_init=10).fit(joined)
        topic_mix = {}
        for name in names:
            counts = np.bincount(topic_model.predict(raw_projected[name]), minlength=64).astype(float) + 1e-8
            topic_mix[name] = counts / counts.sum()
        pairwise["embedding_topic_js_distance"] = matrix(n)
        for i, a_name in enumerate(names):
            for j, b_name in enumerate(names):
                pairwise["embedding_topic_js_distance"][i][j] = float(jensenshannon(topic_mix[a_name], topic_mix[b_name]) ** 2)

        for metric in ("user_jaccard", "user_source_containment", "user_target_containment"):
            pairwise[metric] = matrix(n)
        for i, a_name in enumerate(names):
            for j, b_name in enumerate(names):
                if a_name not in user_files or b_name not in user_files:
                    continue
                a = np.load(user_files[a_name], mmap_mode="r")
                b = np.load(user_files[b_name], mmap_mode="r")
                shared = len(np.intersect1d(a, b, assume_unique=True))
                pairwise["user_jaccard"][i][j] = shared / max(len(a) + len(b) - shared, 1)
                pairwise["user_source_containment"][i][j] = shared / max(len(a), 1)
                pairwise["user_target_containment"][i][j] = shared / max(len(b), 1)

        artifact = {"meta": {"seed": args.seed, "sample_nodes": args.sample_nodes, "fanout": args.fanout,
                              "shared_projection_dims": args.frechet_dims,
                              "local_structure_signature": "center degree plus sampled-neighbor degree moments and histogram",
                              "embedding_topic_clusters": 64,
                              "warning": "64-bit user hashes have negligible but nonzero collision probability"},
                    "graphs": names, "per_graph": summaries, "user_id_availability": availability,
                    "pairwise": pairwise}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(artifact, indent=2) + "\n")


if __name__ == "__main__":
    main()
