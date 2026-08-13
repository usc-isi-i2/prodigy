#!/usr/bin/env python3
"""Reproduce cross-graph bio-overlap and GTE-geometry tables on Tucker."""

from __future__ import annotations

import argparse
import glob
from itertools import combinations
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


SEED = 20260807
SHARD_ROOTS = {
    "ukraine": "ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/shards",
    "covid": "covid19_twitter/bio_embeddings/gte-multilingual-base/version=v001/shards",
    "midterm": "midterm/bio_embeddings/gte-multilingual-base/version=v001/shards",
    "twibot20": "twibot20/bio_embeddings/gte-multilingual-base/version=v001/shards",
    "facebook-page-reference": "facebook_page_reference/bio_embeddings/gte-multilingual-base/version=v001/shards",
}
PT_PATHS = {
    "covid-political": "covid_political/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
    "ukraine-suspended": "ukr_rus_suspended/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
    "election2020-political": "election2020/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
    "hongkong": "cp_hk_twitter/embeddings/user_bio_embeddings_gte_multilingual_base.pt",
}
BIO_TEXTS = {
    "ukraine": "ukr_rus_twitter/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.parquet",
    "covid": "covid19_twitter/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.parquet",
    "midterm": "midterm/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.parquet",
    "covid-political": "covid_political/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.csv",
    "ukraine-suspended": "ukr_rus_suspended/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.csv",
    "election2020-political": "election2020/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.csv",
    "twibot20": "twibot20/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.parquet",
    "hongkong": "cp_hk_twitter/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.csv",
    "facebook-page-reference": "facebook_page_reference/bio_embeddings/gte-multilingual-base/version=v001/bio_texts.parquet",
}


def sample_shards(root: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    files = sorted(glob.glob(str(root / "*.emb.npy")))
    arrays = [np.load(path, mmap_mode="r") for path in files]
    lengths = np.asarray([len(array) for array in arrays])
    total = int(lengths.sum())
    ids = np.sort(rng.choice(total, min(n, total), replace=False))
    chunks, start = [], 0
    for array, end in zip(arrays, np.cumsum(lengths)):
        local = ids[(ids >= start) & (ids < end)] - start
        if len(local):
            chunks.append(np.asarray(array[local], dtype=np.float32))
        start = int(end)
    return np.concatenate(chunks)


def sample_pt(path: Path, n: int, rng: np.random.Generator) -> np.ndarray:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    x = obj["meanpool"].float().numpy()
    hashes = np.asarray(obj.get("bio_hashes", [str(i) for i in range(len(x))]), dtype=object)
    valid = np.flatnonzero((np.linalg.norm(x, axis=1) > 0) & (hashes != ""))
    _, first = np.unique(hashes[valid], return_index=True)
    valid = valid[first]
    return x[rng.choice(valid, min(n, len(valid)), replace=False)]


def unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def write_overlap(data_root: Path, output_dir: Path) -> None:
    conn = duckdb.connect()
    names = list(BIO_TEXTS)
    counts = {}
    for i, name in enumerate(names):
        path = data_root / BIO_TEXTS[name]
        source = f"read_parquet('{path}')" if path.suffix == ".parquet" else f"read_csv_auto('{path}', header=true)"
        conn.execute(
            f"CREATE TEMP VIEW bio_{i} AS SELECT DISTINCT bio_hash FROM {source} "
            "WHERE bio_hash IS NOT NULL AND trim(bio_hash) <> ''"
        )
        counts[name] = conn.execute(f"SELECT count(*) FROM bio_{i}").fetchone()[0]
    rows = []
    for (i, left), (j, right) in combinations(enumerate(names), 2):
        overlap = conn.execute(
            f"SELECT count(*) FROM bio_{i} INNER JOIN bio_{j} USING (bio_hash)"
        ).fetchone()[0]
        rows.append((left, right, overlap, overlap / counts[left], overlap / counts[right]))
    pd.DataFrame(rows, columns=["left", "right", "abs", "frac_left", "frac_right"]).to_csv(
        output_dir / "pairwise_bio_overlap.csv", index=False
    )


def geometry(samples: dict[str, np.ndarray], output_dir: Path) -> None:
    rng = np.random.default_rng(SEED + 1)
    bases, concentration = {}, []
    for name, raw in samples.items():
        x = unit_rows(raw.astype(np.float32))
        mean = x.mean(axis=0)
        centroid_norm = float(np.linalg.norm(mean))
        centroid_cos = x @ (mean / centroid_norm)
        pair_cos = np.sum(x * x[rng.permutation(len(x))], axis=1)
        centered = x - mean
        covariance = centered.T @ centered / (len(x) - 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues, eigenvectors = eigenvalues[::-1], eigenvectors[:, ::-1]
        effective_dim = float(eigenvalues.sum() ** 2 / np.dot(eigenvalues, eigenvalues))
        bases[name] = eigenvectors[:, : round(effective_dim)]
        q10, q50, q90 = np.quantile(centroid_cos, [0.1, 0.5, 0.9])
        concentration.append(
            (name, len(x), centroid_norm, q10, q50, q90, pair_cos.mean(), pair_cos.std(), effective_dim)
        )
    columns = [
        "dataset", "n_sample", "centroid_norm", "centroid_cosine_p10",
        "centroid_cosine_p50", "centroid_cosine_p90", "random_pair_cosine_mean",
        "random_pair_cosine_sd", "effective_dimension",
    ]
    pd.DataFrame(concentration, columns=columns).to_csv(output_dir / "concentration.csv", index=False)

    overlap_rows = []
    for left, right in combinations(samples, 2):
        a, b = bases[left], bases[right]
        shared = float(np.sum((a.T @ b) ** 2))
        smaller = min(a.shape[1], b.shape[1])
        overlap_rows.append((left, right, shared, shared / smaller, a.shape[1] * b.shape[1] / 768))
    pd.DataFrame(
        overlap_rows,
        columns=["left", "right", "shared_soft_dimensions", "fraction_of_smaller", "random_expectation"],
    ).to_csv(output_dir / "subspace_overlap.csv", index=False)


def centroid_distances(samples: dict[str, np.ndarray], output_dir: Path) -> None:
    names = list(samples)
    split_distances, split_labels = [], []
    for half in (0, 1):
        centroids = []
        for x in samples.values():
            x = unit_rows(x.astype(np.float32))
            cut = len(x) // 2
            part = x[:cut] if half == 0 else x[cut:]
            center = part.mean(axis=0)
            centroids.append(center / np.linalg.norm(center))
        centers = np.stack(centroids)
        distance = 1 - centers @ centers.T
        np.fill_diagonal(distance, 0)
        split_distances.append(distance)
        split_labels.append(
            fcluster(linkage(squareform(distance, checks=False), method="average"), 4, criterion="maxclust")
        )
    mean_distance = (split_distances[0] + split_distances[1]) / 2
    table = pd.DataFrame(mean_distance, index=names, columns=names)
    table.index.name = "dataset"
    table.to_csv(output_dir / "centroid_cosine_distance.csv")
    tri = np.triu_indices(len(names), 1)
    diagnostics = pd.DataFrame(
        [{
            "sample_sizes": ";".join(f"{name}:{len(samples[name])}" for name in names),
            "distance_correlation": np.corrcoef(split_distances[0][tri], split_distances[1][tri])[0, 1],
            "mean_absolute_split_difference": np.mean(np.abs(split_distances[0] - split_distances[1])[tri]),
            "max_absolute_split_difference": np.max(np.abs(split_distances[0] - split_distances[1])[tri]),
            "four_cluster_ari_between_halves": adjusted_rand_score(split_labels[0], split_labels[1]),
            "half_1_labels": ";".join(map(str, split_labels[0])),
            "half_2_labels": ";".join(map(str, split_labels[1])),
        }]
    )
    diagnostics.to_csv(output_dir / "centroid_split_diagnostics.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_overlap:
        write_overlap(args.data_root, args.output_dir)
    rng = np.random.default_rng(args.seed)
    samples = {
        name: sample_shards(args.data_root / relative, args.sample_size, rng)
        for name, relative in SHARD_ROOTS.items()
    }
    samples.update({
        name: sample_pt(args.data_root / relative, args.sample_size, rng)
        for name, relative in PT_PATHS.items()
    })
    geometry(samples, args.output_dir)
    centroid_distances(samples, args.output_dir)


if __name__ == "__main__":
    main()
