#!/usr/bin/env python3
"""Build GTE-attributed Cora or PubMed citation graph artifacts.

The graph structure, labels, canonical masks, and raw title/abstract text come
from the pinned Graph-COM/Text-Attributed-Graphs release.  This script replaces
the supplied encoder features with the same pinned GTE multilingual embeddings
used by the social graphs in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from scripts.graph_construction.benchmark_targets import (
    attach_benchmark_targets,
    build_static_edge_split,
)
from scripts.tweet_embeddings.constants import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    OUTPUT_DIM,
)


SOURCE_REPOSITORY = "Graph-COM/Text-Attributed-Graphs"
SOURCE_REVISION = "4a59e25d4ed77b6541d6830ef89be99942a01394"
SOURCE_BASE_URL = f"https://huggingface.co/datasets/{SOURCE_REPOSITORY}/resolve/{SOURCE_REVISION}"

DATASETS = {
    "cora": {
        "nodes": 2708,
        "label_names": [
            "Case Based",
            "Genetic Algorithms",
            "Neural Networks",
            "Probabilistic Methods",
            "Reinforcement Learning",
            "Rule Learning",
            "Theory",
        ],
        "files": {
            "processed_data.pt": "58effe764f4f0a15363d2b168445c2c66dc81220e51621d74156f7139d8ee2ad",
            "raw_texts.pt": "95455e90a37df26903d4dc4d8793c5518599229d4a20703e93607a094081da66",
        },
    },
    "pubmed": {
        "nodes": 19717,
        "label_names": [
            "Diabetes Mellitus Experimental",
            "Diabetes Mellitus Type 1",
            "Diabetes Mellitus Type 2",
        ],
        "files": {
            "processed_data.pt": "10294c852137b76a5fa37303e244ff40d4a86a3823f0a847049bb170b74eb0e1",
            "raw_texts.pt": "04a46051013658f98223b91babee0918ed06363d2aa0f351115a873859262cb0",
        },
    },
}


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def download_sources(dataset: str, source_dir: Path) -> None:
    """Download immutable source files and verify their LFS object hashes."""
    source_dir.mkdir(parents=True, exist_ok=True)
    for filename, expected_sha in DATASETS[dataset]["files"].items():
        destination = source_dir / filename
        if destination.is_file() and sha256_file(destination) == expected_sha:
            print(f"[source] verified existing {destination}", flush=True)
            continue
        url = f"{SOURCE_BASE_URL}/{dataset}/{filename}?download=true"
        print(f"[source] downloading {url}", flush=True)
        with tempfile.NamedTemporaryFile(dir=source_dir, delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(url) as response, temporary_path.open("wb") as output:
                shutil.copyfileobj(response, output)
            actual_sha = sha256_file(temporary_path)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Checksum mismatch for {filename}: expected {expected_sha}, got {actual_sha}"
                )
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)


def verify_sources(dataset: str, source_dir: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for filename, expected_sha in DATASETS[dataset]["files"].items():
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}. Re-run with --download or place the pinned source file there."
            )
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_sha}, got {actual_sha}"
            )
        checksums[filename] = actual_sha
    return checksums


def normalize_raw_texts(raw: Any) -> list[str]:
    """Normalize the published raw-text container without reordering nodes."""
    if isinstance(raw, dict):
        for key in ("raw_texts", "texts", "text"):
            if key in raw:
                raw = raw[key]
                break
        else:
            raise ValueError(f"Unknown raw-text mapping keys: {sorted(raw)}")
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise TypeError(f"Expected a list-like raw-text artifact, got {type(raw).__name__}")
    texts = []
    for value in raw:
        if value is None:
            texts.append("")
        else:
            texts.append(" ".join(str(value).split()))
    return texts


def _graph_value(graph: Any, key: str, default: Any = None) -> Any:
    if isinstance(graph, dict):
        return graph.get(key, default)
    return getattr(graph, key, default)


def _available_graph_keys(graph: Any) -> Iterable[str]:
    if isinstance(graph, dict):
        return graph.keys()
    keys = getattr(graph, "keys", None)
    if callable(keys):
        return keys()
    if isinstance(keys, (list, tuple, set)):
        return keys
    return []


def assemble_graph_artifact(
    dataset: str,
    source_graph: Any,
    texts: list[str],
    embeddings: torch.Tensor,
    *,
    static_holdout_frac: float = 0.15,
    seed: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Align source topology, raw texts, labels, splits, and GTE features."""
    spec = DATASETS[dataset]
    edge_index = _graph_value(source_graph, "edge_index")
    labels = _graph_value(source_graph, "y")
    if edge_index is None or labels is None:
        raise ValueError("processed_data.pt must contain edge_index and y")
    edge_index = torch.as_tensor(edge_index, dtype=torch.long).contiguous()
    labels = torch.as_tensor(labels, dtype=torch.long).view(-1).contiguous()
    embeddings = torch.as_tensor(embeddings, dtype=torch.float32).contiguous()

    num_nodes = int(labels.numel())
    if num_nodes != int(spec["nodes"]):
        raise ValueError(f"{dataset} expected {spec['nodes']} nodes, found {num_nodes}")
    if len(texts) != num_nodes:
        raise ValueError(f"{dataset} text/node mismatch: {len(texts)} texts for {num_nodes} nodes")
    if embeddings.shape != (num_nodes, OUTPUT_DIM):
        raise ValueError(
            f"{dataset} embedding shape must be {(num_nodes, OUTPUT_DIM)}, got {tuple(embeddings.shape)}"
        )
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("edge_index contains an out-of-range node id")
    if not torch.isfinite(embeddings).all():
        raise ValueError("GTE embeddings contain non-finite values")
    embedding_norms = torch.linalg.vector_norm(embeddings, dim=1)
    if not torch.allclose(
        embedding_norms,
        torch.ones_like(embedding_norms),
        rtol=5e-3,
        atol=5e-3,
    ):
        raise ValueError(
            "GTE embeddings must be L2-normalized; observed norm range "
            f"[{float(embedding_norms.min()):.4f}, {float(embedding_norms.max()):.4f}]"
        )
    observed_labels = sorted(set(labels.tolist()))
    expected_labels = list(range(len(spec["label_names"])))
    if observed_labels != expected_labels:
        raise ValueError(f"Expected labels {expected_labels}, found {observed_labels}")

    artifact: dict[str, Any] = {
        "x": embeddings,
        "edge_index": edge_index,
        "y": labels,
        "label_names": list(spec["label_names"]),
        "label_type": "classification",
        "node_ids": list(range(num_nodes)),
        "user_ids": list(range(num_nodes)),
        "feature_names": [f"gte_{index}" for index in range(OUTPUT_DIM)],
        "edge_attr_feature_names": [],
        "raw_texts": texts,
        "text_field": "paper title and abstract",
        "feature_model": DEFAULT_MODEL,
        "feature_model_revision": DEFAULT_REVISION,
    }
    # Preserve the canonical transductive masks when present.  The current
    # episodic loader creates stratified splits, while these fields allow later
    # baselines to reproduce the source benchmark protocol exactly.
    for key in ("train_mask", "val_mask", "valid_mask", "test_mask"):
        value = _graph_value(source_graph, key)
        if value is not None:
            value = torch.as_tensor(value, dtype=torch.bool)
            if value.shape[0] != num_nodes:
                raise ValueError(f"{key} has leading dimension {value.shape[0]}, expected {num_nodes}")
            artifact[key] = value.contiguous()

    split = build_static_edge_split(
        edge_index, holdout_frac=static_holdout_frac, seed=seed
    )
    attach_benchmark_targets(artifact, static_split=split)
    metadata = {
        "dataset": dataset,
        "num_nodes": num_nodes,
        "num_edges": int(edge_index.shape[1]),
        "num_classes": len(spec["label_names"]),
        "empty_text_nodes": sum(not text for text in texts),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_dtype": str(embeddings.dtype).replace("torch.", ""),
        "embedding_norm_min": float(embedding_norms.min()),
        "embedding_norm_max": float(embedding_norms.max()),
        "embedding_model": DEFAULT_MODEL,
        "embedding_model_revision": DEFAULT_REVISION,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION,
        "source_graph_keys": sorted(_available_graph_keys(source_graph)),
        "static_split": split.stats,
    }
    return artifact, metadata


def encode_texts(
    texts: list[str],
    *,
    device: str,
    batch_size: int,
    max_seq_length: int,
    cache_folder: str | None,
) -> torch.Tensor:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        DEFAULT_MODEL,
        revision=DEFAULT_REVISION,
        trust_remote_code=True,
        device=device,
        cache_folder=cache_folder,
    )
    model.max_seq_length = max_seq_length
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return torch.from_numpy(np.asarray(embeddings, dtype=np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--cache-folder")
    parser.add_argument("--static-holdout-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = args.dataset
    dataset_root = Path("/dataMeR1/phil/data") / dataset
    source_dir = (args.source_dir or dataset_root / "raw" / "graph_com_tag").expanduser()
    out_path = (args.out or dataset_root / "graphs" / "citation_graph.pt").expanduser()
    meta_path = out_path.with_suffix(".meta.json")
    if not args.overwrite and (out_path.exists() or meta_path.exists()):
        raise FileExistsError(f"Refusing to overwrite {out_path} or {meta_path}")
    if args.download:
        download_sources(dataset, source_dir)
    source_checksums = verify_sources(dataset, source_dir)

    print(f"[load] {source_dir / 'processed_data.pt'}", flush=True)
    source_graph = torch.load(source_dir / "processed_data.pt", map_location="cpu")
    raw_texts = torch.load(source_dir / "raw_texts.pt", map_location="cpu")
    texts = normalize_raw_texts(raw_texts)
    print(f"[embed] {len(texts):,} node texts with {DEFAULT_MODEL}@{DEFAULT_REVISION}", flush=True)
    embeddings = encode_texts(
        texts,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        cache_folder=args.cache_folder,
    )
    artifact, metadata = assemble_graph_artifact(
        dataset,
        source_graph,
        texts,
        embeddings,
        static_holdout_frac=args.static_holdout_frac,
        seed=args.seed,
    )
    metadata["source_sha256"] = source_checksums

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, delete=False) as temporary:
        temporary_graph = Path(temporary.name)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent, mode="w", encoding="utf-8", delete=False
    ) as temporary:
        temporary_meta = Path(temporary.name)
        json.dump(metadata, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
    try:
        torch.save(artifact, temporary_graph)
        temporary_graph.replace(out_path)
        temporary_meta.replace(meta_path)
    finally:
        temporary_graph.unlink(missing_ok=True)
        temporary_meta.unlink(missing_ok=True)
    print(
        f"[done] {out_path}: {metadata['num_nodes']:,} nodes, "
        f"{metadata['num_edges']:,} directed edges, {metadata['num_classes']} classes",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
