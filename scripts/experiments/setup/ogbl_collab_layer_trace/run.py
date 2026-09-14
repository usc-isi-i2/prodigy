#!/usr/bin/env python3
"""Frozen layer and decoder-input trace for the compact ogbl-collab scorer."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
joint = load_module(HERE.parent / "ogbl_collab_compact_joint" / "run.py", "compact_joint_trace")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(pos, neg):
    # Exact Mann-Whitney AUC with average ranks for ties.
    from scipy.stats import rankdata
    values = np.concatenate([pos, neg])
    ranks = rankdata(values, method="average")
    npos, nneg = len(pos), len(neg)
    return float((ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def hitmask(pos, neg):
    return pos > np.sort(neg)[-50]


def metrics(pos, neg):
    mask = hitmask(pos, neg)
    return {
        "hits_at_50": float(mask.mean()),
        "auc": auc(pos, neg),
        "cutoff": float(np.sort(neg)[-50]),
        "positive_quantiles": np.quantile(pos, [0, .1, .5, .9, 1]).tolist(),
        "negative_quantiles": np.quantile(neg, [0, .5, .9, .99, 1]).tolist(),
    }


@torch.no_grad()
def layer_outputs(model, x, adj):
    values = []
    current = x
    for index, conv in enumerate(model.encoder.convs):
        current = conv(current, adj)
        if index < len(model.encoder.convs) - 1:
            current = F.relu(current)
        values.append(current)
    return values


@torch.no_grad()
def decoder_scores(model, z, edges, structural, zero_embedding=False, zero_structure=False, batch=16384):
    out = []
    for start in range(0, len(edges), batch):
        pair = edges[start:start + batch]
        left, right = z[pair[:, 0]], z[pair[:, 1]]
        learned = torch.cat([left * right, (left - right).abs()], dim=1)
        hand = structural[start:start + batch]
        if zero_embedding:
            learned = torch.zeros_like(learned)
        if zero_structure:
            hand = torch.zeros_like(hand)
        score = model.decoder(torch.cat([learned, hand], dim=1)).flatten()
        score = score.masked_fill(pair[:, 0] == pair[:, 1], -1e9)
        out.append(score.cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def cosine_scores(z, edges, batch=16384):
    out = []
    for start in range(0, len(edges), batch):
        pair = edges[start:start + batch]
        out.append(F.cosine_similarity(z[pair[:, 0]], z[pair[:, 1]], dim=1).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--sample-first", type=Path, required=True)
    parser.add_argument("--sample-second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    assert args.device == "cuda:1"
    args.out.mkdir(parents=True, exist_ok=False)

    checkpoint = args.checkpoint_root / "bce0" / "best.pt"
    saved_scores_path = args.checkpoint_root / "bce0" / "best_scores.npz"
    result_path = args.checkpoint_root / "bce0" / "results.json"
    config_path = args.checkpoint_root / "bce0" / "config.json"
    result = json.loads(result_path.read_text())
    config = json.loads(config_path.read_text())
    assert result["best"]["step"] == 150 and result["revision"] == config["revision"]
    assert sha256(checkpoint) == result["files"]["best.pt"]
    assert sha256(saved_scores_path) == result["files"]["best_scores.npz"]

    panel_path = args.original / "assessment" / "year2018.npz"
    node_path = args.original / "node_features.npy"
    std_path = args.original / "standardization.npz"
    for path in (panel_path, node_path, std_path):
        expected = next(v for k, v in config["files"].items() if k.endswith(str(path.relative_to(args.original))))
        assert sha256(path) == expected

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    panel = np.load(panel_path)
    standardization = np.load(std_path)
    tensors = joint.panel_tensors(panel, standardization["mean"], standardization["std"], device)
    x = torch.as_tensor(np.load(node_path), dtype=torch.float32, device=device)
    adj = joint.adjacency(panel["graph_edges"], len(x), device)
    model = joint.Model("joint").to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    assert state["step"] == 150 and state["seed"] == 0 and state["revision"] == result["revision"]
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    layers = layer_outputs(model, x, adj)
    assert len(layers) == 3 and all(tuple(z.shape) == (len(x), 256) for z in layers)
    full = {}
    decoder = {}
    cosine = {}
    edges = {side: tensors[side][0] for side in ("p", "n")}
    structural = {side: tensors[side][1] for side in ("p", "n")}
    for layer_index, z in enumerate(layers, 1):
        values = {side: decoder_scores(model, z, edges[side], structural[side]) for side in ("p", "n")}
        decoder[f"layer{layer_index}"] = metrics(values["p"], values["n"])
        full[f"layer{layer_index}"] = values
        pair_cos = {side: cosine_scores(z, edges[side]) for side in ("p", "n")}
        cosine[f"layer{layer_index}"] = {
            **metrics(pair_cos["p"], pair_cos["n"]),
            "node_norm_quantiles": np.quantile(torch.linalg.vector_norm(z, dim=1).cpu().numpy(), [0, .1, .5, .9, 1]).tolist(),
        }

    raw_cos = {side: panel[side + "symmetric"][:, 7] for side in ("p", "n")}
    cosine["raw_node_features"] = metrics(raw_cos["p"], raw_cos["n"])
    z = layers[-1]
    for label, zero_embedding, zero_structure in (
        ("zero_learned_pair", True, False),
        ("zero_structural", False, True),
    ):
        values = {side: decoder_scores(model, z, edges[side], structural[side], zero_embedding, zero_structure) for side in ("p", "n")}
        full[label] = values
        decoder[label] = metrics(values["p"], values["n"])

    saved = np.load(saved_scores_path)
    for side in ("p", "n"):
        np.testing.assert_array_equal(full["layer3"][side], saved[side])
    assert decoder["layer3"]["hits_at_50"] == result["best"]["validation_hits"]
    from ogb.linkproppred import Evaluator
    evaluator = Evaluator(name="ogbl-collab")
    evaluator.K = 50
    assert evaluator.eval({"y_pred_pos": full["layer3"]["p"], "y_pred_neg": full["layer3"]["n"]})["hits@50"] == decoder["layer3"]["hits_at_50"]

    baseline_mask = hitmask(full["layer3"]["p"], full["layer3"]["n"])
    feature = panel["psymmetric"]
    strata = {
        "all": np.ones(len(feature), dtype=bool),
        "novel": feature[:, 8] == 0,
        "repeat": feature[:, 8] > 0,
        "zero_aa": feature[:, 12] == 1,
        "nonzero_aa": feature[:, 12] == 0,
    }
    for label, values in full.items():
        mask = hitmask(values["p"], values["n"])
        decoder[label]["relative_to_full"] = {
            name: {
                "count": int(group.sum()),
                "hits_at_50": float(mask[group].mean()),
                "recovered": int((group & mask & ~baseline_mask).sum()),
                "lost": int((group & ~mask & baseline_mask).sum()),
            }
            for name, group in strata.items()
        }

    sample_indices = []
    sample_hashes = {}
    for name, path in (("first", args.sample_first), ("second", args.sample_second)):
        sample = json.loads(path.read_text())
        assert sample["panel_sha256"] == sha256(panel_path)
        assert sample["score_sha256"] == sha256(saved_scores_path)
        sample_indices.extend(sample["indices"])
        sample_hashes[name] = sha256(path)
    assert len(sample_indices) == len(set(sample_indices)) == 40
    sampled = []
    for index in sample_indices:
        row = {"index": index}
        for label, values in full.items():
            row[label] = {
                "score": float(values["p"][index]),
                "hit": bool(hitmask(values["p"], values["n"])[index]),
                "negatives_ahead": int((values["n"] >= values["p"][index]).sum()),
            }
        sampled.append(row)

    output = {
        "complete": True,
        "classification": "post-hoc frozen validation diagnostic; intermediate layers and zero-input ablations are off-distribution",
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint), "step": 150, "seed": 0, "producing_revision": result["revision"]},
        "inputs": {"panel_sha256": sha256(panel_path), "node_features_sha256": sha256(node_path), "standardization_sha256": sha256(std_path), "saved_scores_sha256": sha256(saved_scores_path), "sample_hashes": sample_hashes},
        "tensor_shapes": {"nodes": list(x.shape), "positive_pairs": list(panel["pos"].shape), "negative_pairs": list(panel["neg"].shape), "layers": [list(z.shape) for z in layers]},
        "decoder_scores": decoder,
        "endpoint_cosine": cosine,
        "sampled_40_misses": sampled,
        "exact_saved_score_replay": True,
        "official_evaluator_parity": True,
        "no_fit": True,
        "test_access": False,
        "prior_test_exploration_disclosed": True,
    }
    outpath = args.out / "results.json"
    outpath.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"complete": True, "decoder_scores": decoder, "endpoint_cosine": cosine}, indent=2))


if __name__ == "__main__":
    main()
