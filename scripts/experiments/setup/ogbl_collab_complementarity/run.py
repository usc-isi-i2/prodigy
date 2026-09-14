#!/usr/bin/env python3
"""Replay validation scores and measure AA-DC / nonlinear MLP complementarity."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import time
from pathlib import Path

import numpy as np

ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
UPSTREAM = "b499c2046cfe76448545dfe08fad9effb58dd076"
FINGERPRINT = "07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4"


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def threshold(scores, k=50):
    if len(scores) < k or not np.isfinite(scores).all():
        raise ValueError("invalid negative score panel")
    return float(np.partition(scores, -k)[-k])


def top50(scores):
    # Deterministic index tie-break for identity diagnostics only. Hits uses strict >.
    return set(np.lexsort((np.arange(len(scores)), -scores))[:50].tolist())


def compare(ap, an, mp, mn):
    at, mt = threshold(an), threshold(mn)
    if at <= 0:
        raise ValueError("AA-DC threshold must be positive for this normalization")
    ah, mh = ap > at, mp > mt
    abase = top50(an)
    rows = []
    for alpha in ALPHAS:
        cp = np.maximum(ap.astype(np.float64) / at, alpha * np.exp(mp.astype(np.float64) - mt))
        cn = np.maximum(an.astype(np.float64) / at, alpha * np.exp(mn.astype(np.float64) - mt))
        ct = threshold(cn)
        hit = cp > ct
        rows.append({
            "alpha": alpha, "hits_at_50": float(hit.mean()),
            "delta_pp": float((hit.mean() - ah.mean()) * 100),
            "recovered_positives": int((hit & ~ah).sum()),
            "lost_positives": int((~hit & ah).sum()),
            "negative50_normalized": ct,
            "new_negatives_above_old_threshold": int(((cn > 1) & ~(an > at)).sum()),
            "new_negatives_in_top50": len(top50(cn) - abase),
        })
    return {
        "aadc_hits_at_50": float(ah.mean()), "mlp_hits_at_50": float(mh.mean()),
        "aadc_negative50": at, "mlp_negative50": mt,
        "aadc_missed_positives": int((~ah).sum()),
        "mlp_recovers_aadc_misses": int((mh & ~ah).sum()),
        "mlp_misses_aadc_hits": int((~mh & ah).sum()),
        "both_miss": int((~mh & ~ah).sum()),
        "union_hit_fraction_not_combined_metric": float((mh | ah).mean()),
        "top50_negative_overlap": len(abase & top50(mn)),
        "rescue_curve": rows,
    }


def aadc_replay(aa, graph, split):
    """Use upstream primitives and reproduce upstream gate/anchor calibration."""
    train = np.asarray(split["train"]["edge"], dtype=np.int64)
    pos = np.asarray(split["valid"]["edge"], dtype=np.int64)
    neg = np.asarray(split["valid"]["edge_neg"], dtype=np.int64)
    year = float(np.concatenate([graph["edge_year"].flatten(),
        split["train"]["year"].flatten(), split["valid"]["year"].flatten()]).max())
    weights_by_edge = {(int(u), int(v)): float(w) for (u, v), w in
                      zip(graph["edge_index"].T, graph["edge_weight"].flatten())}
    weights = np.array([weights_by_edge.get((int(u), int(v)),
                         weights_by_edge.get((int(v), int(u)), 1.0))
                        for u, v in train], dtype=np.float32)
    weights *= np.power(0.95, year - np.asarray(split["train"]["year"], dtype=np.float32).flatten())
    del weights_by_edge
    adjacency = aa.build_weighted_adj(int(graph["num_nodes"]), train, weights)
    inv = aa.precompute_inv_log_deg(adjacency)
    weighted = aa.precompute_aa_matrix(adjacency, inv)
    neighbors = aa.build_adj(int(graph["num_nodes"]), train)
    lcc = aa.compute_exact_lcc(adjacency)
    kwargs = dict(A=adjacency, A_invlog=weighted, inv_log_deg=inv, adj=neighbors,
                  use_gate=True, ext_threshold=0.5, ext_penalty=0.5, lcc=lcc,
                  use_l3=True, rescue_mode="anchor", anchor_scale=0.0,
                  collect_l3=True, show_progress=False)
    p0 = aa.score_edges(edges=pos, gate_mode="threshold", **kwargs)
    n0 = aa.score_edges(edges=neg, gate_mode="threshold", **kwargs)
    pi, ni = np.flatnonzero(p0[1] > 0), np.flatnonzero(n0[1] > 0)
    pi = pi[np.argsort(p0[0][pi])[-50:]]
    ni = ni[np.argsort(n0[0][ni])[-50:]]
    def avg_lcc(edges):
        return np.array([(lcc[u] + lcc[v]) / 2.0 for u, v in edges], dtype=np.float32)
    gate = aa.calibrate_gate_thresholds(p0[3][pi], avg_lcc(pos[pi]),
                                      n0[3][ni], avg_lcc(neg[ni]),
                                      pos_protection_q=0.95, margin=0.10,
                                      neg_quantiles=(0.33, 0.67))
    # Call upstream scoring again, as its main does, rather than approximate gates.
    p = aa.score_edges(edges=pos, gate_mode="progressive", gate_thresholds=gate, **kwargs)
    n = aa.score_edges(edges=neg, gate_mode="progressive", gate_thresholds=gate, **kwargs)
    opt = aa.optimize_anchor_q(p[0], n[0], p[4]["l3_values"], n[4]["l3_values"],
                              threshold(n[0][n[1] > 0]), K=50)
    kwargs["anchor_scale"] = opt["best_anchor_scale"]
    p = aa.score_edges(edges=pos, gate_mode="progressive", gate_thresholds=gate, **kwargs)
    n = aa.score_edges(edges=neg, gate_mode="progressive", gate_thresholds=gate, **kwargs)
    return p[0], n[0], p[1], n[1], {"gate": gate, "anchor_q": opt["best_q"],
        "anchor_scale": opt["best_anchor_scale"], "decay_reference_year": year,
        "training_adjacency_nnz": int(adjacency.nnz)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    parser.add_argument("--mlp-root", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_mlp_lp/official_v1"))
    parser.add_argument("--aadc-root", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({"seeds": [0, 1, 2], "alphas": ALPHAS, "split": "valid",
                   "classification": "post_hoc_validation_diagnostic", "test_scored": False})
    print(json.dumps(config), flush=True)
    if args.dry_run:
        return
    import torch
    import wandb
    torch.set_num_threads(args.threads)
    shared = module(Path(__file__).parents[1] / "ogbl_collab_mlp_lp/run.py", "collab_mlp")
    upstream_path = args.aadc_root / "upstream"
    rev = subprocess.check_output(["git", "-C", str(upstream_path), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM:
        raise ValueError("upstream revision mismatch")
    source = upstream_path / "aa_dc.py"
    committed = subprocess.check_output(["git", "-C", str(upstream_path), "show", "HEAD:aa_dc.py"])
    if source.read_bytes() != committed:
        raise ValueError("upstream source has local changes")
    aa = module(source, "aadc")
    args.out.mkdir(parents=True, exist_ok=False)
    config["revision"] = shared.git_revision()
    (args.out / "protocol.json").write_text(json.dumps(config, indent=2) + "\n")
    run = wandb.init(project="ogbl-collab-complementarity", group="validation_v1",
                     name="aadc_nonlinear_mlp_seeds012", mode="offline",
                     dir=str(args.out), config=config)
    start = time.monotonic()
    graph, split, evaluator, version = shared.load_official_dataset(args.dataset_root)
    audit = shared.audit_dataset(graph, split, version)
    if audit["split_fingerprint"] != FINGERPRINT:
        raise ValueError("dataset fingerprint mismatch")
    # Dataset audit fingerprints the existing split arrays; no test score is computed.
    del split["test"]
    ap, an, raw_ap, raw_an, calibration = aadc_replay(aa, graph, split)
    evaluator.K = 50
    ah = evaluator.eval({"y_pred_pos": ap, "y_pred_neg": an})["hits@50"]
    recorded = json.loads((args.aadc_root / "results.json").read_text())
    expected_aa = recorded["validation"]["hits@50"]
    if abs(ah - expected_aa) > 5e-7:
        raise ValueError(f"AA-DC replay mismatch: {ah} vs {expected_aa}")
    arrays = {"positive_edges": split["valid"]["edge"], "negative_edges": split["valid"]["edge_neg"],
              "aadc_positive": ap, "aadc_negative": an, "raw_aa_positive": raw_ap,
              "raw_aa_negative": raw_an}
    x = torch.as_tensor(graph["node_feat"], dtype=torch.float32, device=args.device)
    summaries, checks = [], []
    for seed in (0, 1, 2):
        seed_root = args.mlp_root / f"seed{seed}"
        selection_path = seed_root / "selection_frozen.json"
        result = json.loads((seed_root / "results.json").read_text())
        assert result["complete"] and result["seed"] == seed
        assert result["dataset_fingerprint"] == FINGERPRINT
        assert shared.sha256_file(selection_path) == result["selection_frozen_sha256"]
        selection = json.loads(selection_path.read_text())
        chosen = next(s for s in selection["learned"] if s["arm"] == "nonlinear_mlp_cosine")
        path = seed_root / "nonlinear_mlp_cosine.pt"
        assert shared.sha256_file(path) == chosen["checkpoint_sha256"]
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        assert checkpoint["seed"] == seed and checkpoint["epoch"] == chosen["best_epoch"]
        assert checkpoint["arm"] == "nonlinear_mlp_cosine"
        model = shared.Encoder("nonlinear_mlp_cosine").to(args.device)
        model.load_state_dict(checkpoint["state_dict"])
        mp = shared.score_edges(x, split["valid"]["edge"], 8192, model)
        mn = shared.score_edges(x, split["valid"]["edge_neg"], 8192, model)
        mh = evaluator.eval({"y_pred_pos": mp, "y_pred_neg": mn})["hits@50"]
        assert abs(mh - chosen["best_validation_hits_at_50"]) <= 1 / len(mp)
        summary = compare(ap, an, mp, mn)
        assert summary["mlp_hits_at_50"] == mh and summary["aadc_hits_at_50"] == ah
        summary["seed"] = seed
        summary["recovered_by_raw_aa_stratum"] = {
            "aa_zero": int(((ap <= threshold(an)) & (mp > threshold(mn)) & (raw_ap == 0)).sum()),
            "aa_positive": int(((ap <= threshold(an)) & (mp > threshold(mn)) & (raw_ap > 0)).sum())}
        summaries.append(summary)
        checks.append({"seed": seed, "checkpoint": str(path), "checkpoint_sha256": chosen["checkpoint_sha256"],
                       "epoch": checkpoint["epoch"], "producer_revision": result["revision"],
                       "selection_sha256_verified": True, "checkpoint_hash_verified": True,
                       "recorded_validation_hits_at_50": chosen["best_validation_hits_at_50"],
                       "replayed_validation_hits_at_50": mh})
        arrays[f"mlp{seed}_positive"], arrays[f"mlp{seed}_negative"] = mp, mn
        print(json.dumps({"seed": seed, **summary}), flush=True)
        run.log({"seed": seed, "mlp_validation_hits_at_50": mh,
                 "recovered_positives": summary["mlp_recovers_aadc_misses"]})
    np.savez_compressed(args.out / "validation_scores.npz", **arrays)
    payload = {"complete": True, "classification": config["classification"], "split": "valid",
               "test_scored": False, "positive_count": len(ap), "negative_count": len(an),
               "dataset_fingerprint": FINGERPRINT, "calibration": calibration, "seeds": summaries}
    receipt = {"complete": True, "classification": config["classification"], "revision": config["revision"],
               "runtime_root": str(args.out), "upstream_revision": rev,
               "upstream_source_verified": True, "aadc_recorded_validation": expected_aa,
               "aadc_replayed_validation": ah, "dataset_fingerprint": FINGERPRINT,
               "validation_pair_fingerprint": shared.fingerprint(arrays["positive_edges"], arrays["negative_edges"]),
               "checkpoint_checks": checks, "expected_seeds": [0, 1, 2], "observed_seeds": [0, 1, 2],
               "elapsed_seconds": time.monotonic() - start, "device": args.device,
               "test_scored": False, "score_archive_sha256": shared.sha256_file(args.out / "validation_scores.npz"),
               "wandb_directory": run.dir,
               "qualification": "Same validation panel previously used for checkpoint selection and AA-DC calibration; no independent holdout claim."}
    (args.out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "validation_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    run.summary.update({"complete": True, "elapsed_seconds": receipt["elapsed_seconds"], "aadc_hits_at_50": ah})
    run.finish()
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
