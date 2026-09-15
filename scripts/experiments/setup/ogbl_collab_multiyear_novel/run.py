#!/usr/bin/env python3
"""Multi-year temporal training for the compact ogbl-collab joint scorer."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ogbl_collab_compact_joint"))
import run as joint
import candidate
from fusion import hits

YEARS = (2015, 2016, 2017)
SEEDS = (0, 1, 2)
STEPS = 2000


def contract():
    return {
        "name": "ogbl_collab_multiyear_novel_v1",
        "revision": joint.revision(),
        "arm": "multi_year_2015_2016_2017",
        "control": "archived fresh fixed-pool BCE, 2017 only, standalone earliest-max selection",
        "seeds": list(SEEDS),
        "training_years": list(YEARS),
        "schedule": "step t uses training_years[(t-1)%3]; 2000 total updates",
        "temporal_integrity": "each episode uses graph edges strictly before its target year",
        "negative_policy": "deterministic fresh 100k target-year nonpositives; excludes official 2018 negatives",
        "unchanged": "496385 weights; symmetric14+SAGE; BCE; Adam .001 wd .0001 clip5; 2048 positive, 1024 uniform and 1024 hard negatives; hard top2048 refresh every10 visits per year",
        "selection": "2018 standalone Hits@50 every50 global updates; earliest maximum",
        "advance": "all3 seed gains over archived controls positive; mean gain>=.005; novel net hits positive every seed",
        "stop": "no refit or 2019 scoring unless advancement passes",
        "test_2019_access": False,
        "prior_test_exploration_disclosed": True,
    }


def save(path, value):
    joint.save(path, value)


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = contract(); save(args.out / "protocol.json", cfg)
    prepared = joint.read(args.runtime / "prepared.json")
    graph, split, _, version = joint.shared.load_official_dataset(args.dataset_root)
    identity = joint.shared.audit_dataset(graph, split, version)
    assert identity["split_fingerprint"] == joint.gate.FINGERPRINT
    del split["test"]
    aa = joint.module(args.upstream / "aa_dc.py", "multiyear_aa")
    files, metadata = {}, []
    for name in ("node_features.npy", "standardization.npz"):
        path = args.runtime / name
        assert joint.shared.sha256_file(path) == prepared["files"][name]
        files[str(path)] = prepared["files"][name]
    validation = args.runtime / "assessment/year2018.npz"
    validation_sha = joint.read(args.runtime / "assessment/results.json")["panel_sha256"]
    assert joint.shared.sha256_file(validation) == validation_sha
    files[str(validation)] = validation_sha
    ty = split["train"]["year"].flatten()
    for year in YEARS:
        positive = split["train"]["edge"][ty == year]
        neg = candidate.fresh_negatives(positive, split["valid"]["edge_neg"], int(graph["num_nodes"]), year)
        panel, meta, _ = joint.gate.make_year(
            aa, joint.shared, graph, split, year, prepared["calibration"],
            symmetric_features=True, negative_edges=neg,
        )
        panel["graph_edges"] = joint.graph_edges(graph, split, year)
        expected_edges = np.unique(np.sort(split["train"]["edge"][ty < year], axis=1), axis=0)
        np.testing.assert_array_equal(panel["graph_edges"], expected_edges)
        path = args.out / f"train{year}.npz"
        np.savez_compressed(path, **panel)
        files[str(path)] = joint.shared.sha256_file(path)
        meta["official_validation_negative_overlap"] = int(len(np.intersect1d(
            joint.gate.keys(neg, int(graph["num_nodes"])),
            joint.gate.keys(split["valid"]["edge_neg"], int(graph["num_nodes"])),
        )))
        assert meta["official_validation_negative_overlap"] == 0
        metadata.append(meta)
    save(args.out / "prepared.json", {"complete": True, "revision": joint.revision(),
        "protocol_sha256": joint.shared.sha256_file(args.out / "protocol.json"),
        "identity": identity, "files": files, "metadata": metadata})


def verify(args):
    assert joint.read(args.out / "protocol.json") == contract()
    meta = joint.read(args.out / "prepared.json")
    assert meta["complete"] and meta["revision"] == joint.revision()
    for path, sha in meta["files"].items():
        assert joint.shared.sha256_file(Path(path)) == sha
    return meta


def tensors(panel, mean, std, device):
    return joint.panel_tensors(panel, mean, std, device)


def train(args):
    meta = verify(args)
    if args.device not in {f"cuda:{i}" for i in range(4)}:
        raise ValueError("owned GPUs are cuda:0 through cuda:3")
    out = args.out / f"multiyear{args.seed}"; out.mkdir(exist_ok=False)
    torch.set_num_threads(8); torch.cuda.set_device(args.device)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    ms = np.load(args.runtime / "standardization.npz")
    x = torch.as_tensor(np.load(args.runtime / "node_features.npy"), device=args.device)
    panels, adjs = {}, {}
    for year in YEARS:
        panel = np.load(args.out / f"train{year}.npz")
        panels[year] = tensors(panel, ms["mean"], ms["std"], args.device)
        adjs[year] = joint.adjacency(panel["graph_edges"], len(x), args.device)
    va = np.load(args.runtime / "assessment/year2018.npz")
    vp = tensors(va, ms["mean"], ms["std"], args.device)
    vadj = joint.adjacency(va["graph_edges"], len(x), args.device)
    model = joint.Model("joint").to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    hard, visits = {}, {year: 0 for year in YEARS}
    streams = {year: hashlib.sha256() for year in YEARS}
    history, best, started = [], None, time.monotonic()
    import wandb
    cfg = contract() | {"seed": args.seed, "device": args.device}
    save(out / "config.json", cfg)
    run = wandb.init(project="ogbl-collab-compact-joint", group="multiyear-novel-v1",
                     name=out.name, mode="offline", dir=str(out), config=cfg)
    for step in range(1, STEPS + 1):
        year = YEARS[(step - 1) % len(YEARS)]
        tp, adj = panels[year], adjs[year]
        if visits[year] % 10 == 0:
            with torch.no_grad():
                model.eval(); z = model.encode(x, adj)
                hard[year] = joint.scores(model, z, tp["n"]).topk(2048).indices.cpu(); del z
        pi, ni, raw = joint.sampled_indices(generator, len(tp["p"][0]), len(tp["n"][0]), hard[year])
        streams[year].update(raw); visits[year] += 1
        model.train(); optimizer.zero_grad(set_to_none=True); z = model.encode(x, adj)
        values = []
        for side, index in (("p", pi), ("n", ni)):
            index = index.to(args.device); edges, features = tp[side]
            values.append(model.score(z, edges[index], features[index]))
        loss = F.binary_cross_entropy_with_logits(torch.cat(values), torch.cat([
            torch.ones_like(values[0]), torch.zeros_like(values[1])]))
        assert torch.isfinite(loss); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); assert torch.isfinite(grad)
        optimizer.step(); del z, values
        if step % 50 == 0:
            score = joint.evaluate(model, x, vadj, vp); value = hits(**score)
            row = {"step": step, "training_year": year, "loss": float(loss),
                   "validation_hits_at_50": value, "elapsed_seconds": time.monotonic() - started}
            if best is None or value > best["validation_hits_at_50"]:
                best = row.copy(); torch.save({"model": model.state_dict(), "step": step,
                    "seed": args.seed, "revision": joint.revision()}, out / "best.pt")
                np.savez_compressed(out / "best_scores.npz", **score)
            history.append(row); save(out / "history.json", history); run.log(row)
            print(json.dumps({"seed": args.seed, **row, "best": best}), flush=True)
    torch.save({"model": model.state_dict(), "step": STEPS, "seed": args.seed,
                "revision": joint.revision()}, out / "last.pt")
    state = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
    model.load_state_dict(state["model"]); replay = joint.evaluate(model, x, vadj, vp)
    saved = np.load(out / "best_scores.npz")
    for side in ("p", "n"): np.testing.assert_array_equal(replay[side], saved[side])
    result = {"complete": True, "revision": joint.revision(), "seed": args.seed,
        "best": best, "parameters": sum(p.numel() for p in model.parameters()),
        "stream_sha256": {str(y): streams[y].hexdigest() for y in YEARS},
        "visits": {str(y): visits[y] for y in YEARS}, "test_2019_scored": False,
        "wandb_directory": run.dir, "elapsed_seconds": time.monotonic() - started,
        "files": {name: joint.shared.sha256_file(out / name) for name in
                  ("best.pt", "best_scores.npz", "history.json", "last.pt", "config.json")}}
    save(out / "results.json", result); run.finish(); print(json.dumps(result, indent=2))


def aggregate(args):
    verify(args)
    controls = []
    for seed in SEEDS:
        controls.append(joint.read(args.control / f"fixed{seed}/results.json"))
    va = np.load(args.runtime / "assessment/year2018.npz")
    repeat = va["pfeatures"][:, 8] > 0
    rows = []
    for seed, control in zip(SEEDS, controls):
        treatment = joint.read(args.out / f"multiyear{seed}/results.json")
        assert treatment["complete"] and not treatment["test_2019_scored"]
        tc = np.load(args.out / f"multiyear{seed}/best_scores.npz")
        cc = np.load(args.control / f"fixed{seed}/best_scores.npz")
        tm = tc["p"] > np.partition(tc["n"], -50)[-50]
        cm = cc["p"] > np.partition(cc["n"], -50)[-50]
        th, ch = float(tm.mean()), float(cm.mean())
        rows.append({"seed": seed, "control": ch, "treatment": th, "difference": th - ch,
                     "novel_net_hits": int(tm[~repeat].sum()) - int(cm[~repeat].sum()),
                     "repeat_net_hits": int(tm[repeat].sum()) - int(cm[repeat].sum()),
                     "selected_step": treatment["best"]["step"]})
    mean_difference = float(np.mean([r["difference"] for r in rows]))
    conditions = {"all_seed_differences_positive": all(r["difference"] > 0 for r in rows),
                  "mean_gain_at_least_point005": mean_difference >= .005,
                  "novel_net_hits_positive_every_seed": all(r["novel_net_hits"] > 0 for r in rows)}
    result = {"complete": True, "revision": joint.revision(), "rows": rows,
              "mean_control": float(np.mean([r["control"] for r in rows])),
              "mean_treatment": float(np.mean([r["treatment"] for r in rows])),
              "mean_difference": mean_difference, "conditions": conditions,
              "advance": all(conditions.values()),
              "decision": "advance to protected residual" if all(conditions.values()) else "stop multi-year recipe",
              "test_2019_scored": False}
    save(args.out / "summary.json", result); print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("dry-run", "prepare", "train", "aggregate"))
    p.add_argument("--runtime", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1"))
    p.add_argument("--control", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint/negative_renewal_v1"))
    p.add_argument("--out", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint/multiyear_novel_v1"))
    p.add_argument("--dataset-root", type=Path, default=Path("/dataMeR1/phil/data/ogb"))
    p.add_argument("--upstream", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/upstream"))
    p.add_argument("--seed", type=int, choices=SEEDS, default=0)
    p.add_argument("--device", choices=tuple(f"cuda:{i}" for i in range(4)), default="cuda:0")
    args = p.parse_args()
    if args.phase == "dry-run": print(json.dumps(contract(), indent=2)); return
    {"prepare": prepare, "train": train, "aggregate": aggregate}[args.phase](args)


if __name__ == "__main__":
    main()
