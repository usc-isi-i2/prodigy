#!/usr/bin/env python3
"""Validation-frozen fusion of compact Collab experts; never scores test."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE_REVISION = "be46110e774938c6b567a5340395ea363535ae24"
SEEDS = (0, 1, 2)
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
RULES = ("structure", "joint", "both")
SOURCE_COUNTS = {"structure": 36_865, "joint": 496_385}
TOTAL_COUNTS = {"structure": 36_910, "joint": 496_430, "both": 533_296}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(obj)
    return obj


joint = load_module(HERE.parent / "ogbl_collab_compact_joint/run.py", "compact_joint")


def read(path: Path):
    return json.loads(path.read_text())


def save(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def revision() -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def cutoff(scores: np.ndarray) -> float:
    return float(np.partition(scores, -50)[-50])


def hits(positive: np.ndarray, negative: np.ndarray) -> float:
    return float(np.mean(positive > cutoff(negative)))


def contract() -> dict:
    return {
        "name": "compact_fusion_v1",
        "revision": revision(),
        "source_revision": SOURCE_REVISION,
        "seeds": list(SEEDS),
        "rules": list(RULES),
        "alpha_grid": list(ALPHAS),
        "selection": "one common alpha tuple per rule maximizes mean 2017 Hits@50 across seeds; lowest total weight then lexicographic tie break",
        "calibration": "all AA and expert scale cutoffs are computed on 2017 negatives and frozen for 2018",
        "score": "max(base/base2017_negative50, alpha_s*exp(structure-structure2017_negative50), alpha_j*exp(joint-joint2017_negative50))",
        "primary_estimand": "both-expert fusion minus frozen-2015-calibrated AA-DC on official 2018 validation",
        "controls": ["AA-DC", "AA+structure", "AA+joint"],
        "total_inference_learned_scalars": TOTAL_COUNTS,
        "advancement_rule": "both fusion exceeds AA-DC by at least 1 point and exceeds both single-expert controls in every seed",
        "assessment_previously_inspected": True,
        "test_scored": False,
        "qualification": "post-hoc development campaign; reuses selected compact-joint checkpoints; no assessment-set scale normalization",
    }


def verify_source(source: Path) -> tuple[dict, list[dict]]:
    prepared = read(source / "prepared.json")
    assert prepared["complete"] and prepared["revision"] == SOURCE_REVISION
    selections = []
    for arm in ("structure", "joint"):
        for seed in SEEDS:
            path = source / f"{arm}_seed{seed}"
            selected = read(path / "selection.json")
            assert selected["complete"] and selected["revision"] == SOURCE_REVISION
            assert selected["parameters"] == SOURCE_COUNTS[arm]
            assert sha256(path / "best.pt") == selected["checkpoint_sha256"]
            selections.append(selected)
    assert max(TOTAL_COUNTS.values()) < 1_000_000
    return prepared, selections


def expert_scores(source: Path, year: int, device: str) -> tuple[dict, dict[str, np.ndarray]]:
    prepared, selections = verify_source(source)
    panel = np.load(source / f"year{year}.npz")
    standardization = np.load(source / "standardization.npz")
    tensors = joint.panel_tensors(
        panel, standardization["mean"], standardization["std"], device
    )
    x = torch.as_tensor(np.load(source / "node_features.npy"), device=device)
    adj = joint.adjacency(panel["graph_edges"], len(x), device)
    values = {}
    for selected in selections:
        arm, seed = selected["arm"], selected["seed"]
        model = joint.Model(arm).to(device)
        state = torch.load(
            source / f"{arm}_seed{seed}" / "best.pt",
            map_location=device,
            weights_only=False,
        )
        assert state["revision"] == SOURCE_REVISION
        assert state["step"] == selected["selected_step"]
        model.load_state_dict(state["model"], strict=True)
        prediction = joint.evaluate(model, x, adj if arm == "joint" else None, tensors)
        values[f"{arm}_seed{seed}_p"] = prediction["p"]
        values[f"{arm}_seed{seed}_n"] = prediction["n"]
    return {key: panel[key] for key in panel.files}, values


def fuse(
    base: np.ndarray,
    base_scale: float,
    structure: np.ndarray,
    structure_scale: float,
    joint_score: np.ndarray,
    joint_scale: float,
    alpha_structure: float,
    alpha_joint: float,
) -> np.ndarray:
    output = base / base_scale
    if alpha_structure:
        output = np.maximum(
            output,
            alpha_structure * np.exp(np.clip(structure - structure_scale, -30, 30)),
        )
    if alpha_joint:
        output = np.maximum(
            output,
            alpha_joint * np.exp(np.clip(joint_score - joint_scale, -30, 30)),
        )
    return output


def candidates(rule: str):
    if rule == "structure":
        return ((alpha, 0.0) for alpha in ALPHAS)
    if rule == "joint":
        return ((0.0, alpha) for alpha in ALPHAS)
    return ((a_structure, a_joint) for a_structure in ALPHAS for a_joint in ALPHAS)


def evaluate_tuple(panel: dict, scores: dict[str, np.ndarray], scales: dict, pair: tuple[float, float]):
    a_structure, a_joint = pair
    values = []
    for seed in SEEDS:
        positive = fuse(
            panel["bp"], scales["base"], scores[f"structure_seed{seed}_p"],
            scales[f"structure_seed{seed}"], scores[f"joint_seed{seed}_p"],
            scales[f"joint_seed{seed}"], a_structure, a_joint,
        )
        negative = fuse(
            panel["bn"], scales["base"], scores[f"structure_seed{seed}_n"],
            scales[f"structure_seed{seed}"], scores[f"joint_seed{seed}_n"],
            scales[f"joint_seed{seed}"], a_structure, a_joint,
        )
        values.append(hits(positive, negative))
    return values


def select(args) -> None:
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "protocol.json", contract())
    panel, scores = expert_scores(args.source, 2017, args.device)
    scales = {"base": cutoff(panel["bn"])}
    for arm in ("structure", "joint"):
        for seed in SEEDS:
            scales[f"{arm}_seed{seed}"] = cutoff(scores[f"{arm}_seed{seed}_n"])
    rows, selections = [], {}
    for rule in RULES:
        for pair in candidates(rule):
            values = evaluate_tuple(panel, scores, scales, pair)
            rows.append({
                "rule": rule, "alpha_structure": pair[0], "alpha_joint": pair[1],
                "seed_hits_at_50": values, "mean_hits_at_50": float(np.mean(values)),
            })
        eligible = [row for row in rows if row["rule"] == rule]
        selections[rule] = min(
            eligible,
            key=lambda row: (-row["mean_hits_at_50"], row["alpha_structure"] + row["alpha_joint"],
                             row["alpha_structure"], row["alpha_joint"]),
        )
    np.savez_compressed(args.out / "scores2017.npz", **scores)
    result = {
        "complete": True, "test_scored": False, "revision": revision(),
        "source_revision": SOURCE_REVISION, "scales2017": scales,
        "selections": selections, "grid": rows,
        "protocol_sha256": sha256(args.out / "protocol.json"),
        "source_prepared_sha256": sha256(args.source / "prepared.json"),
        "scores2017_sha256": sha256(args.out / "scores2017.npz"),
    }
    save(args.out / "selection_frozen.json", result)
    print(json.dumps({"selections": selections, "scales2017": scales}, indent=2))


def compare(current: np.ndarray, baseline: np.ndarray, repeat: np.ndarray) -> dict:
    return {
        "hits_at_50": float(current.mean()),
        "recovered": int((current & ~baseline).sum()),
        "lost": int((~current & baseline).sum()),
        "net_hits": int(current.sum()) - int(baseline.sum()),
        "repeat_net_hits": int(current[repeat].sum()) - int(baseline[repeat].sum()),
        "new_pair_net_hits": int(current[~repeat].sum()) - int(baseline[~repeat].sum()),
    }


def assess(args) -> None:
    selection = read(args.out / "selection_frozen.json")
    assert selection["complete"] and selection["revision"] == revision()
    assert selection["protocol_sha256"] == sha256(args.out / "protocol.json")
    source_results = read(args.source / "assessment" / "results.json")
    assert source_results["complete"] and source_results["test_scored"] is False
    panel_file = args.source / "assessment" / "year2018.npz"
    scores_file = args.source / "assessment" / "scores.npz"
    assert sha256(panel_file) == source_results["panel_sha256"]
    assert sha256(scores_file) == source_results["score_sha256"]
    panel, scores = np.load(panel_file), np.load(scores_file)
    baseline = hitmask(panel["bp"], panel["bn"])
    official = hitmask(panel["official_bp"], panel["official_bn"])
    repeat = panel["pfeatures"][:, 8] > 0
    rows = []
    for rule in RULES:
        chosen = selection["selections"][rule]
        pair = chosen["alpha_structure"], chosen["alpha_joint"]
        for seed in SEEDS:
            positive = fuse(
                panel["bp"], selection["scales2017"]["base"],
                scores[f"structure_seed{seed}_p"], selection["scales2017"][f"structure_seed{seed}"],
                scores[f"joint_seed{seed}_p"], selection["scales2017"][f"joint_seed{seed}"], *pair,
            )
            negative = fuse(
                panel["bn"], selection["scales2017"]["base"],
                scores[f"structure_seed{seed}_n"], selection["scales2017"][f"structure_seed{seed}"],
                scores[f"joint_seed{seed}_n"], selection["scales2017"][f"joint_seed{seed}"], *pair,
            )
            current = hitmask(positive, negative)
            rows.append({"rule": rule, "seed": seed, **pair_dict(pair), **compare(current, baseline, repeat)})
    summary = {
        rule: {
            "mean": float(np.mean([r["hits_at_50"] for r in rows if r["rule"] == rule])),
            "sample_sd": float(np.std([r["hits_at_50"] for r in rows if r["rule"] == rule], ddof=1)),
        }
        for rule in RULES
    }
    both = [r for r in rows if r["rule"] == "both"]
    passed = (
        summary["both"]["mean"] >= float(baseline.mean()) + 0.01
        and all(r["hits_at_50"] > next(x["hits_at_50"] for x in rows if x["rule"] == "joint" and x["seed"] == r["seed"]) for r in both)
        and all(r["hits_at_50"] > next(x["hits_at_50"] for x in rows if x["rule"] == "structure" and x["seed"] == r["seed"]) for r in both)
    )
    result = {
        "complete": True, "test_scored": False, "revision": revision(), "rows": rows,
        "summary": summary, "frozen_aadc_hits_at_50": float(baseline.mean()),
        "official_2018_calibrated_aadc_hits_at_50": float(official.mean()),
        "advancement_rule_passed": bool(passed),
        "selection_frozen_sha256": sha256(args.out / "selection_frozen.json"),
        "source_results_sha256": sha256(args.source / "assessment" / "results.json"),
    }
    save(args.out / "results.json", result)
    print(json.dumps(result, indent=2))


def pair_dict(pair: tuple[float, float]) -> dict:
    return {"alpha_structure": pair[0], "alpha_joint": pair[1]}


def audit(args) -> None:
    protocol, selection, result = read(args.out / "protocol.json"), read(args.out / "selection_frozen.json"), read(args.out / "results.json")
    assert protocol == contract() and result["complete"] and result["test_scored"] is False
    assert selection["protocol_sha256"] == sha256(args.out / "protocol.json")
    assert result["selection_frozen_sha256"] == sha256(args.out / "selection_frozen.json")
    assert len(result["rows"]) == 9 and {r["rule"] for r in result["rows"]} == set(RULES)
    assert max(protocol["total_inference_learned_scalars"].values()) < 1_000_000
    save(args.out / "audit.json", {
        "complete": True, "selection_and_scale_freeze_verified": True,
        "nine_assessment_cells_verified": True, "parameter_cap_verified": True,
        "test_scored": False, "revision": revision(), "results_sha256": sha256(args.out / "results.json"),
    })
    print((args.out / "audit.json").read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "assess", "audit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint/joint_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(contract(), indent=2)); return
    {"select": select, "assess": assess, "audit": audit}[args.phase](args)


if __name__ == "__main__":
    main()
