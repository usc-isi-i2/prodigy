#!/usr/bin/env python3
"""Frozen no-training prerequisite for a repeat/novel Collab mixture."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SEEDS = (0, 1, 2)
TARGET = 0.7129


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def revision():
    root = Path(__file__).resolve().parents[4]
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def hits(pos, neg):
    cutoff = np.sort(neg)[-50]
    mask = pos > cutoff
    return float(mask.mean()), mask, float(cutoff)


def fit_cdf(values):
    return np.sort(np.asarray(values, dtype=np.float64))


def apply_cdf(reference, values):
    return np.searchsorted(reference, values, side="right").astype(np.float64) / len(reference)


def recall(mask, regime):
    return float(mask[regime].mean()) if np.any(regime) else None


def protocol():
    return {
        "name": "ogbl_collab_regime_h1_prerequisite_v1",
        "revision": revision(),
        "research_question": "Does a fixed observable repeat/novel expert split transfer from 2017 to 2018?",
        "years": {"calibration_and_selection": 2017, "forward_assessment": 2018},
        "experts": ["AA-DC", "seed-matched compact joint"],
        "gate": "AA-DC if pair appeared before target year; joint otherwise",
        "calibration": "per-expert empirical CDF over pooled 2017 candidate scores, frozen for 2018",
        "metric": "Hits@50 with cutoff recomputed after mixing",
        "primary_control": "seed-matched joint scorer",
        "secondary_controls": ["AA-DC", "better single expert on each panel"],
        "advance": {
            "all_seed_gains_over_joint_positive_both_years": True,
            "mean_gain_over_joint_each_year": 0.005,
            "forward_repeat_recall_loss_vs_aa_max": 0.001,
            "forward_gain_over_better_single": 0.005,
        },
        "stop": "stop before training if any advancement condition fails",
        "test_2019_access": False,
        "qualification": "mechanism prerequisite only; prior 2019 exploration disclosed",
    }


def load_inputs(root):
    paths = {
        "scores2017": root / "fusion_frozen_v1/scores2017.npz",
        "panel2017": root / "joint_v1/year2017.npz",
        "scores2018": root / "joint_v1/assessment/scores.npz",
        "panel2018": root / "joint_v1/assessment/year2018.npz",
        "assessment": root / "joint_v1/assessment/results.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def evaluate_seed(seed, panels, scores):
    p17, p18 = panels
    s17, s18 = scores
    aa_ref = fit_cdf(np.concatenate([p17["bp"], p17["bn"]]))
    joint_ref = fit_cdf(np.concatenate([s17[f"joint_seed{seed}_p"], s17[f"joint_seed{seed}_n"]]))

    def evaluate(panel, score, year):
        repeat_p = panel["pfeatures"][:, 8] > 0
        repeat_n = panel["nfeatures"][:, 8] > 0
        aa_p = apply_cdf(aa_ref, panel["bp"])
        aa_n = apply_cdf(aa_ref, panel["bn"])
        joint_p = apply_cdf(joint_ref, score[f"joint_seed{seed}_p"])
        joint_n = apply_cdf(joint_ref, score[f"joint_seed{seed}_n"])
        mix_p = np.where(repeat_p, aa_p, joint_p)
        mix_n = np.where(repeat_n, aa_n, joint_n)
        aa_h, aa_mask, aa_cut = hits(aa_p, aa_n)
        joint_h, joint_mask, joint_cut = hits(joint_p, joint_n)
        mix_h, mix_mask, mix_cut = hits(mix_p, mix_n)
        return {
            "year": year,
            "seed": seed,
            "aa_hits_at_50": aa_h,
            "joint_hits_at_50": joint_h,
            "mixture_hits_at_50": mix_h,
            "gain_over_joint": mix_h - joint_h,
            "gain_over_aa": mix_h - aa_h,
            "gain_over_better_single": mix_h - max(aa_h, joint_h),
            "aa_repeat_recall": recall(aa_mask, repeat_p),
            "mixture_repeat_recall": recall(mix_mask, repeat_p),
            "repeat_recall_change_vs_aa": recall(mix_mask, repeat_p) - recall(aa_mask, repeat_p),
            "aa_novel_recall": recall(aa_mask, ~repeat_p),
            "joint_novel_recall": recall(joint_mask, ~repeat_p),
            "mixture_novel_recall": recall(mix_mask, ~repeat_p),
            "cutoffs": {"aa": aa_cut, "joint": joint_cut, "mixture": mix_cut},
            "counts": {"positive": len(repeat_p), "negative": len(repeat_n),
                       "repeat_positive": int(repeat_p.sum()), "repeat_negative": int(repeat_n.sum())},
        }

    return evaluate(p17, s17, 2017), evaluate(p18, s18, 2018)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = protocol()
    if args.dry_run:
        print(json.dumps(cfg, indent=2))
        return
    if args.out is None:
        ap.error("--out is required unless --dry-run is used")
    args.out.mkdir(parents=True, exist_ok=False)
    paths = load_inputs(args.root)
    panel17, panel18 = np.load(paths["panel2017"]), np.load(paths["panel2018"])
    score17, score18 = np.load(paths["scores2017"]), np.load(paths["scores2018"])
    rows = []
    for seed in SEEDS:
        rows.extend(evaluate_seed(seed, (panel17, panel18), (score17, score18)))
    by_year = {}
    for year in (2017, 2018):
        selected = [row for row in rows if row["year"] == year]
        by_year[str(year)] = {
            "mean_mixture": float(np.mean([r["mixture_hits_at_50"] for r in selected])),
            "mean_joint": float(np.mean([r["joint_hits_at_50"] for r in selected])),
            "mean_gain_over_joint": float(np.mean([r["gain_over_joint"] for r in selected])),
            "mean_gain_over_better_single": float(np.mean([r["gain_over_better_single"] for r in selected])),
        }
    conditions = {
        "all_seed_gains_over_joint_positive_both_years": all(r["gain_over_joint"] > 0 for r in rows),
        "mean_gain_over_joint_each_year": all(by_year[str(y)]["mean_gain_over_joint"] >= 0.005 for y in (2017, 2018)),
        "forward_repeat_recall_loss_vs_aa_max": all(r["repeat_recall_change_vs_aa"] >= -0.001 for r in rows if r["year"] == 2018),
        "forward_gain_over_better_single": by_year["2018"]["mean_gain_over_better_single"] >= 0.005,
    }
    result = {
        "complete": True,
        "revision": revision(),
        "protocol": cfg,
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in paths.items()},
        "rows": rows,
        "summary": by_year,
        "conditions": conditions,
        "advance": all(conditions.values()),
        "decision": "advance to bounded H1 training" if all(conditions.values()) else "stop H1 before training",
        "target_reference_only": TARGET,
        "test_2019_scored": False,
    }
    (args.out / "protocol.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

