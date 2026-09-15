#!/usr/bin/env python3
"""Bounded 2017-selected, 2018-forward novel-pair selector audit."""
import argparse
import json
from pathlib import Path

import numpy as np

import audit as base

FEATURES = [
    "log_aa", "log_l3", "external_ratio", "lcc_min", "lcc_max",
    "log_degree_min", "log_degree_max", "raw_cosine", "log_prior_pair_events",
    "last_event_age", "log_recent_activity_min", "log_recent_activity_max", "aa_zero",
]


def contract():
    return {
        "name": "ogbl_collab_regime_h1_selector_v1",
        "revision": base.revision(),
        "research_question": "Can a small observable subdivision of novel pairs select AA versus joint across years?",
        "selection_year": 2017,
        "forward_year": 2018,
        "fixed_rule_order": [
            "aa_all", "joint_all_novel", "aa_zero", "nonzero_aa", "l3_positive",
            "recent_both", "aa_zero_and_recent", "l3_and_recent",
            "cosine_q50", "cosine_q75", "degree_min_q50", "degree_min_q75",
            "joint_minus_aa_positive", "joint_minus_aa_q75",
        ],
        "thresholds": "2017 pooled-candidate median/75th percentile; recomputed per seed only for score disagreement",
        "selection": "highest 2017 Hits@50; fixed rule order breaks exact ties",
        "routing": "repeat pairs always AA; selected condition routes novel pairs to joint; all other pairs use AA",
        "calibration": "per-expert empirical CDF over pooled 2017 candidate scores, frozen for 2018",
        "advance": {
            "forward_gain_over_aa_every_seed": 0.005,
            "forward_repeat_recall_loss_vs_aa_max": 0.001,
            "forward_novel_net_hits_positive_every_seed": True,
        },
        "stop": "do not train a learned gate unless every condition passes",
        "test_2019_access": False,
        "qualification": "bounded selector diagnostic; 2018 outcome opened only after per-seed 2017 rules are frozen",
    }


def pooled_threshold(panel, feature, quantile):
    values = np.concatenate([panel["pfeatures"][:, feature], panel["nfeatures"][:, feature]])
    novel = np.concatenate([panel["pfeatures"][:, 8] == 0, panel["nfeatures"][:, 8] == 0])
    return float(np.quantile(values[novel], quantile))


def rules(panel, aa_p, aa_n, joint_p, joint_n):
    p, n = panel["pfeatures"], panel["nfeatures"]
    cosine50, cosine75 = pooled_threshold(panel, 7, .5), pooled_threshold(panel, 7, .75)
    degree50, degree75 = pooled_threshold(panel, 5, .5), pooled_threshold(panel, 5, .75)
    disagreement = np.concatenate([joint_p - aa_p, joint_n - aa_n])
    novel = np.concatenate([p[:, 8] == 0, n[:, 8] == 0])
    disagreement75 = float(np.quantile(disagreement[novel], .75))

    def pair(predicate):
        return predicate(p), predicate(n)

    return [
        ("aa_all", pair(lambda f: np.zeros(len(f), dtype=bool))),
        ("joint_all_novel", pair(lambda f: np.ones(len(f), dtype=bool))),
        ("aa_zero", pair(lambda f: f[:, 12] > .5)),
        ("nonzero_aa", pair(lambda f: f[:, 12] <= .5)),
        ("l3_positive", pair(lambda f: f[:, 1] > 0)),
        ("recent_both", pair(lambda f: f[:, 10] > 0)),
        ("aa_zero_and_recent", pair(lambda f: (f[:, 12] > .5) & (f[:, 10] > 0))),
        ("l3_and_recent", pair(lambda f: (f[:, 1] > 0) & (f[:, 10] > 0))),
        ("cosine_q50", pair(lambda f: f[:, 7] >= cosine50)),
        ("cosine_q75", pair(lambda f: f[:, 7] >= cosine75)),
        ("degree_min_q50", pair(lambda f: f[:, 5] >= degree50)),
        ("degree_min_q75", pair(lambda f: f[:, 5] >= degree75)),
        ("joint_minus_aa_positive", ((joint_p - aa_p) > 0, (joint_n - aa_n) > 0)),
        ("joint_minus_aa_q75", ((joint_p - aa_p) >= disagreement75, (joint_n - aa_n) >= disagreement75)),
    ], {"cosine_q50": cosine50, "cosine_q75": cosine75,
        "degree_min_q50": degree50, "degree_min_q75": degree75,
        "joint_minus_aa_q75": disagreement75}


def score_rule(panel, aa_p, aa_n, joint_p, joint_n, masks):
    repeat_p = panel["pfeatures"][:, 8] > 0
    repeat_n = panel["nfeatures"][:, 8] > 0
    choose_p = (~repeat_p) & masks[0]
    choose_n = (~repeat_n) & masks[1]
    mix_p = np.where(choose_p, joint_p, aa_p)
    mix_n = np.where(choose_n, joint_n, aa_n)
    value, mask, cutoff = base.hits(mix_p, mix_n)
    aa_value, aa_mask, _ = base.hits(aa_p, aa_n)
    return {
        "hits_at_50": value,
        "gain_over_aa": value - aa_value,
        "cutoff": cutoff,
        "joint_routed_positive": int(choose_p.sum()),
        "joint_routed_negative": int(choose_n.sum()),
        "repeat_recall_change_vs_aa": base.recall(mask, repeat_p) - base.recall(aa_mask, repeat_p),
        "novel_net_hits_vs_aa": int(mask[~repeat_p].sum()) - int(aa_mask[~repeat_p].sum()),
    }


def calibrated(seed, p17, p18, s17, s18):
    aa_ref = base.fit_cdf(np.concatenate([p17["bp"], p17["bn"]]))
    joint_ref = base.fit_cdf(np.concatenate([s17[f"joint_seed{seed}_p"], s17[f"joint_seed{seed}_n"]]))
    def one(panel, scores):
        return (
            base.apply_cdf(aa_ref, panel["bp"]), base.apply_cdf(aa_ref, panel["bn"]),
            base.apply_cdf(joint_ref, scores[f"joint_seed{seed}_p"]),
            base.apply_cdf(joint_ref, scores[f"joint_seed{seed}_n"]),
        )
    return one(p17, s17), one(p18, s18)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/dataMeR1/phil/gfm/ogbl_collab_compact_joint"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = contract()
    if args.dry_run:
        print(json.dumps(cfg, indent=2)); return
    if args.out is None:
        ap.error("--out is required unless --dry-run is used")
    args.out.mkdir(parents=True, exist_ok=False)
    paths = base.load_inputs(args.root)
    p17, p18 = np.load(paths["panel2017"]), np.load(paths["panel2018"])
    s17, s18 = np.load(paths["scores2017"]), np.load(paths["scores2018"])
    rows = []
    for seed in base.SEEDS:
        c17, c18 = calibrated(seed, p17, p18, s17, s18)
        candidates, thresholds = rules(p17, *c17)
        selection_rows = []
        for order, (name, masks) in enumerate(candidates):
            selection_rows.append({"rule": name, "order": order, **score_rule(p17, *c17, masks)})
        chosen = max(selection_rows, key=lambda row: (row["hits_at_50"], -row["order"]))
        selected_masks = dict(candidates)[chosen["rule"]]
        # Recreate the selected rule on 2018 using thresholds frozen from 2017.
        def masks_for_2018(name):
            p, n = p18["pfeatures"], p18["nfeatures"]
            pred = {
                "aa_all": lambda f, side: np.zeros(len(f), dtype=bool),
                "joint_all_novel": lambda f, side: np.ones(len(f), dtype=bool),
                "aa_zero": lambda f, side: f[:, 12] > .5,
                "nonzero_aa": lambda f, side: f[:, 12] <= .5,
                "l3_positive": lambda f, side: f[:, 1] > 0,
                "recent_both": lambda f, side: f[:, 10] > 0,
                "aa_zero_and_recent": lambda f, side: (f[:, 12] > .5) & (f[:, 10] > 0),
                "l3_and_recent": lambda f, side: (f[:, 1] > 0) & (f[:, 10] > 0),
                "cosine_q50": lambda f, side: f[:, 7] >= thresholds["cosine_q50"],
                "cosine_q75": lambda f, side: f[:, 7] >= thresholds["cosine_q75"],
                "degree_min_q50": lambda f, side: f[:, 5] >= thresholds["degree_min_q50"],
                "degree_min_q75": lambda f, side: f[:, 5] >= thresholds["degree_min_q75"],
                "joint_minus_aa_positive": lambda f, side: (c18[2 if side == 0 else 3] - c18[side]) > 0,
                "joint_minus_aa_q75": lambda f, side: (c18[2 if side == 0 else 3] - c18[side]) >= thresholds["joint_minus_aa_q75"],
            }[name]
            return pred(p, 0), pred(n, 1)
        forward = score_rule(p18, *c18, masks_for_2018(chosen["rule"]))
        rows.append({"seed": seed, "selected_rule": chosen["rule"], "thresholds": thresholds,
                     "selection": chosen, "forward": forward, "all_selection_candidates": selection_rows})
    conditions = {
        "forward_gain_over_aa_every_seed": all(r["forward"]["gain_over_aa"] >= .005 for r in rows),
        "forward_repeat_recall_loss_vs_aa_max": all(r["forward"]["repeat_recall_change_vs_aa"] >= -.001 for r in rows),
        "forward_novel_net_hits_positive_every_seed": all(r["forward"]["novel_net_hits_vs_aa"] > 0 for r in rows),
    }
    result = {"complete": True, "revision": base.revision(), "protocol": cfg, "rows": rows,
              "summary": {"mean_forward_hits_at_50": float(np.mean([r["forward"]["hits_at_50"] for r in rows])),
                          "mean_forward_gain_over_aa": float(np.mean([r["forward"]["gain_over_aa"] for r in rows]))},
              "conditions": conditions, "advance": all(conditions.values()),
              "decision": "advance to protected residual" if all(conditions.values()) else "stop selector; do not train gate",
              "test_2019_scored": False}
    (args.out / "protocol.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (args.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

