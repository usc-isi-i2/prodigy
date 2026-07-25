#!/usr/bin/env python3
"""Build the analysis-ready VALID dataset for the mixed-objective (multitask SSL) experiments.

Inputs (all local / in-repo, no cluster access needed):
  data/pair_lp/*__pair_lp.csv        rescored link prediction (valid evaluator, slpfix @ 79e173a)
  ../multitask_ssl_pairs/data/combined_all_arms.csv   7-arm lattice cls/reg (+ VOID old sLP rows)

Outputs (written to data/):
  cls_reg_7arms.csv          classification + regression rows only (old sLP rows dropped)
  link_prediction_valid.csv  arm-level LP, all negative kinds, with best-heuristic-floor columns
  combined_valid.csv         one tidy long table: every valid (source, model, task, dataset, metric)

Run:  /opt/homebrew/bin/python3.11 build_valid_dataset.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PAIR_LP_DIR = HERE / "data" / "pair_lp"
LATTICE_CSV = HERE.parent / "multitask_ssl_pairs" / "data" / "combined_all_arms.csv"
OUT_DIR = HERE / "data"

# Arm-name schema in the rescored LP CSVs: <source>_<MODEL> where source is the
# checkpoint family (mtr = rotation run, mtp = pairs run, msc_cov / msc_all8 = corpora
# replication) and MODEL is the objective combination.
SOURCES = ("msc_cov", "msc_all8", "mtr", "mtp")  # longest prefixes first
MODEL_K = {"NM": 1, "CL": 1, "FP": 1, "NMCL": 2, "NMFP": 2, "CLFP": 2, "MIX": 3}
GROUP = {1: "single", 2: "pair", 3: "triple"}


def split_arm(arm: str) -> tuple[str, str]:
    for src in SOURCES:
        if arm.startswith(src + "_"):
            return src, arm[len(src) + 1 :]
    raise ValueError(f"unrecognised arm name: {arm}")


def build_cls_reg() -> pd.DataFrame:
    df = pd.read_csv(LATTICE_CSV)
    df = df[df["task"] != "static_link_prediction"].copy()  # old sLP rows are VOID
    df["source"] = df["model"].map(lambda m: "mtr" if m in ("NM", "CL", "FP", "MIX") else "mtp")
    cols = ["source", "group", "k", "model", "task", "dataset", "shots", "split",
            "roc_auc", "accuracy", "f1", "spearman", "rmse", "mae", "r2", "mse", "run"]
    return df[cols].sort_values(["task", "model", "dataset"]).reset_index(drop=True)


def build_lp() -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(p) for p in sorted(PAIR_LP_DIR.glob("*__pair_lp.csv"))],
                    ignore_index=True)

    floors = raw[raw["model"] == "__floor__"]
    best_floor = (floors.loc[floors.groupby(["dataset", "negative_kind"])["auc"].idxmax(),
                             ["dataset", "negative_kind", "scorer", "auc"]]
                  .rename(columns={"scorer": "best_floor_name", "auc": "best_floor_auc"}))

    arms = raw[raw["model"] != "__floor__"].copy()
    arms[["source", "model_short"]] = arms["model"].map(split_arm).apply(pd.Series)
    arms = arms.drop(columns=["model"]).rename(columns={"model_short": "model"})
    arms["k"] = arms["model"].map(MODEL_K)
    arms["group"] = arms["k"].map(GROUP)

    lp = arms.merge(best_floor, on=["dataset", "negative_kind"], how="left")
    lp["margin_vs_floor"] = lp["auc"] - lp["best_floor_auc"]
    cols = ["source", "group", "k", "model", "dataset", "negative_kind", "scorer",
            "auc", "average_precision", "hits_at_50", "best_floor_name", "best_floor_auc",
            "margin_vs_floor", "n_pairs", "n_positive", "orientation",
            "endpoint_permutation_auc", "endpoint_sensitivity", "leakage_edges"]
    return lp[cols].sort_values(["negative_kind", "source", "model", "dataset"]).reset_index(drop=True)


def build_combined(cls_reg: pd.DataFrame, lp: pd.DataFrame) -> pd.DataFrame:
    base = ["source", "group", "k", "model", "task", "dataset", "metric", "value"]
    rows = []

    cls = cls_reg[cls_reg["task"] == "classification"]
    rows.append(cls.assign(metric="roc_auc", value=cls["roc_auc"])[base])

    reg = cls_reg[cls_reg["task"] == "regression"]
    rows.append(reg.assign(metric="spearman", value=reg["spearman"])[base])

    lp_dm = lp[lp["negative_kind"] == "degree_matched"].assign(task="link_prediction")
    rows.append(lp_dm.assign(metric="auc", value=lp_dm["auc"])[base])
    rows.append(lp_dm.assign(metric="margin_vs_floor", value=lp_dm["margin_vs_floor"])[base])

    return (pd.concat(rows, ignore_index=True)
            .sort_values(["task", "metric", "source", "model", "dataset"])
            .reset_index(drop=True))


def main() -> None:
    cls_reg = build_cls_reg()
    lp = build_lp()
    combined = build_combined(cls_reg, lp)

    cls_reg.to_csv(OUT_DIR / "cls_reg_7arms.csv", index=False)
    lp.to_csv(OUT_DIR / "link_prediction_valid.csv", index=False)
    combined.to_csv(OUT_DIR / "combined_valid.csv", index=False)

    print(f"cls_reg_7arms.csv          {len(cls_reg):4d} rows")
    print(f"link_prediction_valid.csv  {len(lp):4d} rows")
    print(f"combined_valid.csv         {len(combined):4d} rows")

    check = (lp[lp["negative_kind"] == "degree_matched"]
             .groupby(["source", "model"])[["auc", "margin_vs_floor"]].mean()
             .sort_values("auc", ascending=False).round(3))
    print("\nSanity check — mean LP AUC (degree-matched), should match FINDINGS_rescore.md:")
    print(check.to_string())


if __name__ == "__main__":
    main()
