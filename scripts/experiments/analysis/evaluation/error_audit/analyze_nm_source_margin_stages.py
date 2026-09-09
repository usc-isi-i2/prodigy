"""Localize native/foreign NM divergence with episode-relative score separation."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


TARGETS = ("cp_hk", "ukr_rus")
MODELS = ("hk", "ukr")
NATIVE = {"cp_hk": "hk", "ukr_rus": "ukr"}
FOREIGN = {"cp_hk": "ukr", "ukr_rus": "hk"}
STAGES = {"pre": "mean_cosine", "final": "native_logits"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def score_stats(scores):
    scores = np.asarray(scores, dtype=np.float64)
    truth = np.repeat(np.arange(30), 4)
    true_score = scores[np.arange(120), truth]
    rival_scores = scores.copy()
    rival_scores[np.arange(120), truth] = -np.inf
    rival_score = rival_scores.max(axis=1)
    prediction = scores.argmax(axis=1)
    rank = 1 + (scores > true_score[:, None]).sum(axis=1)
    scale = scores.std(axis=1)
    if (scale <= 1e-12).any():
        raise ValueError("degenerate candidate score row")
    return prediction, rank, (true_score - rival_score) / scale


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "positive_fraction": float((values > 0).mean()),
    }


def metric_summary(frame, column):
    result = summarize(frame[column])
    result["node_weighted_mean"] = float(frame.groupby("query")[column].mean().mean())
    return result


def transition_counts(frame, model):
    pre = frame[f"pre_correct_{model}"].astype(bool)
    final = frame[f"final_correct_{model}"].astype(bool)
    labels = np.select(
        [pre & final, pre & ~final, ~pre & final],
        ["correct_to_correct", "correct_to_wrong", "wrong_to_correct"],
        default="wrong_to_wrong",
    )
    counts = pd.Series(labels).value_counts()
    return {
        label: {"count": int(counts.get(label, 0)),
                "fraction": float(counts.get(label, 0) / len(frame))}
        for label in ("correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong")
    }


def load_target(root, target):
    csv_path = root / target / "stage_predictions_private.csv"
    stage = pd.read_csv(csv_path)
    stage = stage[["episode", "sample", "query", "anchor", "model",
                   "native_correct", "native_prediction", "native_rank",
                   "encoded_correct", "encoded_prediction", "encoded_rank"]]
    stage["episode"] = stage.episode.astype(str)
    if len(stage) != 122880 or stage.duplicated(["episode", "sample", "model"]).any():
        raise ValueError(f"unexpected stage table for {target}")

    rows = []
    hashes = {"stage_predictions_private.csv": digest(csv_path)}
    slots = np.asarray([slot for slot in range(210) if slot % 7 >= 3])
    for batch_index in range(16):
        path = root / target / f"batch_{batch_index:03d}_embeddings_private.pt"
        hashes[path.name] = digest(path)
        packed = torch.load(path, map_location="cpu", weights_only=False)
        for model in MODELS:
            for episode_index, episode_data in packed["models"][model].items():
                episode_index = int(episode_index)
                base = {
                    "episode": np.repeat(f"{batch_index}:{episode_index}", 120),
                    "sample": episode_index * 210 + slots,
                    "model": np.repeat(model, 120),
                }
                frame = pd.DataFrame(base)
                for prefix, key in STAGES.items():
                    prediction, rank, margin = score_stats(episode_data[key])
                    frame[f"{prefix}_prediction_recomputed"] = prediction
                    frame[f"{prefix}_rank_recomputed"] = rank
                    frame[f"{prefix}_z_margin"] = margin
                rows.append(frame)
    scores = pd.concat(rows, ignore_index=True)
    joined = stage.merge(scores, on=["episode", "sample", "model"], validate="one_to_one")
    checks = {
        "final_prediction": np.array_equal(joined.native_prediction, joined.final_prediction_recomputed),
        "final_rank": np.array_equal(joined.native_rank, joined.final_rank_recomputed),
        "pre_prediction": np.array_equal(joined.encoded_prediction, joined.pre_prediction_recomputed),
        "pre_rank": np.array_equal(joined.encoded_rank, joined.pre_rank_recomputed),
    }
    if not all(checks.values()):
        raise ValueError(f"score reconstruction failed for {target}: {checks}")
    return joined, hashes, checks


def wide_target(frame):
    columns = ["native_correct", "encoded_correct", "native_rank", "encoded_rank",
               "final_z_margin", "pre_z_margin"]
    wide = frame.pivot(index=["episode", "sample", "query", "anchor"],
                       columns="model", values=columns)
    wide.columns = [f"{stage}_{model}" for stage, model in wide.columns]
    return wide.reset_index()


def analyze_stratum(frame, native, foreign):
    result = {"rows": len(frame), "unique_queries": int(frame["query"].nunique()),
              "transitions": {}, "cohorts": {}}
    for model in MODELS:
        renamed = frame.rename(columns={f"encoded_correct_{model}": f"pre_correct_{model}",
                                        f"native_correct_{model}": f"final_correct_{model}"})
        result["transitions"][model] = transition_counts(renamed, model)

    native_ok = frame[f"native_correct_{native}"].astype(bool)
    foreign_ok = frame[f"native_correct_{foreign}"].astype(bool)
    cohort = np.select(
        [native_ok & foreign_ok, native_ok & ~foreign_ok, ~native_ok & foreign_ok],
        ["both_correct", "native_only", "foreign_only"], default="both_wrong",
    )
    work = frame.copy()
    work["cohort"] = cohort
    for name, subset in work.groupby("cohort", sort=True):
        item = {"rows": len(subset), "unique_queries": int(subset["query"].nunique())}
        for model in (native, foreign):
            item[model] = {
                "pre_correct_fraction": float(subset[f"encoded_correct_{model}"].mean()),
                "pre_rank": metric_summary(subset, f"encoded_rank_{model}"),
                "final_rank": metric_summary(subset, f"native_rank_{model}"),
                "pre_z_margin": metric_summary(subset, f"pre_z_margin_{model}"),
                "final_z_margin": metric_summary(subset, f"final_z_margin_{model}"),
                "rank_improvement": summarize(
                    subset[f"encoded_rank_{model}"] - subset[f"native_rank_{model}"]
                ),
            }
        pre_advantage = subset[f"pre_z_margin_{native}"] - subset[f"pre_z_margin_{foreign}"]
        final_advantage = subset[f"final_z_margin_{native}"] - subset[f"final_z_margin_{foreign}"]
        item["native_minus_foreign"] = {
            "pre_z_margin": summarize(pre_advantage),
            "final_z_margin": summarize(final_advantage),
            "native_has_better_pre_rank_fraction": float(
                (subset[f"encoded_rank_{native}"] < subset[f"encoded_rank_{foreign}"]).mean()
            ),
            "equal_pre_rank_fraction": float(
                (subset[f"encoded_rank_{native}"] == subset[f"encoded_rank_{foreign}"]).mean()
            ),
        }
        result["cohorts"][name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--hk-ambiguity", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = {"complete": True, "protocol": "nm_native_foreign_margin_stages_v1",
              "score_normalization": "true-minus-best-rival divided by within-row candidate-score std",
              "targets": {}}
    for target in TARGETS:
        long, hashes, checks = load_target(args.input_root, target)
        wide = wide_target(long)
        strata = {"all": wide}
        if target == "cp_hk" and args.hk_ambiguity:
            ambiguity = pd.read_csv(args.hk_ambiguity)[
                ["episode", "sample", "query", "valid_test_candidates"]
            ]
            ambiguity["episode"] = ambiguity.episode.astype(str)
            wide = wide.merge(ambiguity, on=["episode", "sample", "query"], validate="one_to_one")
            strata = {
                "all": wide,
                "unique_answer": wide[wide.valid_test_candidates.eq(1)],
                "multiple_valid_answers": wide[wide.valid_test_candidates.gt(1)],
            }
            hashes[str(args.hk_ambiguity)] = digest(args.hk_ambiguity)
        report["targets"][target] = {
            "native_model": NATIVE[target],
            "foreign_model": FOREIGN[target],
            "input_hashes": hashes,
            "reconstruction_checks": checks,
            "strata": {
                name: analyze_stratum(part, NATIVE[target], FOREIGN[target])
                for name, part in strata.items()
            },
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
