"""Analyze matched target queries for a strong and weak transfer source.

The original stream is discovery data. All transforms, clusters, calibration
temperatures, and diagnostic routers are fitted there and frozen before the
fresh stream is evaluated. Routers are deliberately labelled as target-query
supervised diagnostics: they measure exploitable complementarity, not a valid
deployment protocol.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, silhouette_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 17
RAW_VIEWS = ("raw_center", "raw_context", "raw_joint")


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def safe_auc(labels, scores):
    return float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else np.nan


def per_example_ce(logits, labels, temperature=1.0):
    probs = softmax(logits / temperature)
    return -np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))


def fit_temperature(logits, labels):
    result = minimize_scalar(
        lambda log_t: per_example_ce(logits, labels, np.exp(log_t)).mean(),
        bounds=(-4.0, 4.0), method="bounded",
    )
    if not result.success:
        raise RuntimeError("temperature calibration did not converge")
    return float(np.exp(result.x))


def validate_pair(discovery, validation):
    for record in (discovery, validation):
        if record["target"] != discovery["target"]:
            raise ValueError("target differs between streams")
        if set(record["models"]) != set(discovery["models"]):
            raise ValueError("model set differs between streams")
        n = len(record["labels"]["local_y"])
        for model in record["models"].values():
            if len(model["logits"]["full_model"]) != n:
                raise ValueError("logit/query length mismatch")
        for view in RAW_VIEWS:
            if len(record["input"]["embeddings"][view]) != n:
                raise ValueError("input/query length mismatch")
    if discovery["stream"] != "original" or validation["stream"] != "fresh":
        raise ValueError("expected original discovery and fresh validation streams")


def model_ids(record):
    protocol_names = list(record["models"])
    b = next((name for name in protocol_names if "twibot20" in name), protocol_names[0])
    c = next((name for name in protocol_names if "election2020" in name), protocol_names[1])
    if b == c:
        raise ValueError("could not distinguish the two source models")
    return b, c


def metadata_frame(record):
    n = len(record["labels"]["local_y"])
    columns = {}
    for name, value in record["query_metadata"].items():
        array = to_numpy(value)
        if array.ndim == 1 and len(array) == n:
            columns[name] = array
    return pd.DataFrame(columns)


def metadata_features(record):
    frame = metadata_frame(record)
    excluded = {"global_node_id", "query_node_id", "node_id", "episode_id", "target"}
    names = [name for name in frame if name not in excluded and np.issubdtype(frame[name].dtype, np.number)]
    if not names:
        raise ValueError("no numeric non-identity metadata features")
    return frame[names].to_numpy(dtype=np.float64), names


def core_arrays(record, b, c, temperatures):
    y = to_numpy(record["labels"]["local_y"]).astype(int)
    logits_b = to_numpy(record["models"][b]["logits"]["full_model"]).astype(np.float64)
    logits_c = to_numpy(record["models"][c]["logits"]["full_model"]).astype(np.float64)
    pred_b, pred_c = logits_b.argmax(1), logits_c.argmax(1)
    correct_b, correct_c = pred_b == y, pred_c == y
    loss_b_raw, loss_c_raw = per_example_ce(logits_b, y), per_example_ce(logits_c, y)
    loss_b = per_example_ce(logits_b, y, temperatures[b])
    loss_c = per_example_ce(logits_c, y, temperatures[c])
    prob_b = softmax(logits_b / temperatures[b])
    prob_c = softmax(logits_c / temperatures[c])
    return {
        "y": y, "logits_b": logits_b, "logits_c": logits_c,
        "pred_b": pred_b, "pred_c": pred_c,
        "correct_b": correct_b, "correct_c": correct_c,
        "loss_b_raw": loss_b_raw, "loss_c_raw": loss_c_raw,
        "loss_b": loss_b, "loss_c": loss_c,
        "prob_b": prob_b, "prob_c": prob_c,
    }


def stream_summary(arrays):
    y = arrays["y"]
    cb, cc = arrays["correct_b"], arrays["correct_c"]
    result = {
        "n": int(len(y)),
        "accuracy_b": float(cb.mean()), "accuracy_c": float(cc.mean()),
        "auc_b": safe_auc(y, arrays["prob_b"][:, 1]),
        "auc_c": safe_auc(y, arrays["prob_c"][:, 1]),
        "n_b_only": int(np.sum(cb & ~cc)), "n_c_only": int(np.sum(cc & ~cb)),
        "n_both_correct": int(np.sum(cb & cc)), "n_both_wrong": int(np.sum(~cb & ~cc)),
        "disagreement_rate": float(np.mean(cb != cc)),
        "oracle_accuracy": float(np.mean(cb | cc)),
        "raw_nll_b": float(arrays["loss_b_raw"].mean()),
        "raw_nll_c": float(arrays["loss_c_raw"].mean()),
        "calibrated_nll_b": float(arrays["loss_b"].mean()),
        "calibrated_nll_c": float(arrays["loss_c"].mean()),
        "fraction_c_lower_calibrated_loss": float(np.mean(arrays["loss_c"] < arrays["loss_b"])),
    }
    for name, advantage in (
        ("b", arrays["loss_c"] - arrays["loss_b"]),
        ("c", arrays["loss_b"] - arrays["loss_c"]),
    ):
        positive = np.sort(np.clip(advantage, 0, None))[::-1]
        total = positive.sum()
        result[f"{name}_positive_loss_advantage_top10_share"] = (
            float(positive[:max(1, int(np.ceil(.1 * len(positive))))].sum() / total) if total else 0.0
        )
    return result


def occurrence_frame(record, arrays, stream):
    n = len(arrays["y"])
    frame = pd.DataFrame({
        "stream": stream, "occurrence_index": np.arange(n), "label": arrays["y"],
        "pred_b": arrays["pred_b"], "pred_c": arrays["pred_c"],
        "correct_b": arrays["correct_b"], "correct_c": arrays["correct_c"],
        "outcome": np.select(
            [arrays["correct_b"] & arrays["correct_c"], arrays["correct_b"] & ~arrays["correct_c"],
             ~arrays["correct_b"] & arrays["correct_c"]],
            ["both_correct", "b_only", "c_only"], default="both_wrong",
        ),
        "loss_b_raw": arrays["loss_b_raw"], "loss_c_raw": arrays["loss_c_raw"],
        "loss_b_cal": arrays["loss_b"], "loss_c_cal": arrays["loss_c"],
        "c_minus_b_loss": arrays["loss_c"] - arrays["loss_b"],
        "prob1_b_cal": arrays["prob_b"][:, 1], "prob1_c_cal": arrays["prob_c"][:, 1],
    })
    metadata = metadata_frame(record)
    for name in metadata:
        if name not in frame:
            frame[name] = metadata[name].to_numpy()
    if "episode_ids" in record["labels"]:
        frame["episode_id"] = to_numpy(record["labels"]["episode_ids"])
    return frame


def make_cluster_pipeline(x, k):
    n_components = min(32, x.shape[1], x.shape[0] - 1)
    steps = [("scale", StandardScaler())]
    if x.shape[1] > n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("cluster", KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)))
    return Pipeline(steps)


def fit_input_clusters(x):
    candidates = []
    for k in range(2, min(8, len(x) - 1) + 1):
        pipeline = make_cluster_pipeline(x, k)
        labels = pipeline.fit_predict(x)
        transformed = pipeline[:-1].transform(x)
        score = silhouette_score(transformed, labels)
        candidates.append((score, k, pipeline, labels))
    return max(candidates, key=lambda item: item[0])


def cluster_rows(view, stream, labels, arrays, silhouette, selected_k):
    rows = []
    for cluster in sorted(np.unique(labels)):
        mask = labels == cluster
        cb, cc = arrays["correct_b"][mask], arrays["correct_c"][mask]
        rows.append({
            "view": view, "stream": stream, "cluster": int(cluster), "n": int(mask.sum()),
            "selected_k": int(selected_k), "discovery_silhouette": float(silhouette),
            "accuracy_b": float(cb.mean()), "accuracy_c": float(cc.mean()),
            "b_minus_c_accuracy": float(cb.mean() - cc.mean()),
            "b_only_rate": float(np.mean(cb & ~cc)), "c_only_rate": float(np.mean(cc & ~cb)),
            "oracle_accuracy": float(np.mean(cb | cc)),
        })
    return rows


def prediction_features(arrays):
    pb, pc = arrays["prob_b"], arrays["prob_c"]
    return np.column_stack([
        pb[:, 1], pc[:, 1], np.abs(pb[:, 1] - .5), np.abs(pc[:, 1] - .5),
        np.max(pb, axis=1), np.max(pc, axis=1), pb[:, 1] - pc[:, 1],
    ])


def readout_features(record):
    columns = []
    for model in sorted(record["models"]):
        for decoder in sorted(record["models"][model]["logits"]):
            logits = to_numpy(record["models"][model]["logits"][decoder]).astype(np.float64)
            difference = logits[:, 1] - logits[:, 0]
            columns.extend((difference, np.abs(difference)))
    return np.column_stack(columns)


def task_features(record, width):
    mapping = to_numpy(record["labels"]["mapping"]).astype(int)
    result = np.zeros((len(mapping), 2 * width), dtype=np.float64)
    rows = np.arange(len(mapping))
    result[rows, mapping[:, 0]] = 1.0
    result[rows, width + mapping[:, 1]] = 1.0
    return result


def route_features(record, arrays, view):
    if view == "metadata":
        return metadata_features(record)[0]
    if view in RAW_VIEWS:
        return to_numpy(record["input"]["embeddings"][view]).astype(np.float64)
    if view == "prediction_state":
        return prediction_features(arrays)
    if view == "readout_state":
        return readout_features(record)
    raise KeyError(view)


def make_router(x):
    n_components = min(32, x.shape[1], x.shape[0] - 1)
    steps = [("scale", StandardScaler())]
    if x.shape[1] > n_components:
        steps.append(("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)))
    steps.append(("classifier", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                     random_state=RANDOM_STATE)))
    return Pipeline(steps)


def global_ids(frame):
    for name in ("global_node_id", "query_node_id", "node_id"):
        if name in frame:
            return frame[name].to_numpy()
    return np.arange(len(frame))


def routed_metrics(arrays, choose_b):
    predictions = np.where(choose_b, arrays["pred_b"], arrays["pred_c"])
    prob1 = np.where(choose_b, arrays["prob_b"][:, 1], arrays["prob_c"][:, 1])
    selected_prob = np.where(choose_b[:, None], arrays["prob_b"], arrays["prob_c"])
    return {
        "accuracy": float(accuracy_score(arrays["y"], predictions)),
        "auc": safe_auc(arrays["y"], prob1),
        "nll": float(log_loss(arrays["y"], selected_prob, labels=[0, 1])),
    }


def evaluate_router(view, discovery_record, validation_record, discovery_arrays, validation_arrays,
                    discovery_frame, validation_frame):
    if view in ("task_identity", "task_and_readout_state"):
        width = int(max(to_numpy(discovery_record["labels"]["mapping"]).max(),
                        to_numpy(validation_record["labels"]["mapping"]).max()) + 1)
        x_train = task_features(discovery_record, width)
        x_test = task_features(validation_record, width)
        if view == "task_and_readout_state":
            x_train = np.column_stack((x_train, readout_features(discovery_record)))
            x_test = np.column_stack((x_test, readout_features(validation_record)))
    else:
        x_train = route_features(discovery_record, discovery_arrays, view)
        x_test = route_features(validation_record, validation_arrays, view)
    disagree_train = discovery_arrays["correct_b"] != discovery_arrays["correct_c"]
    disagree_test = validation_arrays["correct_b"] != validation_arrays["correct_c"]
    winner_train = discovery_arrays["correct_b"][disagree_train].astype(int)
    winner_test = validation_arrays["correct_b"][disagree_test].astype(int)
    groups = global_ids(discovery_frame)[disagree_train]
    router = make_router(x_train[disagree_train])
    unique_groups = np.unique(groups)
    folds = min(5, len(unique_groups))
    cv_accuracy = np.nan
    if folds >= 2 and len(np.unique(winner_train)) == 2:
        cv = GroupKFold(n_splits=folds)
        cv_accuracy = float(cross_val_score(router, x_train[disagree_train], winner_train,
                                             groups=groups, cv=cv, scoring="accuracy").mean())
    router.fit(x_train[disagree_train], winner_train)
    choose_b = router.predict(x_test).astype(bool)
    metrics = routed_metrics(validation_arrays, choose_b)
    test_ids = global_ids(validation_frame)
    train_ids = set(global_ids(discovery_frame).tolist())
    novel = np.array([item not in train_ids for item in test_ids])
    rows = []
    for subset, mask in (
        ("all_fresh", np.ones(len(x_test), dtype=bool)),
        ("fresh_disagreements", disagree_test),
        ("novel_users", novel),
        ("novel_user_disagreements", novel & disagree_test),
    ):
        if not mask.any():
            continue
        routed = routed_metrics({key: value[mask] for key, value in validation_arrays.items()}, choose_b[mask])
        row = {
            "view": view, "subset": subset, "n": int(mask.sum()),
            "discovery_group_cv_winner_accuracy": cv_accuracy,
            "target_query_supervised_diagnostic": True,
            **{f"routed_{key}": value for key, value in routed.items()},
            "fixed_b_accuracy": float(validation_arrays["correct_b"][mask].mean()),
            "fixed_c_accuracy": float(validation_arrays["correct_c"][mask].mean()),
            "oracle_accuracy": float(np.mean(validation_arrays["correct_b"][mask] |
                                               validation_arrays["correct_c"][mask])),
        }
        if subset.endswith("disagreements"):
            row["winner_prediction_accuracy"] = float(np.mean(choose_b[mask] == winner_test[
                novel[disagree_test] if subset == "novel_user_disagreements" else np.ones(winner_test.shape, bool)
            ]))
        rows.append(row)
    return rows


def example_candidates(frame, per_group=12):
    id_name = next((name for name in ("global_node_id", "query_node_id", "node_id") if name in frame), None)
    selections = []
    definitions = (
        ("b_only_largest_advantage", "b_only", False),
        ("c_only_largest_advantage", "c_only", True),
        ("both_wrong_b_more_confident", "both_wrong", True),
        ("both_wrong_c_more_confident", "both_wrong", False),
    )
    for category, outcome, ascending in definitions:
        subset = frame[frame["outcome"] == outcome].sort_values("c_minus_b_loss", ascending=ascending)
        if id_name:
            subset = subset.drop_duplicates(id_name)
        subset = subset.head(per_group).copy()
        subset.insert(0, "selection", category)
        selections.append(subset)
    return pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()


def decoder_audit(record, b, c, full_arrays, stream):
    """Localize fixed-checkpoint differences across common auxiliary readouts."""
    y = full_arrays["y"]
    outcome = np.select(
        [full_arrays["correct_b"] & full_arrays["correct_c"],
         full_arrays["correct_b"] & ~full_arrays["correct_c"],
         ~full_arrays["correct_b"] & full_arrays["correct_c"]],
        ["both_correct", "b_only", "c_only"], default="both_wrong",
    )
    summary_rows, outcome_rows = [], []
    predictions = {}
    for model in (b, c):
        predictions[model] = {}
        for decoder, tensor in record["models"][model]["logits"].items():
            logits = to_numpy(tensor).astype(np.float64)
            pred = logits.argmax(1)
            predictions[model][decoder] = pred
            summary_rows.append({
                "stream": stream, "model": model, "decoder": decoder, "n": len(y),
                "accuracy": float(np.mean(pred == y)),
                "auc": safe_auc(y, softmax(logits)[:, 1]),
            })
            for group in ("both_correct", "b_only", "c_only", "both_wrong"):
                mask = outcome == group
                if not mask.any():
                    continue
                outcome_rows.append({
                    "stream": stream, "model": model, "decoder": decoder,
                    "full_model_outcome": group, "n": int(mask.sum()),
                    "conditional_accuracy": float(np.mean(pred[mask] == y[mask])),
                })
    transitions = []
    for model in (b, c):
        raw = predictions[model]["raw_joint/ridge"] == y
        encoded = predictions[model]["U1_pre_meta/ridge"] == y
        full = predictions[model]["full_model"] == y
        for group in ("all", "both_correct", "b_only", "c_only", "both_wrong"):
            mask = np.ones(len(y), dtype=bool) if group == "all" else outcome == group
            if not mask.any():
                continue
            transitions.append({
                "stream": stream, "model": model, "full_model_outcome": group,
                "n": int(mask.sum()), "raw_joint_ridge_accuracy": float(raw[mask].mean()),
                "u1_ridge_accuracy": float(encoded[mask].mean()),
                "full_accuracy": float(full[mask].mean()),
                "encoder_repairs_rate": float(np.mean(~raw[mask] & encoded[mask])),
                "encoder_breaks_rate": float(np.mean(raw[mask] & ~encoded[mask])),
                "inference_repairs_rate": float(np.mean(~encoded[mask] & full[mask])),
                "inference_breaks_rate": float(np.mean(encoded[mask] & ~full[mask])),
            })
    raw_identity = {}
    for decoder in record["models"][b]["logits"]:
        if decoder.startswith("raw_"):
            raw_identity[decoder] = bool(np.array_equal(
                to_numpy(record["models"][b]["logits"][decoder]),
                to_numpy(record["models"][c]["logits"][decoder]),
            ))
    return summary_rows, outcome_rows, transitions, raw_identity


def heuristic_audit(record, b, c, arrays, stream):
    """Evaluate fixed, direct-readout, and outcome-label-free expert choices."""
    y = arrays["y"]
    pred_b, pred_c = arrays["pred_b"], arrays["pred_c"]
    disagree = pred_b != pred_c
    logits = {
        model: {decoder: to_numpy(value).astype(np.float64)
                for decoder, value in record["models"][model]["logits"].items()}
        for model in (b, c)
    }
    raw_logits = logits[b]["raw_joint/ridge"]
    raw_pred = raw_logits.argmax(1)
    u1_b = logits[b]["U1_pre_meta/ridge"].argmax(1)
    u1_c = logits[c]["U1_pre_meta/ridge"].argmax(1)
    learned_decoders = [name for name in logits[b] if not name.startswith("raw_")]
    agreement_b = np.mean(np.column_stack([logits[b][name].argmax(1) == pred_b
                                            for name in learned_decoders]), axis=1)
    agreement_c = np.mean(np.column_stack([logits[c][name].argmax(1) == pred_c
                                            for name in learned_decoders]), axis=1)
    choices = {
        "fixed_b": np.ones(len(y), dtype=bool),
        "fixed_c": np.zeros(len(y), dtype=bool),
        "raw_alignment_tie_b": np.where(disagree, pred_b == raw_pred, True),
        "u1_alignment_tie_b": np.where(
            disagree & ((pred_b == u1_b) != (pred_c == u1_c)), pred_b == u1_b, True,
        ),
        "learned_stage_consensus_tie_b": agreement_b >= agreement_c,
        "calibrated_full_confidence": arrays["prob_b"].max(1) >= arrays["prob_c"].max(1),
        "oracle_expert": arrays["correct_b"] | ~arrays["correct_c"],
    }
    rows = []
    fixed_accuracy = float(arrays["correct_b"].mean())
    for name, choose_b in choices.items():
        metrics = routed_metrics(arrays, choose_b)
        rows.append({
            "stream": stream, "method": name, "uses_target_query_outcomes": name == "oracle_expert",
            "uses_discovery_query_labels": name == "calibrated_full_confidence",
            "n": len(y), **metrics, "accuracy_gain_over_fixed_b": metrics["accuracy"] - fixed_accuracy,
            "fraction_choose_b": float(choose_b.mean()),
        })
    raw_probs = softmax(raw_logits)
    rows.append({
        "stream": stream, "method": "raw_joint_ridge_direct", "uses_target_query_outcomes": False,
        "uses_discovery_query_labels": False, "n": len(y),
        "accuracy": float(np.mean(raw_pred == y)), "auc": safe_auc(y, raw_probs[:, 1]),
        "nll": float(log_loss(y, raw_probs, labels=[0, 1])),
        "accuracy_gain_over_fixed_b": float(np.mean(raw_pred == y)) - fixed_accuracy,
        "fraction_choose_b": np.nan,
    })
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    discovery = torch.load(args.input / "original.pt", map_location="cpu", weights_only=False)
    validation = torch.load(args.input / "fresh.pt", map_location="cpu", weights_only=False)
    validate_pair(discovery, validation)
    b, c = model_ids(discovery)
    y_discovery = to_numpy(discovery["labels"]["local_y"]).astype(int)
    temperatures = {
        model: fit_temperature(to_numpy(discovery["models"][model]["logits"]["full_model"]), y_discovery)
        for model in (b, c)
    }
    arrays = {
        "original": core_arrays(discovery, b, c, temperatures),
        "fresh": core_arrays(validation, b, c, temperatures),
    }
    frames = {
        "original": occurrence_frame(discovery, arrays["original"], "original"),
        "fresh": occurrence_frame(validation, arrays["fresh"], "fresh"),
    }
    occurrences = pd.concat(frames.values(), ignore_index=True)
    occurrences.to_csv(args.output / "occurrences.csv", index=False)

    cluster_records = []
    for view in ("metadata",) + RAW_VIEWS:
        x_discovery = metadata_features(discovery)[0] if view == "metadata" else to_numpy(
            discovery["input"]["embeddings"][view])
        x_validation = metadata_features(validation)[0] if view == "metadata" else to_numpy(
            validation["input"]["embeddings"][view])
        silhouette, k, pipeline, labels_discovery = fit_input_clusters(x_discovery)
        labels_validation = pipeline.predict(x_validation)
        cluster_records.extend(cluster_rows(view, "original", labels_discovery, arrays["original"], silhouette, k))
        cluster_records.extend(cluster_rows(view, "fresh", labels_validation, arrays["fresh"], silhouette, k))
    pd.DataFrame(cluster_records).to_csv(args.output / "cluster_summary.csv", index=False)

    router_records = []
    for view in ("metadata",) + RAW_VIEWS + (
        "prediction_state", "task_identity", "readout_state", "task_and_readout_state",
    ):
        router_records.extend(evaluate_router(view, discovery, validation, arrays["original"], arrays["fresh"],
                                              frames["original"], frames["fresh"]))
    pd.DataFrame(router_records).to_csv(args.output / "router_results.csv", index=False)
    example_candidates(frames["original"]).to_csv(args.output / "example_candidates.csv", index=False)

    decoder_rows, outcome_decoder_rows, transition_rows, identities = [], [], [], {}
    for stream, record in (("original", discovery), ("fresh", validation)):
        result = decoder_audit(record, b, c, arrays[stream], stream)
        decoder_rows.extend(result[0])
        outcome_decoder_rows.extend(result[1])
        transition_rows.extend(result[2])
        identities[stream] = result[3]
    pd.DataFrame(decoder_rows).to_csv(args.output / "decoder_summary.csv", index=False)
    pd.DataFrame(outcome_decoder_rows).to_csv(args.output / "outcome_decoder_summary.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(args.output / "transition_summary.csv", index=False)
    heuristic_rows = []
    for stream, record in (("original", discovery), ("fresh", validation)):
        heuristic_rows.extend(heuristic_audit(record, b, c, arrays[stream], stream))
    pd.DataFrame(heuristic_rows).to_csv(args.output / "heuristic_results.csv", index=False)

    summary = {
        "target": discovery["target"], "model_b": b, "model_c": c,
        "discovery_stream": "original", "validation_stream": "fresh",
        "temperatures_fitted_on_original": temperatures,
        "streams": {name: stream_summary(value) for name, value in arrays.items()},
        "raw_decoder_logits_bit_identical_between_models": identities,
        "interpretation_constraints": [
            "raw cross-model losses are not comparable without calibration",
            "fresh episodes are a held-out episode stream, not a training seed or held-out domain",
            "routers use target query outcomes and are diagnostic upper bounds, not deployable methods",
            "input clusters are selected by discovery-stream silhouette without outcome labels",
        ],
    }
    write_json(args.output / "summary.json", summary)
    write_json(args.output / "DONE.json", {
        "complete": True, "artifacts": ["summary.json", "occurrences.csv", "cluster_summary.csv",
                                          "router_results.csv", "example_candidates.csv", "decoder_summary.csv",
                                          "outcome_decoder_summary.csv", "transition_summary.csv",
                                          "heuristic_results.csv"],
    })
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
