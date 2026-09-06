"""A single evidence-backed CPU tie reference; never widen general parity tolerances."""
import math


def validate_numerical_audit(audit):
    def finite_tree(value):
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_tree(v) for v in value)
        return math.isfinite(value) if isinstance(value, (int, float)) else True
    if not finite_tree(audit):
        raise ValueError("nonfinite numerical audit")
    if audit.get("all_weights_unchanged") is not True or audit.get("cpu_direct_logits_bit_exact") is not True or audit.get("fitted_or_selected_outputs") is not False:
        raise ValueError("exact unchanged-weight CPU/GPU audit required")
    cpu, official = audit["cpu_metrics"], audit["official_metrics"]
    records = audit["direct_forwards"]
    gpu = [r for r in records if r["device"] == "cuda:0"]
    host = [r for r in records if r["device"] == "cpu"]
    if len(records) != 4 or len(gpu) != 3 or len(host) != 1 or {r["repeat"] for r in gpu} != {0, 1, 2}:
        raise ValueError("all three GPU passes and direct CPU reproduction required")
    if host[0]["metrics"] != cpu or host[0]["max_abs_logit_difference"] != 0 or host[0]["changed_cross_label_pairs"]:
        raise ValueError("CPU direct replay is not exact")
    changed_pairs = set()
    for record in gpu:
        if min(record["positive_queries"], record["negative_queries"]) <= 0 or record["positive_queries"] + record["negative_queries"] != audit["cpu_prediction_audit"]["queries"]:
            raise ValueError("cross-label pair counts differ from prediction inventory")
        if record["changed_decisions"] != 0 or not 0 <= record["max_abs_logit_difference"] <= 1e-4 or not 0 <= record["max_abs_probability_difference"] <= 1e-5:
            raise ValueError("not a small decision-preserving numerical difference")
        if any(abs(record["metrics"][m] - official[m]) > 1e-12 or abs(cpu[m] - official[m]) > 1e-12 for m in ("accuracy", "f1")):
            raise ValueError("official discrete metrics differ")
        if abs(record["metrics"]["roc_auc"] - official["roc_auc"]) > 1e-12:
            raise ValueError("every GPU pass must reproduce official AUC")
        quantum = .5 / (record["positive_queries"] * record["negative_queries"])
        if abs(record["auc_half_pair_quantum"] - quantum) > 1e-12:
            raise ValueError("wrong pair quantum")
        if abs(abs(cpu["roc_auc"] - official["roc_auc"]) - quantum) > 1e-12:
            raise ValueError("difference is not one half-pair tie quantum")
        pairs = record["changed_cross_label_pairs"]
        if len(pairs) != 1 or abs(record["auc_other_minus_cpu"] - (official["roc_auc"] - cpu["roc_auc"])) > 1e-12:
            raise ValueError("AUC difference must be entirely one cross-label pair")
        pair = pairs[0]
        if any(not 0 <= pair[k] < audit["cpu_prediction_audit"]["queries"] for k in ("positive_query", "negative_query")):
            raise ValueError("pair query index outside the input set")
        changed_pairs.add((pair["positive_query"], pair["negative_query"]))
        if pair["cpu_order"] != 0 or abs(pair["other_order"]) != 1 or pair["cpu_scores"][0] != pair["cpu_scores"][1]:
            raise ValueError("CPU tie and GPU tie-breaking required")
        margin = pair["other_scores"][0] - pair["other_scores"][1]
        if not 0 < abs(margin) <= 1e-5 or math.copysign(1, margin) != pair["other_order"]:
            raise ValueError("GPU pair is not a numerically near tie")
        if abs(quantum * pair["other_order"] - record["auc_other_minus_cpu"]) > 1e-12:
            raise ValueError("reported pair order does not explain the AUC delta")
    if len(changed_pairs) != 1:
        raise ValueError("GPU repeats must identify the same pair")
    prediction = audit["cpu_prediction_audit"]
    if prediction["weights_sha256"] != audit["model"]["weights_sha256"] or prediction["episode_fingerprint"] != audit["input_cache"]["episode_fingerprint"]:
        raise ValueError("numerical audit identities disagree")
    return {"model_id": audit["model"]["model_id"], "dataset": audit["target"],
            "weights_sha256": audit["model"]["weights_sha256"], "checkpoint": audit["model"]["checkpoint"],
            "episode_fingerprint": audit["input_cache"]["episode_fingerprint"],
            "official_roc_auc": official["roc_auc"], "audited_cpu_roc_auc": cpu["roc_auc"],
            "cpu_minus_official_roc_auc": cpu["roc_auc"] - official["roc_auc"], "changed_cross_label_pairs": 1}


def numerical_reference(reference, audit):
    change = validate_numerical_audit(audit)
    matched = (reference.model_id == change["model_id"]) & (reference.dataset == change["dataset"])
    if matched.sum() != 1:
        raise ValueError("numerical reference cell must be uniquely resolved")
    row = reference.loc[matched].iloc[0]
    if row.episode_fingerprint != change["episode_fingerprint"] or any(abs(row[m] - audit["official_metrics"][m]) > 1e-12 for m in ("roc_auc", "accuracy", "f1")):
        raise ValueError("numerical audit does not match immutable official reference")
    result = reference.copy(deep=True)
    result.loc[matched, "roc_auc"] = change["audited_cpu_roc_auc"]
    return result, change
