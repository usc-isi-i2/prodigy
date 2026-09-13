"""Account for HK support changes using saved logits, without model inference.

The two hybrid score vectors change only the true-class entry or only the 29
rival entries. They are controlled score interventions, not support policies.
For the recorded single-M, no-final-back eval model, query outputs do not depend
on changed support rows; these hybrids correspond to final class-reference swaps.
No claim about which training operation created the reference behavior is made.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(logits, truth):
    rows = np.arange(len(truth))
    other = logits.copy()
    other[rows, truth] = -np.inf
    value = logits[rows, truth]
    return dict(
        correct=logits.argmax(1) == truth,
        prediction=logits.argmax(1),
        rank=1 + (logits > value[:, None]).sum(1),
        margin=value - other.max(1),
    )


def summarize(d):
    out = dict(draws=len(d), cases=int(d.case.nunique()))
    if not len(d):
        return out
    for head in ["original", "full", "true_only", "rivals_only", "mean_cosine", "prototype"]:
        out[head] = dict(correct=int(d[head].sum()), accuracy=float(d[head].mean()),
            cases_correct_at_least_once=int(d.groupby("case")[head].max().sum()))
    out["full_true_only_outcome_disagreements"] = int(d.full.ne(d.true_only).sum())
    out["native_wrong_mean_correct"] = int((~d.full & d.mean_cosine).sum())
    out["native_wrong_prototype_correct"] = int((~d.full & d.prototype).sum())
    out["native_correct_mean_wrong"] = int((d.full & ~d.mean_cosine).sum())
    out["true_score_change"] = dict(mean=float(d.true_delta.mean()), median=float(d.true_delta.median()))
    out["best_rival_score_change"] = dict(mean=float(d.rival_delta.mean()), median=float(d.rival_delta.median()))
    out["maximum_absolute_rival_entry_change"] = float(d.max_rival_change.max())
    out["hybrid_near_ties_within_1e_6"] = int((d.true_only_margin.abs().le(1e-6) | d.rivals_only_margin.abs().le(1e-6)).sum())
    if "selected_mean_cosine" in d:
        better = d.selected_mean_cosine > d.original_mean_cosine + 1e-6
        out["increased_cosine"] = int(better.sum())
        out["increased_cosine_and_lower_native_true_score"] = int((better & d.true_delta.lt(-1e-6)).sum())
        out["increased_cosine_but_broke_original_correct"] = int((better & d.original & ~d.full).sum())
    return out


def analyze(cache_path, original_root, extremes_root):
    cache = np.load(cache_path, allow_pickle=False)
    original_receipt = json.loads((original_root / "receipt.json").read_text())
    extremes_receipt = json.loads((extremes_root / "receipt.json").read_text())
    config = json.loads((original_root / "effective_config.json").read_text())
    assert config["layers"] == "S,U,M" and not config["has_final_back"]
    assert not config["skip_path"] and config["ignore_label_embeddings"]
    assert not config["meta_gnn_pos_only"]
    assert extremes_receipt["prior_cache_sha256"] == original_receipt["cache_sha256"]
    assert str(cache["original_cache_sha256"]) == original_receipt["cache_sha256"]
    assert str(cache["extremes_cache_sha256"]) == extremes_receipt["candidate_bank_sha256"]
    truth_map = {int(k): v for k, v in json.loads(str(cache["case_truth"])).items()}
    frames, logits = {}, {}
    for name, root, filename, keycols, receipt in [
        ("original", original_root, "head_comparison_private.csv", ["case", "condition", "draw"], original_receipt),
        ("extremes", extremes_root, "selection_results_private.csv", ["case", "method", "draw"], extremes_receipt),
    ]:
        path = root / filename
        assert digest(path) == receipt["csv_sha256"]
        d = pd.read_csv(path)
        keyframe = pd.DataFrame(json.loads(str(cache[name + "_keys"])))
        assert keyframe.equals(d[keycols]) and not d.duplicated(keycols).any()
        assert json.loads(str(cache[name + "_heads"])) == ["native", "prototype", "mean_cosine"]
        z = cache[name + "_logits"].astype(np.float64)
        assert z.shape == (len(d), 3, 30) and np.isfinite(z).all()
        truth = d.case.map(truth_map).to_numpy(dtype=int)
        assert len(truth_map) == 200 and np.isin(truth, np.arange(30)).all()
        for i, head in enumerate(["native", "prototype", "mean_cosine"]):
            actual = metrics(z[:, i], truth)
            for field in ["correct", "prediction", "rank"]:
                assert np.array_equal(actual[field], d[head + "_" + field])
            assert np.allclose(actual["margin"], d[head + "_margin"], atol=2e-6, rtol=0)
        frames[name], logits[name] = d, z
    original = frames["original"]
    base_rows = original.index[original.condition.eq("original")]
    base = {int(original.loc[i, "case"]): logits["original"][i, 0] for i in base_rows}
    assert len(base) == 200
    result = dict(protocol="hk_cached_true_rival_score_paths_v1", cache_sha256=digest(cache_path),
        original_cache_sha256=original_receipt["cache_sha256"],
        extremes_cache_sha256=extremes_receipt["candidate_bank_sha256"],
        original_csv_sha256=original_receipt["csv_sha256"],
        extremes_csv_sha256=extremes_receipt["csv_sha256"],
        effective_config_sha256=digest(original_root / "effective_config.json"),
        verified_logit_rows=3800, verified_head_decisions=11400, conditions={})
    for name in ["original", "extremes"]:
        d = frames[name].copy()
        a = np.stack([base[int(case)] for case in d.case])
        b = logits[name][:, 0]
        truth = d.case.map(truth_map).to_numpy(dtype=int)
        index = np.arange(len(d))
        true_only, rivals_only = a.copy(), b.copy()
        true_only[index, truth] = b[index, truth]
        rivals_only[index, truth] = a[index, truth]
        for head, z in [("original", a), ("full", b), ("true_only", true_only), ("rivals_only", rivals_only)]:
            m = metrics(z, truth)
            d[head], d[head + "_margin"] = m["correct"], m["margin"]
        d["mean_cosine"] = d.mean_cosine_correct.astype(bool)
        d["prototype"] = d.prototype_correct.astype(bool)
        d["true_delta"] = b[index, truth] - a[index, truth]
        mask = np.ones(b.shape, dtype=bool)
        mask[index, truth] = False
        ar, br = a[mask].reshape(-1, 29), b[mask].reshape(-1, 29)
        d["rival_delta"] = br.max(1) - ar.max(1)
        d["max_rival_change"] = np.abs(br - ar).max(1)
        assert np.allclose(d.full_margin - d.original_margin, d.true_delta - d.rival_delta, atol=1e-12, rtol=0)
        condition = "condition" if name == "original" else "method"
        for label, frame in d.groupby(condition):
            cohorts = [("all", frame), *list(frame.groupby("cohort"))]
            if "previously_persistent" in frame:
                cohorts.append(("previously_persistent", frame[frame.previously_persistent.astype(bool)]))
            section = {str(cohort): summarize(rows) for cohort, rows in cohorts}
            if "previously_persistent" in frame:
                subset = frame[frame.previously_persistent.astype(bool) & frame.mean_cosine & ~frame.full]
                section["persistent_mean_correct_native_wrong"] = summarize(subset)
            result["conditions"][name + ":" + label] = section
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--original-root", type=Path, required=True)
    p.add_argument("--extremes-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = analyze(args.cache, args.original_root, args.extremes_root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["conditions"]["extremes:nearest"], indent=2))
