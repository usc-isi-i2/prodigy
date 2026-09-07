"""Account for saved endpoint AUC changes using fixed query-pair strata.

Observational accounting, not causal mediation or a target adaptation rule.
No model forwards. Equal episode weights; half-credit for score ties.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

GROUPS = ("equal_degree", "cue_correct", "cue_wrong", "cue_tied")
DECODERS = ("S0_pool/prototype", "S0_pool/ridge", "U1_pre_meta/prototype",
            "U1_pre_meta/ridge", "full_model")


def pair_accounting(y, degree, cue, early, late):
    arrays = [np.asarray(x) for x in (y, degree, cue, early, late)]
    y, degree, cue, early, late = arrays
    if any(x.ndim != 1 or x.shape != y.shape or not np.isfinite(x).all() for x in arrays):
        raise ValueError("Aligned finite vectors required")
    if set(y) != {0, 1} or np.any(degree < 0) or np.any(degree != np.floor(degree)):
        raise ValueError("Binary labels and integer incoming degrees required")
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    def diff(x):
        return x[pos, None] - x[neg][None, :]
    equal, cd = diff(degree) == 0, diff(cue)
    masks = (equal, ~equal & (cd > 0), ~equal & (cd < 0), ~equal & (cd == 0))
    if not np.all(np.sum(masks, axis=0) == 1):
        raise ValueError("Pair partition incomplete")
    a, b = [((diff(x) > 0) + .5*(diff(x) == 0)) for x in (early, late)]
    rows = []
    for name, mask in zip(GROUPS, masks):
        n = int(mask.sum())
        rows.append(dict(group=name, pairs=n, mass=n/a.size,
            early_contribution=float(a[mask].sum()/a.size),
            late_contribution=float(b[mask].sum()/a.size),
            change_contribution=float((b-a)[mask].sum()/a.size),
            lost_contribution=float(np.maximum(a-b, 0)[mask].sum()/a.size),
            gained_contribution=float(np.maximum(b-a, 0)[mask].sum()/a.size)))
    if abs(sum(r["change_contribution"] for r in rows) - float((b-a).mean())) > 1e-12:
        raise ValueError("AUC accounting failed")
    return rows, float(a.mean()), float(b.mean())


def main():
    import torch
    from sklearn.metrics import roc_auc_score
    from .analyze_mixture_predictions import input_labels
    from .probe_cached_inputs import standardized_probe
    from .replay import batch_hash
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists() or torch.cuda.is_available():
        raise ValueError("New output and hidden GPUs required")
    torch.set_num_threads(2)
    done = json.loads((args.root/"DONE.json").read_text())
    if done.get("models") != 36 or done.get("all_cached_inputs_match") is not True:
        raise ValueError("Completed corrected replay required")
    rows, receipts = [], []
    for stream in ("original", "fresh"):
        directory = args.root/stream/"twibot20"
        labels = input_labels(directory, "twibot20")
        degrees, cues = [], []
        for bi, fingerprint in enumerate(labels["cache"]["batch_sha256"]):
            batch = torch.load(directory/"batches"/f"batch_{bi:03d}.pt", map_location="cpu", weights_only=False)
            if batch_hash(batch) != fingerprint:
                raise ValueError("Input changed")
            graph = batch[0]
            degree = torch.bincount(graph.edge_index[1], minlength=len(graph.x))[graph.ptr[:-1]]
            query = batch[5].reshape(-1, 2)[:, 0].bool()
            degrees.append(degree[query])
            cue = standardized_probe(degree.float().log1p()[:, None], batch)
            cues.append(cue[:, 1] - cue[:, 0])
        degree, cue = torch.cat(degrees).numpy(), torch.cat(cues).numpy()
        y, episodes = labels["local_y"].numpy(), labels["episode_ids"].numpy()
        metadata = [json.loads(x) for x in (directory/"metrics.jsonl").read_text().splitlines()]
        early = [r for r in metadata if r["decoder"] == "full_model" and r["model_id"].endswith("_step100")]
        if len(early) != 9 or len({r["sources"][0] for r in early}) != 9:
            raise ValueError("All nine sources required")
        for record in early:
            exports = {}
            for step in (100, 2500):
                mid = record["model_id"].rsplit("_step", 1)[0] + f"_step{step}"
                path = directory/(mid+"__baseline.pt")
                saved = torch.load(path, map_location="cpu", weights_only=False)
                if len(saved) != 32:
                    raise ValueError("Incomplete prediction batches")
                for bi, item in enumerate(saved):
                    if item["batch"] != bi or item["batch_sha256"] != labels["cache"]["batch_sha256"][bi]:
                        raise ValueError("Prediction input/order mismatch")
                exports[step] = {}
                for decoder in DECODERS:
                    logits = torch.cat([item["logits"][decoder] for item in saved])
                    if logits.shape != (len(y), 2) or not torch.isfinite(logits).all():
                        raise ValueError("Invalid prediction tensor")
                    prob = logits.softmax(1)
                    ref = [r for r in metadata if r["model_id"] == mid and r["decoder"] == decoder]
                    if len(ref) != 1:
                        raise ValueError("Missing unique metric reference")
                    mapping = labels["mapping"]
                    semantic_y = mapping[torch.arange(len(y)), labels["local_y"]].numpy()
                    semantic_score = prob[torch.arange(len(y)), (mapping == 1).long().argmax(1)].numpy()
                    error = abs(roc_auc_score(semantic_y, semantic_score)-ref[0]["roc_auc"])
                    if error > 1e-6:
                        raise ValueError("Saved pooled AUC not reproduced")
                    exports[step][decoder] = prob[:, 1].numpy()
                receipts.append(dict(stream=stream, model_id=mid,
                    prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            for decoder in DECODERS:
                parts, a_values, b_values = [], [], []
                for ep in np.unique(episodes):
                    mask = episodes == ep
                    part, a, b = pair_accounting(y[mask], degree[mask], cue[mask],
                        exports[100][decoder][mask], exports[2500][decoder][mask])
                    for step, got in ((100, a), (2500, b)):
                        expected = roc_auc_score(y[mask], exports[step][decoder][mask])
                        if abs(got-expected) > 1e-12:
                            raise ValueError("Within-episode AUC not reproduced")
                    parts.extend(part); a_values.append(a); b_values.append(b)
                for group in GROUPS:
                    selected = [r for r in parts if r["group"] == group]
                    row = dict(stream=stream, source=record["sources"][0], decoder=decoder,
                        group=group, episodes=len(selected), pairs=sum(r["pairs"] for r in selected),
                        episodes_present=sum(r["pairs"] > 0 for r in selected),
                        early_auc=float(np.mean(a_values)), late_auc=float(np.mean(b_values)))
                    for key in ("mass", "early_contribution", "late_contribution", "change_contribution",
                                "lost_contribution", "gained_contribution"):
                        row[key] = float(np.mean([r[key] for r in selected]))
                    row["early_group_ranking"] = row["early_contribution"]/row["mass"] if row["mass"] else None
                    row["late_group_ranking"] = row["late_contribution"]/row["mass"] if row["mass"] else None
                    rows.append(row)
    if len(rows) != 360 or len(receipts) != 36:
        raise ValueError("Incomplete endpoint/source/decoder/stream grid")
    args.output.mkdir(parents=True)
    result = dict(revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        new_model_forwards=0, new_training=False, weighting="equal episode; half-credit ties",
        rows=rows, receipts=receipts)
    (args.output/"results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"rows":len(rows), "receipts":len(receipts), "output":str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
