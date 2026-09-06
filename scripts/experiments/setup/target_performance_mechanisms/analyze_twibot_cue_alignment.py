"""Post-hoc cue agreement of saved TwiBot predictions; no new model fitting.

The reference degree decoder was fitted only on each episode's supports. This
measures decision agreement, not causal mediation or information destruction.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
import torch


def episode_alignment(model, cue, episodes=4):
    if model.shape != cue.shape or model.ndim != 2 or model.shape[1] != 2 or len(model) % episodes:
        raise ValueError("unaligned binary episode logits")
    m = (model[:, 1] - model[:, 0]).reshape(episodes, -1).astype(np.float64)
    c = (cue[:, 1] - cue[:, 0]).reshape(episodes, -1).astype(np.float64)
    results = []
    for a, b in zip(m, c):
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("nonfinite margin")
        if np.std(a) <= 1e-10 or np.std(b) <= 1e-10:
            results.append(None)
            continue
        results.append({"spearman": float(np.corrcoef(rankdata(a), rankdata(b))[0, 1]),
                        "pearson": float(np.corrcoef(a, b)[0, 1]),
                        "decision_agreement": float(np.mean((a > 0) == (b > 0)))})
    return results


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--original-trajectory", type=Path, required=True)
    p.add_argument("--fresh-trajectory", type=Path, required=True)
    p.add_argument("--original-scalars", type=Path, required=True)
    p.add_argument("--fresh-scalars", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    results = []
    for stream in ("original", "fresh"):
        trajectory = getattr(args, f"{stream}_trajectory")
        scalars = getattr(args, f"{stream}_scalars")
        if not (trajectory / "DONE").is_file() or not (scalars / "DONE").is_file():
            raise ValueError("incomplete input experiment")
        target = trajectory / "twibot20"
        cache = json.loads((target / "cache.json").read_text())
        metadata = [json.loads(line) for line in (target / "metrics.jsonl").read_text().splitlines()]
        full = [r for r in metadata if r["decoder"] == "full_model"]
        if len(full) != 36 or len({r["model_id"] for r in full}) != 36:
            raise ValueError("requires all nine four-checkpoint trajectories")
        cues = torch.load(scalars / "twibot20.pt", map_location="cpu", weights_only=False)
        if len(cues) != 32:
            raise ValueError("incomplete degree-cue exports")
        for row in full:
            predictions = torch.load(target / f"{row['model_id']}__baseline.pt", map_location="cpu", weights_only=False)
            if len(predictions) != 32:
                raise ValueError("incomplete trajectory exports")
            for decoder in ("S0_pool/ridge", "U1_pre_meta/ridge", "full_model"):
                comparisons = {key: [] for key in ("center_indegree", "raw_center", "raw_context")}
                for index, (model, cue) in enumerate(zip(predictions, cues)):
                    if model["batch"] != index or cue["batch"] != index or model["batch_sha256"] != cue["batch_sha256"] or model["batch_sha256"] != cache["batch_sha256"][index]:
                        raise ValueError("query order or cached features differ")
                    source = model["logits"][decoder].numpy()
                    if source.shape != (96, 2):
                        raise ValueError("expected four 24-query TwiBot episodes")
                    refs = {"center_indegree": cue["logits"]["scalar_center_indegree"],
                            "raw_center": model["logits"]["raw_center/ridge"],
                            "raw_context": model["logits"]["raw_context/ridge"]}
                    for name, ref in refs.items():
                        comparisons[name].extend(episode_alignment(source, ref.numpy()))
                for name, per_episode in comparisons.items():
                    valid = [r for r in per_episode if r is not None]
                    if not valid:
                        raise ValueError("no informative cue comparisons")
                    results.append({"stream": stream, "source": row["sources"][0], "model_id": row["model_id"],
                        "step": int(Path(row["checkpoint"]).stem.split("_")[-1]), "decoder": decoder, "cue": name,
                        "weights_sha256": row["weights_sha256"], "episode_fingerprint": row["episode_fingerprint"],
                        "valid_episodes": len(valid), "total_episodes": len(per_episode),
                        **{f"mean_within_episode_{k}": float(np.mean([r[k] for r in valid])) for k in valid[0]}})
    (args.output / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    (args.output / "protocol.json").write_text(json.dumps({**{k: str(v) for k, v in vars(args).items()},
        "post_hoc_followup": True, "query_labels_used_for_fitting": False,
        "primary_alignment": "mean within-episode Spearman of local binary margins",
        "constant_model_or_cue_episodes_excluded_and_counted": True,
        "causal_mediation_claim": False, "independent_training_seeds": 1}, indent=2) + "\n")
    (args.output / "DONE").write_text("All 648 cue-alignment cells completed on identical cached inputs.\n")
    print(json.dumps({"rows": len(results), "queries_fitted": 0}))


if __name__ == "__main__":
    main()
