"""Does omitted, row-aligned account metadata explain the suspension-task floor?

This diagnostic adds a modality the pretrained models never received. It is not
a like-for-like model win or a claim that all metadata precedes suspension.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import torch

from experiments.trainer import TrainerFS, _concat_global_eval_parts
from scripts.social_llm.generate_graph import read_user_data_csv
from .replay import batch_hash
from .probe_cached_inputs import standardized_probe


FIELDS = ("acc_age", "friends_count", "listed_count", "followers_count", "favourites_count",
          "statuses_count", "n_tweets", "n_orig", "n_rt", "n_qtd", "n_replies")


def metadata_features(frame):
    numbers = frame[list(FIELDS)].apply(pd.to_numeric, errors="coerce")
    values = torch.from_numpy(numbers.fillna(0).to_numpy()).float().clamp_min(0).log1p()
    missing = torch.from_numpy(numbers.isna().to_numpy()).float()
    verified = frame["verified"].astype(str).str.strip().str.lower().map({"true": 1., "false": 0., "1": 1., "0": 0., "1.0": 1., "0.0": 0.})
    return torch.cat([values, torch.tensor(verified.fillna(0).to_numpy())[:, None].float()], 1), torch.cat([missing, torch.tensor(verified.isna().to_numpy())[:, None].float()], 1)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data-root", type=Path, default=Path("/dataMeR1/phil/data"))
    p.add_argument("--replay-roots", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    target = "ukr_rus_suspended"
    frame = read_user_data_csv(str(args.data_root / "social_llm_data" / target / "user_data.csv"))
    artifact = torch.load(args.data_root / target / "graphs/retweet_graph.pt", map_location="cpu", weights_only=False)
    if list(map(int, artifact["user_ids"])) != list(range(len(frame))):
        raise ValueError("raw table and graph do not share row-index identities")
    raw_labels = torch.tensor(pd.to_numeric(frame["label_suspended"], errors="coerce").fillna(-1).to_numpy()).long()
    torch.testing.assert_close(raw_labels, artifact["y"].long(), rtol=0, atol=0)
    values, missing = metadata_features(frame)
    if not torch.isfinite(values).all():
        raise ValueError("nonfinite metadata")
    metric = TrainerFS.__new__(TrainerFS)
    metric.parameter = {"task_name": "classification"}
    metric.is_feature_prediction = metric.is_regression = False
    rows = []
    with torch.no_grad():
        for replay in args.replay_roots:
            root = replay / target
            if not (root / "random_init__baseline.pt").exists():
                raise ValueError("incomplete target replay")
            cache = json.loads((root / "cache.json").read_text())
            collected = defaultdict(lambda: {"yt": [], "yp": [], "global": []})
            exports = []
            for i, digest in enumerate(cache["batch_sha256"]):
                batch = torch.load(root / "batches" / f"batch_{i:03d}.pt", weights_only=False)
                if batch_hash(batch) != digest:
                    raise ValueError("cached input changed")
                centers = batch[0].ptr[:-1]
                ids = batch[0].global_node_ids[centers]
                torch.testing.assert_close(artifact["x"][ids], batch[0].x[centers], rtol=0, atol=0)
                label = batch[2].argmax(1)
                global_label = batch[0].task_label_map[batch[0].task_id_per_sample, label]
                torch.testing.assert_close(raw_labels[ids], global_label.long(), rtol=0, atol=0)
                predictors = {"metadata_values": standardized_probe(values[ids], batch),
                              "metadata_missingness": standardized_probe(missing[ids], batch),
                              "metadata_values_and_missingness": standardized_probe(torch.cat([values[ids], missing[ids]], 1), batch),
                              "metadata_plus_raw_bio": standardized_probe(values[ids], batch, batch[0].x[centers])}
                query = batch[5].reshape(-1, 2)[:, 0].bool()
                yt = batch[2][query]
                for name, logits in predictors.items():
                    collected[name]["yt"].append(yt)
                    collected[name]["yp"].append(logits)
                    collected[name]["global"].append(metric._extract_global_classification_eval(batch, yt, logits))
                exports.append({"batch_sha256": digest, "metadata_values": values[ids], "metadata_missingness": missing[ids], "logits": predictors})
            for probe, part in collected.items():
                yt, yp = torch.cat(part["yt"]), torch.cat(part["yp"])
                scores = metric._compute_eval_metrics(yt, yp, global_eval=_concat_global_eval_parts(part["global"]))
                row = {"dataset": target, "probe": probe, "replay_root": str(replay),
                       "episode_fingerprint": cache["episode_fingerprint"], "queries": len(yt), **scores}
                rows.append(row)
                print(json.dumps(row), flush=True)
            torch.save(exports, args.output / f"{replay.name}.pt")
    (args.output / "metrics.json").write_text(json.dumps(rows, indent=2))
    (args.output / "protocol.json").write_text(json.dumps({"columns": [*FIELDS, "verified"],
        "table_rows": len(frame), "all_row_labels_match_graph": True, "all_cached_features_match_graph": True,
        "normalization": "log1p nonnegative numeric values; support-only centering/scaling; ridge alpha 1",
        "metadata_temporally_precedes_suspension": "not verified", "model_input_comparison_is_matched": False}, indent=2))
    (args.output / "DONE").write_text("Completed verified row-aligned metadata diagnostic.\n")


if __name__ == "__main__":
    main()
