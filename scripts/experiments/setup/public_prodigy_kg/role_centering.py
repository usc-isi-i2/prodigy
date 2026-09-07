"""Fixed exploratory role-centering diagnostic on completed saved geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .run_native import file_sha256, write_json
from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics


def predict(support, support_labels, query, center):
    if center not in ("none", "support", "separate"):
        raise ValueError("Unknown centering rule")
    if center != "none":
        mean = support.mean(0, keepdim=True)
        query_mean = query.mean(0, keepdim=True) if center == "separate" else mean
        support, query = support - mean, query - query_mean
    support, query = F.normalize(support, dim=1), F.normalize(query, dim=1)
    weights = torch.linalg.solve(support @ support.T + torch.eye(len(support), device=support.device,
                                                               dtype=support.dtype), support_labels.to(support))
    return query @ support.T @ weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for directory in (args.run, args.source):
        if json.loads((directory / "execution_status.json").read_text())["status"] != "complete":
            raise ValueError("Source runs must be complete")
    protocol = json.loads((args.run / "protocol.json").read_text())
    index_path = args.source / "paired_episodes/index.json"
    if file_sha256(index_path) != protocol["source_index_sha256"]:
        raise ValueError("Native episode index differs")
    records = json.loads(index_path.read_text())["records"]
    if len(records) != 500 or [r["ordinal"] for r in records] != list(range(500)):
        raise ValueError("Expected complete nominated 500-episode stream")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {
        "stage_run": str(args.run.resolve()), "source_run": str(args.source.resolve()),
        "stage_protocol_sha256": file_sha256(args.run / "protocol.json"),
        "centering": ["none", "support", "separate"], "ridge_lambda": 1,
        "normalization": "row L2 after centering", "scale": 1, "intercept": False,
        "status": "exploratory; hypotheses formed after inspecting this stream",
        "transduction": "separate uses unlabeled query mean; balanced episodes limit prior-shift generalization"})
    rows, receipts = [], []
    try:
        for record in records:
            source = args.source / "paired_episodes" / record["file"]
            if source.resolve().parent != (args.source / "paired_episodes").resolve() or file_sha256(source) != record["sha256"]:
                raise ValueError("Native artifact receipt failed")
            saved_path = args.run / record["file"]
            saved = torch.load(saved_path, map_location="cpu")
            if saved["source_sha256"] != record["sha256"]:
                raise ValueError("Stage artifact source differs")
            native = torch.load(source, map_location="cpu")["native_capture"]
            result = saved["result"]
            if len(result["tasks"]) != 1:
                raise ValueError("Expected one task per episode")
            task = result["tasks"][0]
            support, query = task["support_rows"], task["query_rows"]
            labels = native["input"][2]
            # Use actual saved support labels, not an assumed class-block layout.
            y = labels[query].argmax(1).numpy()
            if not torch.equal(labels[query], result["native"]["y_true"]):
                raise ValueError("Query order differs")
            scores = {"native": episode_metrics(y, result["native"]["logits"].numpy())}
            predictions = {}
            for stage in ("pre", "post"):
                x = result["geometry"][stage]
                for center in ("none", "support", "separate"):
                    name = stage + "/" + center
                    z = predict(x[support], labels[support], x[query], center)
                    if center == "none":
                        expected = result["ridge_logits" if stage == "pre" else "post_ridge_logits"]
                        if not torch.allclose(z, expected, atol=1e-5, rtol=0):
                            raise ValueError("Uncentered control fails saved readout parity")
                    scores[name] = episode_metrics(y, z.numpy())
                    predictions[name] = z
            rows.append(scores)
            torch.save({"source_sha256": record["sha256"], "query_labels": labels[query],
                        "logits": predictions, "metrics": scores}, args.output / record["file"])
            receipts.append({"file": record["file"], "stage_sha256": file_sha256(saved_path)})
            if (record["ordinal"] + 1) % 50 == 0:
                print(f"Scored {record['ordinal'] + 1}/500", flush=True)
        means = {arm: {metric: float(np.mean([row[arm][metric] for row in rows]))
                       for metric in rows[0][arm]} for arm in rows[0]}
        comparisons = {}
        for arm in means:
            comparisons[arm] = {}
            for base in ("native", "pre/none", "post/none"):
                differences = np.array([row[arm]["accuracy"] - row[base]["accuracy"] for row in rows])
                comparisons[arm][base] = {"accuracy_delta": float(differences.mean()),
                    "wins": int((differences > 0).sum()), "losses": int((differences < 0).sum())}
        write_json(args.output / "summary.json", {"episodes": len(rows), "means": means,
                   "comparisons": comparisons, "receipts": receipts})
    except BaseException as error:
        write_json(args.output / "execution_status.json", {"status": "failed", "error": repr(error)})
        raise
    write_json(args.output / "execution_status.json", {"status": "complete"})


if __name__ == "__main__":
    main()
