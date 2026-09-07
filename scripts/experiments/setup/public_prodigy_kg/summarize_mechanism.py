"""Score completed paired public episodes without pooling local class columns."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics
from .run_native import file_sha256, write_json


ARMS = ("native", "support_context_removed", "query_context_removed", "keys", "values", "joint")
VIEWS = ("logits", "reference_only_logits", "query_only_logits")


def score_episode(ordinal, result):
    onehot = np.asarray(result["y_true"])
    if onehot.ndim != 2 or not np.isin(onehot, [0, 1]).all() or not (onehot.sum(1) == 1).all():
        raise ValueError("Expected native one-hot query labels")
    labels = onehot.argmax(1)
    if set(result["arms"]) != set(ARMS):
        raise ValueError("Incomplete or unrecognized arm inventory")
    if result["parity"]["max_abs_error"] != 0 or result["parity"]["atol"] != 0 or result["parity"]["rtol"] != 0:
        raise ValueError("Expected nominated exact native parity")
    rows, examples = [], []
    native_prediction = np.asarray(result["arms"]["native"]["logits"]).argmax(1)
    for arm in ARMS:
        for view in VIEWS:
            logits = np.asarray(result["arms"][arm][view])
            if logits.shape != onehot.shape:
                raise ValueError("Query/class inventory differs")
            metrics = episode_metrics(labels, logits)
            prediction = logits.argmax(1)
            native_correct = native_prediction == labels
            correct = prediction == labels
            rows.append({"episode": ordinal, "arm": arm, "view": view, **metrics,
                         "corrected_queries": int((~native_correct & correct).sum()),
                         "corrupted_queries": int((native_correct & ~correct).sum()),
                         "query_max_abs_change": result["arms"][arm]["query_max_abs_change"],
                         "reference_max_abs_change": result["arms"][arm]["reference_max_abs_change"]})
            # Preserve all cases, not just favorable examples. The episode file
            # contains exact graphs; query index is in its native query ordering.
            shifted = logits - logits.max(1, keepdims=True)
            nll = np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(labels)), labels]
            for query, label in enumerate(labels):
                examples.append({"episode": ordinal, "query": query, "arm": arm, "view": view,
                                 "local_label": int(label), "prediction": int(prediction[query]),
                                 "native_prediction": int(native_prediction[query]),
                                 "correct": bool(correct[query]), "nll": float(nll[query])})
    return rows, examples


def summarize_rows(rows, draws=10000):
    by_key = {(row["episode"], row["arm"], row["view"]): row for row in rows}
    episodes = sorted({row["episode"] for row in rows})
    if len(by_key) != len(rows) or len(rows) != len(episodes) * len(ARMS) * len(VIEWS):
        raise ValueError("Duplicate or incomplete episode rows")
    rng = np.random.default_rng(20260907)
    indices = rng.integers(0, len(episodes), (draws, len(episodes)))

    def contrast(left, right, metric):
        values = np.array([by_key[(e, *left)][metric] - by_key[(e, *right)][metric] for e in episodes])
        return {"mean_delta": float(values.mean()),
                "episode_bootstrap_95_interval": np.quantile(values[indices].mean(1), [.025, .975]).tolist(),
                "positive_episodes": int((values > 0).sum()), "negative_episodes": int((values < 0).sum())}

    summary = {"episodes": len(episodes), "metrics": {},
               "uncertainty_scope": "Episode bootstrap conditional on this checkpoint and target; not training-seed or domain-general uncertainty. Recurring entities can make episodes dependent.",
               "interpretation": "Crossed decoder inputs are controlled computational contrasts, not an additive mediation decomposition. NLL improvement is negative; other metric improvement is positive."}
    for metric in ("accuracy", "macro_f1", "ovr_auc", "nll"):
        summary["metrics"][metric] = {
            "means": {f"{arm}/{view}": float(np.mean([by_key[(e, arm, view)][metric] for e in episodes]))
                      for arm in ARMS for view in VIEWS},
            "vs_native": {f"{arm}/{view}": contrast((arm, view), ("native", "logits"), metric)
                          for arm in ARMS if arm != "native" for view in VIEWS},
            "primary_value_minus_key": contrast(("values", "logits"), ("keys", "logits"), metric)}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.run / "execution_status.json").read_text())
    protocol = json.loads((args.run / "protocol.json").read_text())
    index = json.loads((args.run / "paired_episodes/index.json").read_text())
    records = index["records"]
    expected = protocol["paired_mechanism"]["episode_count"]
    if status["status"] != "complete" or len(records) != expected:
        raise ValueError("Run incomplete: do not summarize a favorable partial stream")
    if [r["ordinal"] for r in records] != list(range(expected)):
        raise ValueError("Noncontiguous episode inventory")
    # The report is new and analysis-only; never overwrite prior outputs.
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    with (args.output / "query_audit.csv").open("w", newline="") as stream:
        writer = None
        for record in records:
            path = args.run / "paired_episodes" / record["file"]
            if path.resolve().parent != (args.run / "paired_episodes").resolve() or file_sha256(path) != record["sha256"]:
                raise ValueError("Episode artifact path or hash mismatch")
            artifact = torch.load(path, map_location="cpu")
            scored, examples = score_episode(record["ordinal"], artifact["mechanism"])
            rows.extend(scored)
            if writer is None:
                writer = csv.DictWriter(stream, fieldnames=list(examples[0]))
                writer.writeheader()
            writer.writerows(examples)
    with (args.output / "episode_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    summary["source_protocol_sha256"] = file_sha256(args.run / "protocol.json")
    summary["source_index_sha256"] = file_sha256(args.run / "paired_episodes/index.json")
    write_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
