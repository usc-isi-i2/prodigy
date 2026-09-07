"""Apply the frozen public support-centered rule to saved social-model features."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .role_centering import predict
from .run_native import write_json, file_sha256
from scripts.experiments.setup.target_performance_mechanisms.analyze_mixture_predictions import input_labels, evaluate_logits

TARGETS = ("covid_political", "election2020", "facebook_page_reference", "twibot20", "ukr_rus_suspended")


def probe(x, batch, center):
    labels = batch[2]
    query = batch[5].reshape(len(labels), -1)[:, 0].bool()
    tasks = batch[0].task_id_per_sample
    if len(x) != len(labels) or set(tasks.tolist()) != {0, 1, 2, 3}:
        raise ValueError("Expected four complete episodes")
    output = x.new_zeros(labels.shape)
    for task in range(4):
        support, q = (tasks == task) & ~query, (tasks == task) & query
        if not torch.all(labels[support].sum(0) > 0):
            raise ValueError("Support class missing")
        output[q] = predict(x[support], labels[support], x[q], center)
    return output[query]


def score(logits, labels):
    result = evaluate_logits(logits, labels)
    truth, pred = labels["local_y"], logits.argmax(1)
    if labels["use_global"]:
        truth = labels["mapping"][torch.arange(len(truth)), truth]
        probabilities = torch.zeros_like(logits).scatter_add_(1, labels["mapping"], logits.softmax(1))
        pred = probabilities.argmax(1)
    result["accuracy"] = float(accuracy_score(truth, pred))
    result["macro_f1"] = float(f1_score(truth, pred, labels=[0,1], average="macro", zero_division=0))
    result["f1"] = float(f1_score(truth, pred, zero_division=0))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=False)
    cells, receipts = [], []
    try:
        for stream in ("original", "fresh"):
            for target in TARGETS:
                directory = args.replay / stream / target / target
                if not (directory.parent / "DONE").is_file():
                    raise ValueError("Replay incomplete")
                labels = input_labels(directory, target)
                batches = [torch.load(directory / "batches" / f"batch_{i:03d}.pt", map_location="cpu") for i in range(32)]
                for schedule in ("blocked", "interleaved"):
                    for mode in ("native", "joint", "isolated", "ridge_only"):
                        model = f"r4_{schedule}_s0_{mode}"
                        path = directory / (model + "__baseline.pt")
                        records = torch.load(path, map_location="cpu")
                        if len(records) != 32:
                            raise ValueError("Prediction inventory incomplete")
                        predictions = {"full": [], "u1": [], "u1_centered": [], "raw": [], "raw_centered": []}
                        for i, (record, batch) in enumerate(zip(records, batches)):
                            if record["batch"] != i or record["batch_sha256"] != labels["cache"]["batch_sha256"][i]:
                                raise ValueError("Prediction/input receipt differs")
                            predictions["full"].append(record["logits"]["full_model"])
                            for prefix, stage in (("u1", "U1_pre_meta"), ("raw", "raw_center")):
                                x = record["embeddings"][stage]
                                baseline = probe(x, batch, "none")
                                if not torch.allclose(baseline, record["logits"][stage+"/ridge"], atol=1e-5, rtol=0):
                                    raise ValueError("Uncentered social control parity failed")
                                predictions[prefix].append(baseline)
                                predictions[prefix+"_centered"].append(probe(x, batch, "support"))
                        for name, logits in predictions.items():
                            cells.append({"stream": stream, "target": target, "schedule": schedule,
                                "mode": mode, "model": model, "readout": name, **score(torch.cat(logits), labels)})
                        receipts.append({"path": str(path), "sha256": file_sha256(path)})
                print(f"Scored {stream}/{target}", flush=True)
        means = []
        for stream in ("original", "fresh"):
            for panel in ("all5", "fixed4"):
                for schedule in ("blocked", "interleaved"):
                    for mode in ("native", "joint", "isolated", "ridge_only"):
                        for readout in ("full", "u1", "u1_centered", "raw", "raw_centered"):
                            selected = [r for r in cells if r["stream"]==stream and r["schedule"]==schedule and r["mode"]==mode and r["readout"]==readout and (panel=="all5" or r["target"]!="ukr_rus_suspended")]
                            means.append(dict(stream=stream, panel=panel, schedule=schedule, mode=mode, readout=readout,
                                **{m:float(np.mean([r[m] for r in selected])) for m in ("accuracy","macro_f1","f1","roc_auc","nll")}))
        write_json(args.output / "summary.json", {"cells":cells, "means":means, "receipts":receipts,
            "protocol":"Frozen support centering before row-L2; ridge lambda1, scale1, no intercept; no query mean/labels in fit",
            "scope":"Eight rung4 seed0 models, two existing streams; retrospective breadth, not independent seeds"})
    except BaseException as error:
        write_json(args.output / "execution_status.json", {"status":"failed","error":repr(error)})
        raise
    write_json(args.output / "execution_status.json", {"status":"complete"})


if __name__ == "__main__":
    main()
