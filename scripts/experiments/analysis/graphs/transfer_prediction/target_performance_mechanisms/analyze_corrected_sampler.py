"""Historical versus corrected sampler trajectories on exactly matched test inputs."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experiments.setup.final_core.core_plan import SOURCES
from .analyze_trajectories import STEPS, PANEL, DECODERS, endpoint_changes, read_jsonl
from .analyze_member_initial_reference import validate_inputs

METRICS = ("roc_auc", "accuracy", "f1", "nll")
PAIR_KEY = ["stream", "dataset", "source", "step", "decoder"]


def validate_inventory(receipt, manifest):
    if any(receipt.get(k) != v for k, v in {"rows": 6120, "models": 36, "underlying_training_runs": 9,
            "training_seeds": [0], "steps": list(STEPS)}.items()) or not all(receipt.get(k) is True for k in
            ("training_sidecars_and_configs_verified", "all_cached_inputs_match")):
        raise ValueError("complete corrected replay receipt required")
    if receipt.get("new_training") is not False or receipt.get("matched_training_inputs_claim") is not False:
        raise ValueError("unmatched existing-run reuse must remain explicit")
    if manifest.get("training_provenance", {}).get("commit") != "edd1649e8446416213d2e48893d641b3b697eca1":
        raise ValueError("unresolved corrected training revision")
    if manifest.get("new_training") is not False or manifest.get("changes_retention_and_roles_jointly") is not True or manifest.get("matched_training_inputs_claim") is not False:
        raise ValueError("incorrect intervention interpretation")
    records = pd.DataFrame(manifest["models"])
    if len(records) != 36 or records.model_id.duplicated().any() or records.checkpoint.duplicated().any() or records.weights_sha256.nunique() != 36:
        raise ValueError("36 unique corrected checkpoints required")
    if set(map(tuple, records[["source", "step"]].to_numpy())) != set(product(SOURCES, STEPS)):
        raise ValueError("nine-source four-step inventory incomplete")
    for row in records.itertuples():
        if row.training_seed != 0 or row.sources != [row.source] or row.completed_steps_verified != row.step or row.optimizer_steps_verified != row.step:
            raise ValueError("checkpoint source/seed/optimizer metadata differs")
        config = row.effective_config
        if config["neighbor_sampling_source_subset"] != row.source or any(config[k] != v for k, v in
                {"seed": 0, "batch_size": 4, "epochs": 1, "dataset_len_cap": 2500, "task_name": "neighbor_matching"}.items()):
            raise ValueError("effective training source or budget differs")
        if not row.checkpoint.endswith(f"state_dict_{row.step}.ckpt"):
            raise ValueError("checkpoint filename step differs")
    return records


def validate_cells(cells, inventory, historical):
    key = ["stream", "dataset", "model_id", "decoder"]
    expected = set(product(("original", "fresh"), PANEL, inventory.model_id, DECODERS))
    if len(cells) != 6120 or cells.duplicated(key).any() or set(map(tuple, cells[key].to_numpy())) != expected:
        raise ValueError("complete corrected target/step/decoder/stream grid required")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128}:
        raise ValueError("unexpected corrected replay protocol")
    for m in METRICS:
        if not np.isfinite(cells[m]).all() or (cells[m] < 0).any() or (m != "nll" and (cells[m] > 1).any()):
            raise ValueError("invalid corrected metric")
    cols = ["model_id", "source", "step", "checkpoint", "weights_sha256", "sources"]
    joined = cells.merge(inventory[cols], on="model_id", suffixes=("", "_inventory"), validate="many_to_one")
    for field in ("checkpoint", "weights_sha256"):
        if not (joined[field] == joined[f"{field}_inventory"]).all():
            raise ValueError("corrected replay checkpoint identity differs")
    if any(a != b for a, b in zip(joined.sources, joined.sources_inventory)):
        raise ValueError("corrected replay source composition differs")
    joined = joined.drop(columns=["checkpoint_inventory", "weights_sha256_inventory", "sources_inventory"])
    expected_old = set(product(("original", "fresh"), PANEL, SOURCES, STEPS, DECODERS))
    if len(historical) != 6120 or historical.duplicated(PAIR_KEY).any() or set(map(tuple, historical[PAIR_KEY].to_numpy())) != expected_old:
        raise ValueError("complete historical trajectory reference required")
    paired = joined.merge(historical, on=PAIR_KEY, suffixes=("_corrected", "_historical"), validate="one_to_one")
    if len(paired) != 6120 or not (paired.episode_fingerprint_corrected == paired.episode_fingerprint_historical).all() or not (paired.queries_corrected == paired.queries_historical).all():
        raise ValueError("historical and corrected query identities differ")
    for metric in METRICS:
        paired[f"corrected_minus_historical_{metric}"] = paired[f"{metric}_corrected"] - paired[f"{metric}_historical"]
    raw = paired[paired.decoder.str.startswith("raw_")]
    if (raw[[f"corrected_minus_historical_{m}" for m in METRICS]].abs() > 1e-5).any().any():
        raise ValueError("raw input reference changed")
    for (stream, target), group in joined.groupby(["stream", "dataset"]):
        if group.episode_fingerprint.nunique() != 1 or group.queries.nunique() != 1:
            raise ValueError("test inputs vary across corrected checkpoints")
    cross = joined[joined.stream == "original"].merge(joined[joined.stream == "fresh"], on=["dataset", "model_id", "decoder"], suffixes=("_original", "_fresh"))
    if len(cross) != 3060 or not (cross.weights_sha256_original == cross.weights_sha256_fresh).all() or (cross.episode_fingerprint_original == cross.episode_fingerprint_fresh).any():
        raise ValueError("cross-stream weights/input contract differs")
    return joined, paired


def summarize(cells, paired, historical):
    now, old = endpoint_changes(cells), endpoint_changes(historical)
    changes = now.merge(old, on=["stream", "dataset", "source", "decoder"], suffixes=("_corrected", "_historical"), validate="one_to_one")
    if len(changes) != 1530:
        raise ValueError("complete endpoint changes required")
    changes["difference_in_100_to_2500_changes"] = changes.change_100_to_2500_corrected - changes.change_100_to_2500_historical
    summaries, ranking = [], []
    for panel in ("all", "foreign"):
        subset = changes if panel == "all" else changes[changes.source != changes.dataset]
        expected = 9 if panel == "all" else 8
        for key, group in subset.groupby(["stream", "dataset", "decoder"]):
            if len(group) != expected:
                raise ValueError("source panel incomplete")
            row = dict(zip(("stream", "dataset", "decoder"), key), panel=panel, sources=len(group))
            for column in ("change_100_to_2500_corrected", "change_100_to_2500_historical", "difference_in_100_to_2500_changes"):
                row[f"mean_{column}"] = float(group[column].mean())
                row[f"positive_{column}"] = int(group[column].gt(0).sum())
                row[f"negative_{column}"] = int(group[column].lt(0).sum())
            summaries.append(row)
        terminal = paired[paired.step == 2500]
        if panel == "foreign":
            terminal = terminal[terminal.source != terminal.dataset]
        for key, group in terminal.groupby(["stream", "dataset", "decoder"]):
            if len(group) != expected:
                raise ValueError("terminal source panel incomplete")
            x, y = group.roc_auc_historical.rank(), group.roc_auc_corrected.rank()
            rho = float(x.corr(y)) if x.nunique() > 1 and y.nunique() > 1 else None
            ranking.append(dict(zip(("stream", "dataset", "decoder"), key), panel=panel, sources=len(group),
                descriptive_source_rank_correlation=rho, mean_historical_auc=float(group.roc_auc_historical.mean()),
                mean_corrected_auc=float(group.roc_auc_corrected.mean()),
                mean_corrected_minus_historical_auc=float(group.corrected_minus_historical_roc_auc.mean()),
                corrected_better=int(group.corrected_minus_historical_roc_auc.gt(0).sum()),
                corrected_worse=int(group.corrected_minus_historical_roc_auc.lt(0).sum())))
    return changes, pd.DataFrame(summaries), pd.DataFrame(ranking)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    root = args.data / "corrected_sampler_verified"
    receipt = json.loads((root / "DONE.json").read_text())
    inventory = validate_inventory(receipt, json.loads((root / "checkpoint_inventory.json").read_text()))
    validate_inputs(json.loads((root / "input_validation.json").read_text()),
                    json.loads((args.data / "trajectory_input_validation.json").read_text()))
    streams = [read_jsonl(args.data / f"corrected_sampler_{stream}").assign(stream=stream) for stream in ("original", "fresh")]
    historical = pd.read_csv(args.data / "trajectory_cells.csv")
    cells, paired = validate_cells(pd.concat(streams, ignore_index=True), inventory, historical)
    changes, summary, ranking = summarize(cells, paired, historical)
    cells.assign(sources=cells.sources.map(json.dumps)).to_csv(args.data / "corrected_sampler_cells.csv", index=False)
    paired.to_csv(args.data / "corrected_sampler_vs_historical.csv", index=False)
    changes.to_csv(args.data / "corrected_sampler_endpoint_changes.csv", index=False)
    summary.to_csv(args.data / "corrected_sampler_endpoint_summary.csv", index=False)
    ranking.to_csv(args.data / "corrected_sampler_terminal_ranking.csv", index=False)
    (args.data / "corrected_sampler_validation.json").write_text(json.dumps({"rows": 6120, "source_runs": 9,
        "training_seeds": [0], "steps": list(STEPS), "all_cached_inputs_match": True,
        "joint_retention_role_and_rng_recipe_change": True, "matched_training_inputs": False,
        "untouched_targets": False, "target_checkpoint_selection": False,
        "training_receipt": receipt, "checkpoint_inventory_sha256": hashlib.sha256((root / "checkpoint_inventory.json").read_bytes()).hexdigest()}, indent=2) + "\n")
    chosen = summary[(summary.panel == "foreign") & summary.decoder.isin(["S0_conv_center/ridge", "S0_pool/ridge", "U1_pre_meta/ridge", "full_model"])]
    print(chosen.to_string(index=False))


if __name__ == "__main__":
    main()
