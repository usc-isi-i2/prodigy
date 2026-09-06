"""Validate fixed-input checkpoint trajectories; never select a best checkpoint."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


STEPS = (100, 300, 900, 2500)
STAGES = ("raw_center", "raw_context", "raw_joint", "S0_conv_center",
          "S0_pool", "U1_pre_meta", "M2_post_meta", "final_input")
DECODERS = {"full_model"} | {f"{s}/{d}" for s in STAGES for d in ("prototype", "ridge")}
PANEL = ("covid_political", "facebook_page_reference", "election2020",
         "twibot20", "ukr_rus_suspended")
KEY = ["dataset", "model_id", "decoder"]


def read_jsonl(root):
    return pd.DataFrame([json.loads(line) for p in sorted(root.glob("*.jsonl"))
                         for line in p.read_text().splitlines()])


def validate(cells, manifest, reference, stream):
    expected = pd.MultiIndex.from_product([PANEL, manifest.model_id, sorted(DECODERS)], names=KEY)
    if cells.empty or cells.duplicated(KEY).any() or set(pd.MultiIndex.from_frame(cells[KEY])) != set(expected):
        raise ValueError("incomplete or duplicate checkpoint-by-target-by-decoder grid")
    if set(cells.variant) != {"baseline"} or set(cells.episodes) != {128}:
        raise ValueError("unexpected intervention or episode count")
    if not np.isfinite(cells[["roc_auc", "accuracy", "f1", "nll"]].to_numpy()).all():
        raise ValueError("nonfinite metrics")
    joined = cells.merge(manifest, on="model_id", suffixes=("", "_manifest"), validate="many_to_one")
    if not (joined.checkpoint == joined.checkpoint_manifest).all():
        raise ValueError("checkpoint path does not match manifest")
    if not all(s == [t] for s, t in zip(joined.sources, joined.sources_manifest)):
        raise ValueError("source does not match manifest")
    if (joined.groupby("model_id").weights_sha256.nunique() != 1).any():
        raise ValueError("weights changed across targets or decoders")
    for target, part in joined.groupby("dataset"):
        ref = reference[reference.dataset == target]
        if part.episode_fingerprint.nunique() != 1 or set(part.episode_fingerprint) != set(ref.episode_fingerprint):
            raise ValueError("trajectory episode fingerprint differs from established reference")
        if part.queries.nunique() != 1 or set(part.queries) != set(ref.queries):
            raise ValueError("query count differs from reference")
        for decoder, raw in part[part.decoder.str.startswith("raw_")].groupby("decoder"):
            if raw.roc_auc.max() - raw.roc_auc.min() > 1e-8:
                raise ValueError("raw probe depends on checkpoint")
    terminal = joined[joined.step == 2500].merge(
        reference[KEY + ["weights_sha256", "roc_auc", "accuracy", "f1"]],
        on=KEY, suffixes=("", "_reference"), validate="one_to_one")
    if len(terminal) != len(PANEL) * len(DECODERS) * 9:
        raise ValueError("missing terminal reference cells")
    if not (terminal.weights_sha256 == terminal.weights_sha256_reference).all():
        raise ValueError("terminal checkpoint weights differ from previous replay")
    for metric, tolerance in (("roc_auc", 1e-5), ("accuracy", 1e-6), ("f1", 1e-6)):
        if ((terminal[metric] - terminal[f"{metric}_reference"]).abs() > tolerance).any():
            raise ValueError(f"terminal {metric} differs from previous replay")
    if stream == "original":
        full = terminal[terminal.decoder == "full_model"]
        if full.official_metric_max_abs_error.isna().any() or (full.official_metric_max_abs_error > 1e-5).any():
            raise ValueError("missing historical endpoint parity")
    return joined.drop(columns=["checkpoint_manifest", "sources_manifest"]).assign(stream=stream)


def endpoint_changes(cells):
    index = ["stream", "dataset", "source", "decoder"]
    wide = cells.pivot(index=index, columns="step", values="roc_auc")
    if set(wide.columns) != set(STEPS) or wide.isna().any().any():
        raise ValueError("incomplete four-step trajectory")
    wide["change_100_to_2500"] = wide[2500] - wide[100]
    # Report shape without choosing a checkpoint based on target outcomes.
    increments = np.diff(wide[list(STEPS)].to_numpy(), axis=1)
    wide["all_three_increments_positive"] = (increments > 0).all(axis=1)
    wide["all_three_increments_negative"] = (increments < 0).all(axis=1)
    return wide.reset_index()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[6]
    manifest = pd.read_csv(repo / "scripts/experiments/setup/target_performance_mechanisms/data/trajectory_model_list.tsv", sep="\t")
    manifest["step"] = manifest.checkpoint.str.extract(r"state_dict_(\d+)\.ckpt$").astype(int)
    manifest["source"] = manifest.sources
    if len(manifest) != 36 or manifest.model_id.duplicated().any() or any(set(g.step) != set(STEPS) for _, g in manifest.groupby("source")):
        raise ValueError("invalid nine-source four-checkpoint manifest")
    originals = pd.read_csv(args.data / "replay_cells.csv")
    references = {"original": originals[originals.variant == "baseline"],
                  "fresh": pd.read_csv(args.data / "fresh_replay_cells.csv")}
    streams = []
    for stream, ref in references.items():
        streams.append(validate(read_jsonl(args.data / f"trajectory_{stream}"), manifest, ref, stream))
    cells = pd.concat(streams, ignore_index=True)
    inventory = pd.read_json(args.data / "trajectory_checkpoint_inventory.json")
    if len(inventory) != 36 or inventory.model_id.duplicated().any() or not inventory.finite.all():
        raise ValueError("missing finite-checkpoint inventory")
    if not (cells.weights_sha256 == cells.model_id.map(inventory.set_index("model_id").weights_sha256)).all():
        raise ValueError("checkpoint digest differs from read-only inventory")
    input_receipt = json.loads((args.data / "trajectory_input_validation.json").read_text())
    if not input_receipt.get("all_cached_batches_identical") or input_receipt.get("stream_target_cells") != 10:
        raise ValueError("missing exact cached-input comparison")
    matched = streams[0].merge(streams[1], on=KEY, suffixes=("_original", "_fresh"), validate="one_to_one")
    if not (matched.weights_sha256_original == matched.weights_sha256_fresh).all() or (matched.episode_fingerprint_original == matched.episode_fingerprint_fresh).any():
        raise ValueError("changed weights or unchanged episodes across streams")
    cells.assign(sources=cells.sources.map(json.dumps)).to_csv(args.data / "trajectory_cells.csv", index=False)
    changes = endpoint_changes(cells)
    changes.to_csv(args.data / "trajectory_endpoint_changes.csv", index=False)
    summary = changes.groupby(["stream", "dataset", "decoder"]).change_100_to_2500.agg(
        mean="mean", minimum="min", maximum="max", positive=lambda x: int((x > 0).sum()),
        negative=lambda x: int((x < 0).sum()), sources="size")
    summary.to_csv(args.data / "trajectory_endpoint_summary.csv")
    (args.data / "trajectory_validation.json").write_text(json.dumps({
        "rows": len(cells), "steps": STEPS, "training_seeds": [0],
        "same_episode_inputs": True, "same_weights_across_streams": True,
        "terminal_weights_and_metrics_match_reference": True,
        "best_checkpoint_selected": False, "historical_step_zero_available": False,
    }, indent=2) + "\n")
    print(summary.loc[(slice(None), slice(None), ["S0_pool/ridge", "U1_pre_meta/ridge", "full_model"]), :].to_string())


if __name__ == "__main__":
    main()
