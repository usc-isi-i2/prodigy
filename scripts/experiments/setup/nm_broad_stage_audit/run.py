"""Capture episode-relative NM stages for four specialists and four targets."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("WANDB_MODE", "disabled")
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
FINAL_CORE = ROOT / "scripts/experiments/setup/final_core"
sys.path[:0] = [str(ROOT), str(FINAL_CORE)]

from evaluate_fixed_grid import (  # noqa: E402
    AuditedLoader,
    fingerprint_plan,
    git_commit,
    load_dataset,
    make_frozen_loader,
    reset_fixed_eval_rng,
    seed_everything,
)
from experiments.trainer import TrainerFS  # noqa: E402
from scripts.experiments.setup.nm_complete_input_audit.run import pack, update_hash  # noqa: E402
from scripts.experiments.setup.nm_hk_mechanism.run import build_model  # noqa: E402
from scripts.experiments.setup.nm_source_stage_audit.run import matrix_metrics  # noqa: E402
from scripts.experiments.setup.nm_support_resampling.run import (  # noqa: E402
    capture,
    digest_state,
    episode_inputs,
)
from scripts.experiments.setup.nm_support_geometry.run import digest  # noqa: E402


DEFAULT_SOURCES = ("ukr_rus", "covid", "midterm", "cp_hk")
ARCHIVE_STATE = Path(
    "/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/"
    "prodigy-final-core/files/state/final_core"
)


def parse_csv(text):
    values = tuple(value.strip() for value in text.split(",") if value.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError(f"invalid distinct CSV values: {text!r}")
    return values


def checkpoint_path(root, source):
    return root / f"finalcore_ss_{source}_s0_20260807/checkpoint/state_dict_2500.ckpt"


def load_fingerprints(path):
    frame = pd.read_csv(path, sep="\t").set_index("target")
    if not frame.index.is_unique:
        raise ValueError("duplicate reference target fingerprints")
    return frame


def expected_accuracy(root, model, target):
    path = root / "seed_0" / f"ss_{model}" / f"{target}.json"
    payload = json.loads(path.read_text())
    if payload["target"] != target or payload["model_id"] != f"ss_{model}":
        raise ValueError(f"unexpected fixed-result identity in {path}")
    return path, float(payload["accuracy"])


def self_test():
    z = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    truth = torch.tensor([0, 2])
    metrics = matrix_metrics(z, truth, "x")
    assert metrics["x_prediction"].tolist() == [0, 1]
    assert metrics["x_rank"].tolist() == [1, 2]
    assert np.allclose(metrics["x_margin"], [1.0, -1.0])
    print("SELF_TEST_OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(
        "/dataMeR1/phil/gfm/error_audit/nm_canonical_split_20260908/effective_config.json"))
    parser.add_argument("--checkpoint-root", type=Path, default=ARCHIVE_STATE)
    parser.add_argument("--fixed-results-root", type=Path, default=ROOT /
        "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
        "prodigy_final_core/fixed_test/results")
    parser.add_argument("--fingerprints", type=Path, default=ROOT /
        "scripts/experiments/analysis/transfer/matrices/cross_model/final_core/data/"
        "prodigy_final_core/fixed_test/summary/episode_fingerprints.tsv")
    parser.add_argument("--targets", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--models", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--out", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    targets, model_names = parse_csv(args.targets), parse_csv(args.models)
    if args.batch_size <= 0 or 512 % args.batch_size:
        raise ValueError("batch-size must be a positive divisor of 512")
    unknown = (set(targets) | set(model_names)) - set(DEFAULT_SOURCES)
    if unknown:
        raise ValueError(f"unsupported sources: {sorted(unknown)}")
    checkpoints = {name: checkpoint_path(args.checkpoint_root, name) for name in model_names}
    expected = {}
    for target in targets:
        for model in model_names:
            path, accuracy = expected_accuracy(args.fixed_results_root, model, target)
            expected[(target, model)] = {"path": path, "accuracy": accuracy}
    references = load_fingerprints(args.fingerprints)
    required_paths = [args.config, args.fingerprints, *checkpoints.values(),
                      *(item["path"] for item in expected.values())]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    plan = {
        "protocol": "nm_broad_source_stage_v1",
        "targets": list(targets),
        "models": list(model_names),
        "episodes_per_target": 512,
        "queries_per_cell": 61440,
        "checkpoint_step": 2500,
        "batch_size": args.batch_size,
        "device": args.device,
        "training": False,
        "new_episode_sampling": False,
        "episode_protocol": "published fixed-test deterministic reconstruction",
        "checkpoints": {name: str(path) for name, path in checkpoints.items()},
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    if args.out is None:
        raise ValueError("--out is required outside --dry-run")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    started = time.time()

    params = json.loads(args.config.read_text())
    params.update(device=args.device, batch_size=args.batch_size,
                  test_len_cap=512 // args.batch_size, workers=0,
                  eval_only=True, eval_only_split="test",
                  log_dir=str(args.out / "logs"), state_dir=str(args.out / "state"),
                  override_log=True)
    seed_everything(params)
    dataset = load_dataset(params)
    if dataset.nm_background_edge_view != "static_train" or dataset.nm_holdout_edge_view != "static_test":
        raise AssertionError("unexpected NM graph views")
    trainer = TrainerFS(dataset, params)

    models = {}
    state_hashes = {}
    checkpoint_info = {}
    for name, path in checkpoints.items():
        checkpoint_info[name] = {"path": str(path), "sha256": digest(path)}
        models[name] = build_model(params, path, args.device)
        state_hashes[name] = digest_state(models[name])

    report = {
        **plan,
        "revision": git_commit(),
        "complete": False,
        "config": {"path": str(args.config), "sha256": digest(args.config)},
        "fingerprint_reference": {"path": str(args.fingerprints), "sha256": digest(args.fingerprints)},
        "checkpoints": checkpoint_info,
        "model_state_hashes": state_hashes,
        "targets": {},
    }
    truth = torch.repeat_interleave(torch.arange(30), 4)
    query_slots = torch.tensor([slot for slot in range(210) if slot % 7 >= 3])

    for target in targets:
        target_started = time.time()
        reset_fixed_eval_rng(target)
        trainer.parameter.update(neighbor_sampling_source_subset=target, eval_only_split="test")
        loader = trainer._build_dataloaders(dataset, trainer.dataset_name)[3]
        batches = list(loader.batch_sampler)
        raw_fingerprint, episode_count = fingerprint_plan(
            target, batches, expected_batch_size=args.batch_size, dataset=dataset)
        expected_raw = references.loc[target, "episode_plan_fingerprint"]
        if (raw_fingerprint != expected_raw or episode_count != 512
                or len(batches) != 512 // args.batch_size):
            raise AssertionError((target, raw_fingerprint, expected_raw, episode_count, len(batches)))

        reset_fixed_eval_rng(target)
        audited = AuditedLoader(
            make_frozen_loader(dataset, loader.collate_fn, batches), args.batch_size)
        target_dir = args.out / target
        target_dir.mkdir()
        tables = []
        batch_receipts = []
        input_hash = hashlib.sha256()
        input_hash.update(b"nm-broad-stage-full-input-v1\0")
        accuracy_counts = {name: 0 for name in model_names}

        for batch_index, batch in enumerate(audited):
            update_hash(input_hash, batch)
            input_path = target_dir / f"batch_{batch_index:03d}_input_private.pt"
            pack(batch, dataset.graph.x, input_path)
            packed = {
                "schema": 1,
                "target": target,
                "batch_index": batch_index,
                "model_states": state_hashes,
                "input_sha256": digest(input_path),
                "models": {},
            }
            graph = batch[0]
            with torch.inference_mode():
                for model_name, model in models.items():
                    saved, logits = capture(model, batch, args.device)
                    logits = logits.reshape(args.batch_size, 120, 30)
                    episodes = {}
                    for episode in range(args.batch_size):
                        start = episode * 210
                        local_query_slots = query_slots + start
                        pre = saved["pre"][start:start + 210]
                        support = pre.reshape(30, 7, -1)[:, :3]
                        query = F.normalize(pre[query_slots.to(pre.device)], dim=1)
                        mean_cosine = query @ F.normalize(support, dim=2).mean(dim=1).T
                        final = logits[episode]
                        rows = {
                            "episode": [f"{batch_index}:{episode}"] * 120,
                            "sample": local_query_slots.numpy(),
                            "query": graph.center_node_idx[local_query_slots].cpu().numpy(),
                            "anchor": graph.task_label_map[episode, truth].cpu().numpy(),
                            "model": [model_name] * 120,
                        }
                        rows.update(matrix_metrics(mean_cosine, truth.to(mean_cosine.device), "pre"))
                        rows.update(matrix_metrics(final, truth.to(final.device), "final"))
                        frame = pd.DataFrame(rows)
                        accuracy_counts[model_name] += int(frame.final_correct.sum())
                        tables.append(frame)
                        episodes[episode] = {
                            "inputs": tuple(value.cpu() for value in episode_inputs(batch, saved, episode)),
                            "final_logits": final.cpu(),
                            "mean_cosine": mean_cosine.cpu(),
                        }
                    packed["models"][model_name] = episodes
                    del saved, logits
                    torch.cuda.empty_cache()
            score_path = target_dir / f"batch_{batch_index:03d}_stages_private.pt"
            torch.save(packed, score_path)
            batch_receipts.append({
                "input_file": input_path.name,
                "input_sha256": packed["input_sha256"],
                "stage_file": score_path.name,
                "stage_sha256": digest(score_path),
                "stage_bytes": score_path.stat().st_size,
            })
            print("BATCH_OK", target, batch_index, "elapsed", round(time.time() - started, 1), flush=True)
            del packed, batch, graph
            gc.collect()

        observed_expected = references.loc[target, "observed_episode_fingerprint"]
        if audited.fingerprint != observed_expected:
            raise AssertionError((target, audited.fingerprint, observed_expected))
        table = pd.concat(tables, ignore_index=True)
        if len(table) != 61440 * len(model_names) or table.duplicated(
                ["episode", "sample", "model"]).any():
            raise AssertionError(f"unexpected row table for {target}")
        csv_path = target_dir / "stage_predictions_private.csv"
        table.to_csv(csv_path, index=False)
        parity = {}
        for model_name in model_names:
            observed = accuracy_counts[model_name] / 61440
            wanted = expected[(target, model_name)]["accuracy"]
            difference_queries = abs(observed - wanted) * 61440
            if difference_queries > 3.01:
                raise AssertionError((target, model_name, observed, wanted, difference_queries))
            parity[model_name] = {
                "observed_accuracy": observed,
                "published_accuracy": wanted,
                "absolute_difference": abs(observed - wanted),
                "published_result": str(expected[(target, model_name)]["path"]),
                "published_result_sha256": digest(expected[(target, model_name)]["path"]),
            }
        report["targets"][target] = {
            "raw_plan_fingerprint": raw_fingerprint,
            "observed_fingerprint": audited.fingerprint,
            "full_input_tensors_sha256": input_hash.hexdigest(),
            "rows": len(table),
            "csv_sha256": digest(csv_path),
            "batches": batch_receipts,
            "accuracy_parity": parity,
            "seconds": time.time() - target_started,
        }
        (args.out / "receipt.partial.json").write_text(json.dumps(report, indent=2) + "\n")
        del tables, table, batches
        gc.collect()
        torch.cuda.empty_cache()

    for name, model in models.items():
        if digest_state(model) != state_hashes[name]:
            raise AssertionError(f"model state changed for {name}")
    report.update(complete=True, model_states_unchanged=True, seconds=time.time() - started)
    (args.out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETE", round(report["seconds"], 2), flush=True)


if __name__ == "__main__":
    main()
