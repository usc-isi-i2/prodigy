"""Saved-example context sensitivity; descriptive paired test, dry-run by default."""
import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from .run import probe, query_truth, check_identities
from scripts.experiments.setup.public_prodigy_kg.run_native import file_sha256, write_json
from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics


def shuffle_checked(batch, seed):
    from scripts.experiments.setup.target_performance_mechanisms.replay import clone_batch, intervene
    result = clone_batch(batch, batch[0].x.device)
    before = result[0].clone()
    intervene(result, "shuffle_context", seed)
    centers = before.ptr[:-1]
    eligible = before.global_node_ids >= 0
    eligible[centers] = False
    changed = (before.x != result[0].x).any(1)
    if torch.any(changed & ~eligible):
        raise ValueError("Shuffle changed center/synthetic features")
    for key, value in before:
        if key != "x" and isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, result[0][key], atol=0, rtol=0)
        elif key != "x" and value != result[0][key]:
            raise ValueError(f"Shuffle changed graph metadata: {key}")
    for a, b in zip(batch[1:], result[1:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    return result, dict(changed_rows=int(changed.sum()), eligible_rows=int(eligible.sum()), seed=seed)


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("cora", "social", "checkpoint", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--gpu", type=int, choices=range(4), default=3)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    protocol = dict(sources={"cora":str(args.cora), "covid_political":str(args.social)},
        checkpoint=str(args.checkpoint), intervention="within-episode background-feature permutation",
        seed="710000 + batch index", mode="eval", batches_per_target=32,
        decoder="U1 support-centered ridge lambda1 scale1", parity_atol=1e-5,
        interpretation="Conditional sensitivity, not clean domain interaction: episode ways/shots differ")
    print(json.dumps(protocol, indent=2))
    if not args.execute:
        return
    from experiments.trainer import TrainerFS
    from data.tag_citation import get_cora_dataset
    from scripts.experiments.setup.icl_arch_matrix.common_protocol import build_classification_dataset, classification_targets
    from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import resolved_params
    from scripts.experiments.setup.target_performance_mechanisms.replay import clone_batch, trace_stages, batch_hash
    torch.set_num_threads(4)
    if json.loads((args.cora / "execution_status.json").read_text())["status"] != "complete" or not (args.social.parent / "DONE").is_file():
        raise ValueError("Completed source runs required")
    protocol.update(commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), checkpoint_sha256=file_sha256(args.checkpoint))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "execution_status.json", dict(status="running"))
    try:
        rows, receipts = [], []
        for target, root in (("cora", args.cora), ("covid_political", args.social)):
            cache = json.loads((root / "cache.json").read_text())
            if target == "cora":
                params = json.loads((root / "effective_params.json").read_text())
                source_protocol = json.loads((root / "protocol.json").read_text())
                if source_protocol["sha256"][source_protocol["checkpoints"]["blocked"]] != protocol["checkpoint_sha256"]:
                    raise ValueError("Cora checkpoint differs")
                graph = Path(source_protocol["graph"])
                dataset = get_cora_dataset(root=str(graph.parent), graph_filename=graph.name, n_hop=2,
                    task_name="classification", feature_subset="all", seed=0,
                    neighbor_sampling_hop_sizes="9,9", neighbor_sampling_node_limit=101)
                records = None
            else:
                spec = classification_targets("docs/graph_catalog.json", include_facebook=True)[target]
                dataset, _, graph = build_classification_dataset(dataset_name=target, data_root="/dataMeR1/phil/data", target=spec)
                opts = SimpleNamespace(config="scripts/experiments/setup/final_core/training.yaml", eval_state_root=str(args.output / "state"),
                    state_root=str(args.output / "state"), training_seed=0, eval_episode_seed_offset=0,
                    device=str(args.gpu), run_stamp="sensitivity", log_root=str(args.output / "log"))
                params = resolved_params(opts, target, spec, graph, None, "sensitivity", 2500)
                records = torch.load(root / "r4_blocked_s0_native__baseline.pt", map_location="cpu", weights_only=False)
                metadata = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
                checkpoint_paths = {r["checkpoint"] for r in metadata if r["model_id"] == "r4_blocked_s0_native"}
                if len(checkpoint_paths) != 1 or file_sha256(Path(next(iter(checkpoint_paths)))) != protocol["checkpoint_sha256"]:
                    raise ValueError("Social checkpoint differs")
            params.update(device=f"cuda:{args.gpu}", state_dir=str(args.output / "state"), log_dir=str(args.output / "log"))
            write_json(args.output / f"{target}_effective_params.json", params)
            trainer = TrainerFS(dataset, params)
            trainer.model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            trainer.model.eval()
            dest = args.output / target
            dest.mkdir()
            with torch.no_grad():
                for i in range(32):
                    path = root / "batches" / f"batch_{i:03d}.pt"
                    batch = torch.load(path, map_location="cpu", weights_only=False)
                    if target == "cora":
                        if file_sha256(path) != cache["records"][i]["sha256"]:
                            raise ValueError("Cora cache mismatch")
                        saved = torch.load(root / "blocked" / f"batch_{i:03d}.pt", map_location="cpu", weights_only=False)
                        ref_u1, ref_native = saved["embeddings"]["u1"], saved["predictions"]["native"]
                    else:
                        if batch_hash(batch) != cache["batch_sha256"][i] or records[i]["batch_sha256"] != cache["batch_sha256"][i]:
                            raise ValueError("Social cache mismatch")
                        ref_u1, ref_native = records[i]["embeddings"]["U1_pre_meta"], records[i]["logits"]["full_model"]
                    batch = clone_batch(batch, trainer.device)
                    query = batch[5].reshape(len(batch[2]), -1)[:, 0].bool()
                    tasks = batch[0].task_id_per_sample
                    check_identities(batch[0].global_node_ids[batch[0].ptr[:-1]], query, tasks)
                    raw = batch[0].x[batch[0].ptr[:-1]].clone()
                    altered, audit = shuffle_checked(batch, 710000 + i)
                    torch.testing.assert_close(raw, altered[0].x[altered[0].ptr[:-1]], atol=0, rtol=0)
                    predictions, embeddings = {}, {}
                    for variant, current in (("factual", batch), ("shuffled", altered)):
                        with trace_stages(trainer.model, current[0]) as traces:
                            yt, native, _ = trainer.model(*current)
                        truth = query_truth(yt, current[2], query)
                        u1 = traces["U1_pre_meta"]
                        if variant == "factual":
                            torch.testing.assert_close(u1.cpu(), ref_u1, atol=1e-5, rtol=0)
                            torch.testing.assert_close(native.cpu(), ref_native, atol=1e-5, rtol=0)
                        embeddings[variant] = u1.cpu()
                        predictions[variant] = probe(u1, current[2], query, tasks, "support").cpu()
                    predictions["raw"] = probe(raw, batch[2], query, tasks, "support").cpu()
                    for task in tasks.unique(sorted=True):
                        mask = (tasks[query] == task).cpu()
                        metrics = {k: episode_metrics(truth.cpu()[mask].numpy(), v[mask].numpy()) for k,v in predictions.items()}
                        rows.append(dict(target=target, batch=i, task=int(task), metrics=metrics,
                            shuffled_minus_factual={k:metrics["shuffled"][k]-metrics["factual"][k] for k in metrics["factual"]}))
                    output = dest / f"batch_{i:03d}.pt"
                    torch.save(dict(predictions=predictions, embeddings=embeddings, local_query_labels=truth.cpu(), query_tasks=tasks[query].cpu(), intervention=audit), output)
                    geometry_path = root / "blocked" / f"batch_{i:03d}.pt" if target == "cora" else root / "r4_blocked_s0_native__baseline.pt"
                    receipts.append(dict(file=str(output.relative_to(args.output)), sha256=file_sha256(output), source=str(path), source_sha256=file_sha256(path), geometry=str(geometry_path), geometry_sha256=file_sha256(geometry_path)))
                    print(f"{target} {i+1}/32", flush=True)
        means = {target:{key:float(np.mean([r["shuffled_minus_factual"][key] for r in rows if r["target"]==target])) for key in rows[0]["shuffled_minus_factual"]} for target in ("cora", "covid_political")}
        write_json(args.output / "summary.json", dict(rows=rows, mean_effects=means, receipts=receipts))
        write_json(args.output / "execution_status.json", dict(status="complete"))
    except BaseException as exc:
        write_json(args.output / "execution_status.json", dict(status="failed", error=repr(exc)))
        raise


if __name__ == "__main__":
    main()
