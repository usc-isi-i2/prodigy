"""Frozen Cora episodic breadth test; dry-run unless --execute is supplied."""
import argparse
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from scripts.experiments.setup.public_prodigy_kg.role_centering import predict
from scripts.experiments.setup.public_prodigy_kg.run_native import file_sha256, write_json
from scripts.experiments.setup.trace_health_guided.multiclass_metrics import episode_metrics


def probe(x, labels, query, tasks, center):
    """Never inspect query labels when fitting or predicting."""
    result = x.new_zeros(labels.shape)
    for task in tasks.unique(sorted=True):
        support = (tasks == task) & ~query
        target = (tasks == task) & query
        result[target] = predict(x[support], labels[support], x[target], center)
    return result[query]


def check_identities(centers, query, tasks):
    for task in tasks.unique(sorted=True):
        s = centers[(tasks == task) & ~query].tolist()
        q = centers[(tasks == task) & query].tolist()
        if len(set(s)) != len(s) or len(set(q)) != len(q) or set(s) & set(q):
            raise ValueError("Support/query centers must be unique and disjoint within each episode")


def query_truth(y_true_onehot, labels, query):
    """Validate production's one-hot return contract before scoring local classes."""
    expected = labels[query]
    if y_true_onehot.ndim != 2 or not torch.equal(y_true_onehot, expected):
        raise ValueError("Native one-hot query labels/order changed")
    if not torch.all((y_true_onehot == 0) | (y_true_onehot == 1)) or not torch.all(y_true_onehot.sum(1) == 1):
        raise ValueError("Expected one-hot classification query labels")
    return y_true_onehot.argmax(1)


def validate_features(graph):
    """Allow all columns only for this verified embedding-only citation schema."""
    x, names = graph["x"], graph["feature_names"]
    if x.ndim != 2 or x.shape[1] != 768:
        raise ValueError("Citation input must have exactly 768 embedding columns")
    if list(names) != [f"gte_{i}" for i in range(768)]:
        raise ValueError("Citation columns must be exactly ordered gte_0..gte_767")
    if not torch.is_floating_point(x) or not torch.isfinite(x).all():
        raise ValueError("Citation embeddings must be finite floating point")


def plan(args):
    return dict(dataset="cora", ways=7, shots=3, queries_per_class=4,
                batch_size=4, batches=32, episodes=128, episode_seed_offset=400009,
                episode_sampler_seed=448 + 400009,
                checkpoints={"blocked": str(args.blocked.resolve()),
                             "interleaved": str(args.interleaved.resolve())},
                graph=str(args.graph.resolve()), module_mode="eval", ridge_lambda=1,
                ridge_scale=1, ridge_intercept=False, centering=["none", "support"],
                normalization="row L2 after centering", sources=["ukr_rus", "covid", "midterm", "covid_political"],
                scope="Cross-domain episodic classification, not standard Cora benchmark",
                split="Native loader custom stratified 60/20/20; supports and queries from test pool",
                transduction="Full graph topology and unlabeled features are available; support-only readout fit",
                query_labels="Scoring only; never read by probe", trace_parity_atol=1e-5)


def execute(args, protocol):
    from data.tag_citation import get_cora_dataset
    from experiments.trainer import TrainerFS
    from scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy import resolved_params
    from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
        iter_episodes, new_fingerprint, reset_episode_rng, update_episode_fingerprint)
    from scripts.experiments.setup.target_performance_mechanisms.replay import (
        clone_batch, trace_stages, raw_stages, batch_hash)

    if args.output.exists():
        raise FileExistsError(args.output)
    for path in (args.graph, args.blocked, args.interleaved):
        if not path.is_file():
            raise FileNotFoundError(path)
    validate_features(torch.load(args.graph, map_location="cpu", weights_only=False))
    torch.set_num_threads(4)
    protocol["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    protocol["sha256"] = {str(p.resolve()): file_sha256(p) for p in (args.graph, args.blocked, args.interleaved)}
    args.output.mkdir(parents=True)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "execution_status.json", {"status": "running"})
    try:
        options = SimpleNamespace(config=args.config, eval_state_root=str(args.output / "state"),
            state_root=str(args.output / "state"), training_seed=0, eval_episode_seed_offset=400009,
            device=str(args.gpu), run_stamp="citation_breadth", log_root=str(args.output / "log"))
        target = dict(n_query=4, eval_random_query=False)
        params = resolved_params(options, "cora", target, args.graph, None, "citation_breadth", 2500)
        params.update(n_way=7, n_shots=3, n_query=4, batch_size=4,
                      test_len_cap=32, val_len_cap=32, dataset_len_cap=32,
                      feature_subset="all")
        write_json(args.output / "effective_params.json", params)
        dataset = get_cora_dataset(root=str(args.graph.parent), graph_filename=args.graph.name,
            n_hop=2, task_name="classification", feature_subset="all", seed=0,
            neighbor_sampling_hop_sizes="9,9", neighbor_sampling_node_limit=101)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        trainer = TrainerFS(dataset, params)
        cache = args.output / "batches"
        cache.mkdir()
        fingerprint, receipts = new_fingerprint(), []
        reset_episode_rng()
        for index, batch in enumerate(trainer.test_dataloader):
            batch = clone_batch(batch)
            query = batch[5].reshape(-1, 7)[:, 0].bool()
            check_identities(batch[0].global_node_ids[batch[0].ptr[:-1]], query, batch[0].task_id_per_sample)
            for episode in iter_episodes(batch, n_way=7, n_shot=3, n_query=4):
                update_episode_fingerprint(fingerprint, episode)
            path = cache / f"batch_{index:03d}.pt"
            torch.save(batch, path)
            receipts.append(dict(file=str(path.relative_to(args.output)), sha256=file_sha256(path), batch_hash=batch_hash(batch)))
        if len(receipts) != 32:
            raise ValueError("Expected exactly 32 batches")
        write_json(args.output / "cache.json", dict(records=receipts, episode_fingerprint=fingerprint.hexdigest()))
        rows, outputs = [], []
        for model_name, checkpoint in (("blocked", args.blocked), ("interleaved", args.interleaved)):
            trainer.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model"], strict=True)
            trainer.model.eval()
            directory = args.output / model_name
            directory.mkdir()
            with torch.no_grad():
                for index, receipt in enumerate(receipts):
                    path = args.output / receipt["file"]
                    if file_sha256(path) != receipt["sha256"]:
                        raise ValueError("Cached episode changed")
                    batch = clone_batch(torch.load(path, weights_only=False), trainer.device)
                    raw = raw_stages(batch[0])["raw_center"]
                    _, plain, _ = trainer.model(*clone_batch(batch, trainer.device))
                    with trace_stages(trainer.model, batch[0]) as traces:
                        yt, native, _ = trainer.model(*batch)
                    torch.testing.assert_close(plain, native, atol=1e-5, rtol=0)
                    u1 = traces["U1_pre_meta"]
                    query = batch[5].reshape(-1, 7)[:, 0].bool()
                    tasks = batch[0].task_id_per_sample
                    truth = query_truth(yt, batch[2], query)
                    predictions = {"native": native}
                    for name, x in (("raw", raw), ("u1", u1)):
                        for center in ("none", "support"):
                            predictions[f"{name}_{center}"] = probe(x, batch[2], query, tasks, center)
                    for task in tasks.unique(sorted=True):
                        mask = tasks[query] == task
                        metrics = {name: episode_metrics(truth[mask].cpu().numpy(), logits[mask].cpu().numpy())
                                   for name, logits in predictions.items()}
                        rows.append(dict(model=model_name, batch=index, task=int(task), metrics=metrics))
                    saved = directory / f"batch_{index:03d}.pt"
                    torch.save(dict(input_sha256=receipt["sha256"], y_true_onehot=yt.cpu(), local_query_labels=truth.cpu(),
                        predictions={k:v.cpu() for k,v in predictions.items()},
                        embeddings={"raw":raw.cpu(), "u1":u1.cpu()},
                        query_tasks=tasks[query].cpu()), saved)
                    outputs.append(dict(file=str(saved.relative_to(args.output)), sha256=file_sha256(saved)))
                    print(f"{model_name} batch {index + 1}/32", flush=True)
        means = {model: {arm: {metric: float(np.mean([r["metrics"][arm][metric] for r in rows if r["model"] == model]))
                    for metric in rows[0]["metrics"][arm]} for arm in rows[0]["metrics"]}
                 for model in ("blocked", "interleaved")}
        write_json(args.output / "summary.json", dict(rows=rows, means=means, outputs=outputs))
        write_json(args.output / "execution_status.json", dict(status="complete", models=2, episodes_per_model=128))
    except BaseException as exc:
        write_json(args.output / "execution_status.json", dict(status="failed", error=repr(exc)))
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    for name in ("blocked", "interleaved", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--graph", type=Path, default=Path("/dataMeR1/phil/data/cora/graphs/citation_graph.pt"))
    p.add_argument("--config", default="scripts/experiments/setup/final_core/training.yaml")
    p.add_argument("--gpu", type=int, choices=range(4), default=3)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    protocol = plan(args)
    print(json.dumps(protocol, indent=2))
    if args.execute:
        execute(args, protocol)


if __name__ == "__main__":
    main()
