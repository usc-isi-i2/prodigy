"""Bounded CPU numerical audit on synthetic NM episodes; not transfer evidence.

Use the production model, loss, optimizer and free-readout audit hook. No target
data or query outcomes are used. Never change completed checkpoints or defaults.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import torch
from torch_geometric.data import Data
import wandb

from data.dataset import SubgraphDataset
from experiments.params import get_params
from experiments.run_single_experiment import seed_everything
from experiments.sampler import NeighborSampler
from experiments.trainer import TrainerFS
from .readout_training_constraint import configure_trainer
from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest

BASE = Path(__file__).parent / "member_configs/00_memberctl_ukr_rus_lowest_sorted_s0.yaml"


def synthetic_dataset():
    rng = torch.Generator().manual_seed(7193)
    n = 320
    x = torch.randn(n, 256, generator=rng)
    x = torch.nn.functional.normalize(x, dim=1)
    edges = torch.tensor([(i, (i + j) % n) for i in range(n) for j in range(1, 25)]).T.contiguous()
    graph = Data(x=x, edge_index=edges, y=torch.zeros(n).long(),
                 graph_id=torch.zeros(n).long(), num_nodes=n,
                 source_graph_names=["ukr_rus"])
    dataset = SubgraphDataset(graph, NeighborSampler(graph, 2, hop_sizes=[9, 9], limit=101, walk_hops=1),
                              bidirectional=False)
    for name in ("nm_validation_neighbor_sampler", "nm_test_neighbor_sampler", "nm_holdout_neighbor_sampler"):
        setattr(dataset, name, dataset.neighbor_sampler)
    return dataset


def differences(a, b):
    if a.keys() != b.keys():
        raise ValueError("tensor inventory differs")
    changed = {k: float((a[k].double() - b[k].double()).abs().max())
               for k in a if not torch.equal(a[k], b[k])}
    return {"bit_exact": not changed, "changed_keys": len(changed),
            "maximum_absolute_difference": max(changed.values(), default=0),
            "largest_differences": dict(sorted(changed.items(), key=lambda x: x[1], reverse=True)[:5])}


def indexed_gradient_audit():
    rng = torch.Generator().manual_seed(441)
    x = torch.randn(1000, 256, generator=rng, requires_grad=True)
    index = torch.randint(1000, (64000,), generator=rng)
    weights = torch.randn(64000, 256, generator=rng)
    rows = []
    for deterministic in (False, True):
        torch.use_deterministic_algorithms(deterministic)
        grads = []
        for _ in range(3):
            x.grad = None
            (x[index] * weights).sum().backward()
            grads.append(x.grad.clone())
        rows.append({"deterministic": deterministic,
                     "comparisons": [differences({"gradient": grads[0]}, {"gradient": g}) for g in grads[1:]]})
    return rows


def replay_updates(trainer, template, batches, rng_state, folder, deterministic, hook):
    torch.use_deterministic_algorithms(deterministic)
    model, optimizer = copy.deepcopy(template)
    parameter_ids = {id(v) for v in model.parameters()}
    if any(id(p) not in parameter_ids for group in optimizer.param_groups for p in group["params"]):
        raise ValueError("copied optimizer is not attached to copied model")
    folder.mkdir()
    shell = SimpleNamespace(model=model, optimizer=optimizer,
                            parameter={**trainer.parameter, "readout_training_condition": "free"},
                            resume_step=0, logging_dir=str(folder))
    if hook:
        configure_trainer(shell)
    trainer._restore_rng_state(rng_state)
    initial = model_digest(model.state_dict())
    rows, tensors = [], []
    for step, cached in enumerate(batches, 1):
        batch = clone_batch(cached)
        expected_input = batch_hash(cached)
        if batch_hash(batch) != expected_input:
            raise ValueError("input clone changed values")
        model.train()
        optimizer.zero_grad()
        truth, logits, graph = model(*batch)
        loss, _ = trainer.get_loss_and_acc(truth, logits)
        total = loss + trainer.get_aux_loss(graph) * trainer.parameter["attr_regression_weight"]
        total.backward()
        gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        optimizer.step()
        if hook:
            shell.training_step_observer(step)
        state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        tensors.append({"logits": {"logits": logits.detach().clone()}, "gradients": gradients, "weights": state})
        rows.append({"step": step, "input_sha256": expected_input, "loss": float(loss.detach()),
                     "logits_sha256": model_digest(tensors[-1]["logits"]),
                     "gradients_sha256": model_digest(gradients), "weights_sha256": model_digest(state)})
        if batch_hash(cached) != expected_input:
            raise ValueError("production forward changed the retained input")
    return {"initial_sha256": initial, "deterministic": deterministic, "hook": hook, "steps": rows}, tensors


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if not 2 <= args.steps <= 8 or not 1 <= args.threads <= 8 or args.output.exists() or torch.cuda.is_available():
        raise ValueError("invalid bounded CPU diagnostic or existing output")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    primitive = indexed_gradient_audit()
    torch.use_deterministic_algorithms(False)
    dataset = synthetic_dataset()
    params = get_params(["--config", str(BASE), "--device", "123", "--workers", "0",
                         "--dataset_len_cap", str(args.steps), "--checkpoint_steps", "0",
                         "--prefix", "synthetic_update_audit_only", "--timestamp", "fixed",
                         "--state_dir", str(args.output / "state"), "--log_dir", str(args.output / "log")])
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    initial_rng = trainer._rng_state_dict()
    template = copy.deepcopy((trainer.model, trainer.optimizer))
    batches = [clone_batch(b) for b in trainer.train_dataloader]
    if len(batches) != args.steps:
        raise ValueError("incomplete fixed synthetic workload")
    torch.save(batches, args.output / "synthetic_batches.pt")
    rows, comparisons = [], []
    for deterministic in (False, True):
        reference = None
        for hook in (False, True):
            for repeat in (0, 1):
                name = f"{'deterministic' if deterministic else 'default'}_{'hook' if hook else 'plain'}_{repeat}"
                result, tensors = replay_updates(trainer, template, batches, initial_rng, args.output / name,
                                                 deterministic, hook)
                result["name"] = name
                rows.append(result)
                if reference is None:
                    reference = tensors
                else:
                    comparisons.append({"name": name, "against": f"{'deterministic' if deterministic else 'default'}_plain_0",
                        "steps": [{"step": i + 1, **{key: differences(a[key], b[key])
                                  for key in ("logits", "gradients", "weights")}}
                                  for i, (a, b) in enumerate(zip(reference, tensors, strict=True))]})
                print(json.dumps({"completed": name, "losses": [r["loss"] for r in result["steps"]]}), flush=True)
    if len({r["initial_sha256"] for r in rows}) != 1 or not all(
            [s["input_sha256"] for s in row["steps"]] == [s["input_sha256"] for s in rows[0]["steps"]] for row in rows):
        raise ValueError("numerical audit did not hold inputs and initial weights fixed")
    receipt = {"complete": True, "research_model_result": False, "synthetic_workload": True,
               "target_data_used": False, "torch_version": torch.__version__, "threads": args.threads,
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "indexed_gradient": primitive, "replays": rows, "comparisons": comparisons}
    (args.output / "DONE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    wandb.finish()
    print(json.dumps({"complete": True, "replays": len(rows), "steps_per_replay": args.steps}), flush=True)


if __name__ == "__main__":
    main()
