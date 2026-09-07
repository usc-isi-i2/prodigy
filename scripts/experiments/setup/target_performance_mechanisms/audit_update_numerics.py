"""Bounded CPU numerical audit on synthetic NM episodes; not transfer evidence.

Use the production model, loss, optimizer and free-readout audit hook. No target
data or query outcomes are used. Never change completed checkpoints or defaults.
"""
import argparse
import copy
import gzip
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
from types import MethodType, SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("WANDB_MODE", "offline")

import torch
from torch_geometric.data import Data
import wandb

from data.dataset import SubgraphDataset
from experiments.params import get_params
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.sampler import NeighborSampler
from experiments.trainer import TrainerFS
from .readout_training_constraint import configure_trainer
from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest

BASE = Path(__file__).parent / "member_configs/00_memberctl_ukr_rus_lowest_sorted_s0.yaml"


def synthetic_dataset(feature_dim=768):
    rng = torch.Generator().manual_seed(7193)
    n = 320
    x = torch.randn(n, feature_dim, generator=rng)
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


def check_reference_inputs(batches, path):
    with gzip.open(path, "rt") as handle:
        expected = [json.loads(line) for line in islice(handle, len(batches))]
    actual = [{"step": i + 1, "batch_sha256": batch_hash(batch)} for i, batch in enumerate(batches)]
    if actual != expected:
        raise ValueError("real source batches do not match the completed training input audit")
    return {"path": str(path), "steps_checked": len(actual), "all_inputs_bit_exact": True}


def index_select_decode(self, input_x, label_x, metagraph_edge_index, edgelist_bipartite=False):
    """Same cosine decoder, replacing only its two advanced indexed reads."""
    ind0, ind1 = metagraph_edge_index[0], metagraph_edge_index[1]
    if edgelist_bipartite:
        return (self.cos(input_x.index_select(0, ind0), label_x.index_select(0, ind1)) + 1) / 2
    x = torch.cat((input_x, label_x))
    return self.cos(x.index_select(0, ind0), x.index_select(0, ind1)) * self.logit_scale.exp()


def replay_updates(trainer, template, batches, rng_state, folder, deterministic, hook, decoder_index_select=False):
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
    forward_parity = []
    if decoder_index_select:
        original_decode = model.decode
        def checked_decode(module, *args, **kwargs):
            value = index_select_decode(module, *args, **kwargs)
            with torch.no_grad():
                exact = torch.equal(value.detach(), original_decode(*args, **kwargs))
            if not exact:
                raise ValueError("index-select decoder changed forward values")
            forward_parity.append(exact)
            return value
        model.decode = MethodType(checked_decode, model)
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
        if not torch.isfinite(total).all() or not torch.isfinite(logits).all():
            raise ValueError("non-finite numerical diagnostic")
        total.backward()
        gradients = {k: p.grad.detach().clone() for k, p in model.named_parameters() if p.grad is not None}
        if not all(torch.isfinite(v).all() for v in gradients.values()):
            raise ValueError("non-finite gradient")
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
    return {"initial_sha256": initial, "deterministic": deterministic, "hook": hook, "steps": rows,
            "decoder_index_select": decoder_index_select, "decoder_forward_parity": forward_parity}, tensors


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--feature-dim", type=int, choices=(256, 768), default=768)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--localize-decoder", action="store_true")
    parser.add_argument("--real-training-input-audit", type=Path,
                        help="Use real Ukraine NM data; require exact prefix hashes from the completed training run.")
    args = parser.parse_args()
    if not 2 <= args.steps <= 8 or not 1 <= args.threads <= 8 or args.output.exists() or torch.cuda.is_available():
        raise ValueError("invalid bounded CPU diagnostic or existing output")
    real = args.real_training_input_audit is not None
    if real:
        if not args.initial_checkpoint or args.feature_dim != 768:
            raise ValueError("real source replay requires its actual initialization and 768-dimensional inputs")
        mem = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
        shm = os.statvfs('/dev/shm')
        if mem['MemAvailable'] < 512 * 1024**2 or shm.f_bavail * shm.f_frsize < 200 * 1024**3:
            raise RuntimeError("insufficient host/shared-memory headroom for real source replay")
    args.output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    primitive = indexed_gradient_audit()
    torch.use_deterministic_algorithms(False)
    params = get_params(["--config", str(BASE), "--device", "123", "--workers", "2" if real else "0",
                         "--dataset_len_cap", "2500" if real else str(args.steps), "--checkpoint_steps", "0",
                         "--prefix", "numerical_update_audit_only", "--timestamp", "fixed",
                         "--state_dir", str(args.output / "state"), "--log_dir", str(args.output / "log")])
    if real:
        from experiments.run_shared_graph import prepare_shared_dataset
        params['loader_start_method'] = 'spawn'
        torch.multiprocessing.set_sharing_strategy('file_system')
        dataset = prepare_shared_dataset(load_dataset(params))
    else:
        dataset = synthetic_dataset(args.feature_dim)
    seed_everything(params)
    trainer = TrainerFS(dataset, params)
    initial_rng = trainer._rng_state_dict()
    if args.initial_checkpoint:
        actual = torch.load(args.initial_checkpoint, map_location="cpu", weights_only=False)
        if actual["_training_checkpoint"]["completed_steps"] != 0:
            raise ValueError("only an actual step-zero training checkpoint is allowed")
        trainer.model.load_state_dict(actual["model"], strict=True)
        trainer.optimizer.load_state_dict(actual["_training_checkpoint"]["optimizer"])
        initial_rng = actual["_training_checkpoint"]["rng"]
        if real:
            if trainer._resume_parameter_contract() != actual['_training_checkpoint']['parameter_contract']:
                raise ValueError("real source training parameter contract differs")
            trainer._training_batch_sampler().load_state_dict(actual['_training_checkpoint']['train_batch_sampler'])
            trainer._restore_rng_state(initial_rng)
    template = copy.deepcopy((trainer.model, trainer.optimizer))
    batches = [clone_batch(b) for b in islice(trainer.train_dataloader, args.steps)]
    if len(batches) != args.steps:
        raise ValueError("incomplete fixed workload")
    reference_inputs = check_reference_inputs(batches, args.real_training_input_audit) if real else None
    if real:
        source_id = list(dataset.graph.source_graph_names).index('ukr_rus')
        for batch in batches:
            nodes = batch[0].global_node_ids
            if not (dataset.graph.graph_id[nodes[nodes >= 0]] == source_id).all():
                raise ValueError("real source replay contains another source")
    torch.save(batches, args.output / ("real_source_batches.pt" if real else "synthetic_batches.pt"))
    print(json.dumps({"inputs_ready": True, "real_source": real, "reference_inputs": reference_inputs}), flush=True)
    rows, comparisons = [], []
    modes = [("default", False, False), ("deterministic", True, False)]
    if args.localize_decoder:
        modes.append(("decoder_index_select", False, True))
    for mode, deterministic, decoder_index_select in modes:
        reference = None
        for hook in (False, True):
            for repeat in (0, 1):
                name = f"{mode}_{'hook' if hook else 'plain'}_{repeat}"
                result, tensors = replay_updates(trainer, template, batches, initial_rng, args.output / name,
                                                 deterministic, hook, decoder_index_select)
                result["name"] = name
                rows.append(result)
                if reference is None:
                    reference = tensors
                else:
                    comparisons.append({"name": name, "against": f"{mode}_plain_0",
                        "steps": [{"step": i + 1, **{key: differences(a[key], b[key])
                                  for key in ("logits", "gradients", "weights")}}
                                  for i, (a, b) in enumerate(zip(reference, tensors, strict=True))]})
                print(json.dumps({"completed": name, "losses": [r["loss"] for r in result["steps"]]}), flush=True)
    if len({r["initial_sha256"] for r in rows}) != 1 or not all(
            [s["input_sha256"] for s in row["steps"]] == [s["input_sha256"] for s in rows[0]["steps"]] for row in rows):
        raise ValueError("numerical audit did not hold inputs and initial weights fixed")
    receipt = {"complete": True, "research_model_result": False, "synthetic_workload": not real,
               "target_evaluation_episodes_used": False, "torch_version": torch.__version__, "threads": args.threads,
               "synthetic_input_dimension": None if real else args.feature_dim,
               "real_source": "ukr_rus" if real else None, "reference_inputs": reference_inputs,
               "actual_initial_checkpoint": str(args.initial_checkpoint) if args.initial_checkpoint else None,
               "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "indexed_gradient": primitive, "replays": rows, "comparisons": comparisons}
    (args.output / "DONE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    wandb.finish()
    print(json.dumps({"complete": True, "replays": len(rows), "steps_per_replay": args.steps}), flush=True)


if __name__ == "__main__":
    main()
