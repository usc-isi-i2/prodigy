"""Isolated launcher for the pinned, unmodified public PRODIGY KG recipe.

Dry run by default: --execute is required to import the upstream runtime, load
data, or train. Observation wrappers write provenance and native metrics only;
they never call eval(), replace a forward, repair a split, or change aggregation.
"""
from __future__ import annotations

import argparse
import ast
from collections.abc import Set
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shlex
import subprocess
import sys
import time


UPSTREAM_COMMIT = "107ba57234d3188227cda5b78a2dbcfb84a1c694"
NOMINATED_CHECKPOINT = "state_dict_8000.ckpt"
CORE_VERSIONS = {
    "torch": "2.0.1", "torch-geometric": "2.3.1",
    "sentence-transformers": "2.2.2", "transformers": "4.29.2",
}
PROJECT_PACKAGES = ("experiments", "models", "data")


def literal_assignment(source: Path, name: str, function: str | None = None):
    """Read a literal from released source without importing or executing it."""
    tree = ast.parse(source.read_text())
    statements = tree.body
    if function is not None:
        matches = [node for node in statements
                   if isinstance(node, ast.FunctionDef) and node.name == function]
        if len(matches) != 1:
            raise ValueError(f"Expected one function {function} in {source}")
        statements = matches[0].body
    values = [node.value for node in statements if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == name
                      for target in node.targets)]
    if len(values) != 1:
        raise ValueError(f"Expected one literal {name} in {source}")
    return ast.literal_eval(values[0])


def absolute_path(path: Path, name: str) -> Path:
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must be an absolute path without '..'")
    if path.is_symlink():
        raise ValueError(f"Refusing symlink {name}: {path}")
    return path.resolve()


def replace_option(arguments: list[str], flag: str, value: str) -> None:
    if arguments.count(flag) != 1:
        raise ValueError(f"Expected exactly one released {flag}")
    index = arguments.index(flag)
    arguments[index + 1] = value


def verify_upstream(upstream: Path) -> dict:
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(upstream), *arguments],
                                       text=True).strip()
    head = git("rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise ValueError(f"Wrong upstream revision: {head}")
    status = git("status", "--porcelain", "--untracked-files=all")
    if status:
        raise ValueError(f"Upstream must be clean, including untracked files:\n{status}")
    return {"head": head, "clean": True}


def build_plan(*, upstream: Path, root: Path, output: Path, phase: str,
               gpu: int, seed: int = 0, checkpoint: Path | None = None) -> dict:
    """Build commands/protocol only; no model imports, output creation or data IO."""
    if phase not in ("smoke", "train", "eval") or gpu not in (2, 3):
        raise ValueError("Use phase smoke/train/eval and owned GPU 2 or 3")
    upstream = absolute_path(upstream, "upstream")
    root = absolute_path(root, "root")
    output = absolute_path(output, "output")
    if output == root or output == upstream or output in root.parents or output in upstream.parents:
        raise ValueError("Output must not contain the assets or upstream checkout")
    if output.is_relative_to(upstream) or output.is_relative_to(root):
        raise ValueError("Output must be separate from assets and upstream")
    if output.exists():
        raise FileExistsError(f"Output must be new: {output}")
    if phase == "eval":
        if checkpoint is None:
            raise ValueError("Eval requires the pre-nominated checkpoint")
        checkpoint = absolute_path(checkpoint, "checkpoint")
        if checkpoint.name != NOMINATED_CHECKPOINT:
            raise ValueError(f"Eval only accepts {NOMINATED_CHECKPOINT} (8001 updates)")
    elif checkpoint is not None:
        raise ValueError("Smoke/train must start natively, without a checkpoint")

    source = upstream / "kg_commands.py"
    deviations = [
        "Explicit seed; the released command supplies no seed.",
        "Phase-specific fixed timestamp and isolated absolute log/state directories.",
        "Owned physical GPU selected by CUDA_VISIBLE_DEVICES; native device is cuda:0.",
        "Offline W&B and observation-only parameter/model/metric/initial-state capture.",
        "Python3.11 only: scoped MulticlassTask RNG proxy restores Python3.10 Random.sample(set) "
        "by converting to tuple without sorting; original RNG state and other sampling are retained.",
    ]
    if phase in ("smoke", "train"):
        template = literal_assignment(source, "pretrain_cmds")["PRODIGY"]
        native = shlex.split(template.format(root=shlex.quote(str(root)), device=0))
        if phase == "smoke":
            for flag, value in (("-ds_cap", "3"), ("--workers", "0"),
                                ("-val_cap", "1"), ("-test_cap", "1")):
                replace_option(native, flag, value)
            deviations.append("Smoke only: 3 updates, workers=0, val/test batch caps=1; "
                              "unchanged model, batch size, task mix, augmentations and loss.")
        dataset = "Wiki"
    else:
        template = literal_assignment(source, "cmd_template",
                                      "print_in_context_learning_evaluation_cmds")
        runs = literal_assignment(source, "in_context_learning_eval_runs")
        if runs[0][1:] != ["S2,UX,M2", "InContext_eval_PRODIGY"]:
            raise ValueError("Unexpected released evaluation model")
        if literal_assignment(source, "n_rels")["FB15K-237"] != 200:
            raise ValueError("Unexpected released FB relation candidate count")
        suffix = "--no_split_labels True --label_set " + " ".join(map(str, range(200)))
        native = shlex.split(template.format(
            dataset_path=shlex.quote(str(root)), ds="FB15K-237", nshot=3,
            device=0, layers=runs[0][1], prefix=runs[0][2], nway=20,
            pos_only_msg="-meta_pos True", pt_suffix="-pretrained " + shlex.quote(str(checkpoint)),
            ignore_lbl_embs="--ignore_label_embeddings True", suffix_lblsplit=suffix))
        dataset = "FB15K-237"
        deviations.append("Checkpoint pre-nominated at native loop index 8000 (=8001 updates); "
                          "neither target-selected nor the terminal/source-best snapshot.")
        deviations.append("Eval-only observation hooks save immutable CPU copies of every actual "
                          "pre-forward argument tuple and paired native labels/logits; no resampling.")
    if native[:2] != ["python3", "experiments/run_single_experiment.py"]:
        raise ValueError("Unexpected released entry point")
    native += ["--seed", str(seed), "--timestamp", f"native_{phase}_seed{seed}",
               "--log_dir", str(output / "log"), "--state_dir", str(output / "state")]
    command = [sys.executable, str(upstream / native[1]), *native[2:]]
    return {
        "schema_version": 1, "phase": phase, "upstream": str(upstream),
        "upstream_commit": UPSTREAM_COMMIT, "root": str(root), "output": str(output),
        "dataset": dataset, "seed": seed, "physical_gpu": gpu,
        "command": command, "shell_display_only": shlex.join(command),
        "environment": {"CUDA_VISIBLE_DEVICES": str(gpu), "WANDB_MODE": "offline",
                        "WANDB_DIR": str(output / "wandb"), "PYTHONDONTWRITEBYTECODE": "1"},
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "nomination": {"filename": NOMINATED_CHECKPOINT, "optimizer_updates": 8001,
                       "selection": "Fixed before FB outcomes; no automatic target selection."},
        "required_asset_receipts": [dataset], "deviations_from_released_command": deviations,
        "native_protocol_preserved": {
            "architecture": "S2,UX,M2; 256d; edge features; 770d node input including head/tail flags",
            "pooling": "UX projects head + tail + global-max concatenation; encoder center reset also uses global mean",
            "aggregation": "Native PyG2.3.1 AddAggregation retained despite SAGE self.aggr='mean'",
            "labels": "ignore_label_embeddings=True in both phases; frozen random embedding table",
            "training": "10010 updates, batch10, 15way, 3shot, 4query, cls_nm (98:2), attr weight1000",
            "training_validation": "all_test=True; shared Wiki graph/classes; different episode streams, not disjoint edge pools",
            "eval": "FB20way/3shot/4query, batch1, 500 episodes, 40000 queries, workers15; no train_cap",
            "eval_split": "Unmodified alias/complement mask: native test support=train+test (~90%), query=valid (~10%); pools disjoint",
            "eval_mode": "Native eval_only starts in train mode, including BN buffer updates; no model.eval() inserted",
            "periodic_training_eval_mode": "Native periodic source eval calls model.eval(); initial source evals do not",
            "sampler": "Wiki2hop, FB1hop; fanout100, node limit2000; native per-example center edge removal",
            "multi_layer_caveat": "Two metagraph layers; the social one-M fixed-query substitution theorem is not assumed",
            "checkpoint_steps": "Loop-index filename k is saved after optimizer update k+1; no terminal save added",
            "scope": "No suppression/intervention, aggregation fix, split fix, lazy label interface, or model rewrite",
        },
        "runtime_contract": {"python_major_minor": ["3.10", "3.11"], "core_packages": CORE_VERSIONS,
                             "sbert": "Native constructor retained; all-mpnet-base-v2 must be cached under root/sbert or retrieved by native code"},
        "dry_run": True,
    }


def verify_receipts(root: Path, datasets: list[str]) -> list[dict]:
    stage = Path(__file__).with_name("stage_assets.py")
    expected = literal_assignment(stage, "ASSETS")
    extras = literal_assignment(stage, "EXTRA_REQUIRED")
    feature = literal_assignment(stage, "FEATURE_FILE")
    receipts = []
    for dataset in datasets:
        path = root / "receipts" / f"{dataset}.json"
        if path.is_symlink():
            raise ValueError(f"Refusing symlink receipt: {path}")
        receipt = json.loads(path.read_text())
        target = root / dataset
        if (receipt.get("dataset") != dataset or receipt.get("status") != "complete"
                or receipt.get("expected") != expected[dataset]
                or receipt.get("url") != f"https://snap.stanford.edu/prodigy/{dataset}.zip"
                or receipt.get("target") != str(target)):
            raise ValueError(f"Missing/mismatched complete receipt: {path}")
        digest = receipt.get("sha256", "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid recorded archive SHA256: {path}")
        for relative in ("graph.pt", f"{dataset}_adj.pt", feature, *extras[dataset]):
            asset = target / relative
            if asset.is_symlink() or not asset.is_file() or asset.stat().st_size == 0:
                raise ValueError(f"Missing/non-regular/empty asset: {asset}")
        receipts.append({"path": str(path), "receipt": receipt,
                         "verification_scope": "Matching staging receipt and required file presence; not a new archive hash/content validation"})
    return receipts


def runtime_versions() -> dict:
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise RuntimeError("Use the audited Python3.10/3.11 prodigy runtime")
    versions = {name: importlib.metadata.version(name) for name in (
        *CORE_VERSIONS, "torch-scatter", "torch-sparse", "numpy", "scipy",
        "scikit-learn", "pandas", "wandb", "lmdb", "ogb")}
    for name, expected in CORE_VERSIONS.items():
        if versions[name].split("+")[0] != expected:
            raise RuntimeError(f"Expected {name}=={expected}, got {versions[name]}")
    return {"python": sys.version, "executable": sys.executable, "packages": versions}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + "\n")


def verify_imports(upstream: Path) -> dict:
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in PROJECT_PACKAGES:
            continue
        filename = getattr(module, "__file__", None)
        namespace = list(getattr(module, "__path__", []))
        if filename is None:
            if not namespace or any(not Path(path).resolve().is_relative_to(upstream) for path in namespace):
                raise RuntimeError(f"Foreign/ambiguous native namespace: {name}: {namespace}")
            paths[name] = {"namespace_paths": [str(Path(path).resolve()) for path in namespace]}
        elif (not Path(filename).resolve().is_relative_to(upstream)
              or any(not Path(path).resolve().is_relative_to(upstream) for path in namespace)):
            raise RuntimeError(f"Foreign/ambiguous native module already imported: {name}: {filename}")
        else:
            paths[name] = str(Path(filename).resolve())
    return paths


class LegacySetSampleRNG:
    """Restore only pre-3.11 sample(set) conversion, using the SAME native RNG."""
    def __init__(self, rng):
        self.rng = rng

    def sample(self, population, k, **kwargs):
        if isinstance(population, Set):
            population = tuple(population)  # Python3.10 did not sort the set.
        return self.rng.sample(population, k, **kwargs)

    def __getattr__(self, name):
        return getattr(self.rng, name)


def install_sampler_compat(task_class, python_version=None) -> dict:
    """No source/container edits: wrap only MulticlassTask's sample RNG on3.11."""
    version = tuple(python_version or sys.version_info[:2])
    active = version == (3, 11)
    if active:
        original = task_class.sample

        def compatible_sample(self, num_label, num_member, num_shot, num_query, rng):
            return original(self, num_label, num_member, num_shot, num_query,
                            LegacySetSampleRNG(rng))

        task_class.sample = compatible_sample
    return {"active": active, "python_major_minor": list(version),
            "callsite": "data/dataloader.py:134 MulticlassTask.sample",
            "correction": "sample(set) -> sample(tuple(set)); no sorting, reseeding, RNG cloning, or population mutation"}


def install_episode_capture(model, *, output: Path, torch_module) -> None:
    """Capture exact native eval inputs before in-place mutations, without reruns."""
    directory = output / "episode_capture"
    directory.mkdir(exist_ok=False)
    records, pending = [], []

    def pre_forward(_module, arguments):
        if pending:
            raise RuntimeError("Unexpected nested forward during native episode capture")
        ordinal = len(records)
        inputs = directory / f"batch_{ordinal:05d}.pt"
        # Native arguments are Data/Batch or tensors. Their clone().cpu() recursively
        # clones graph tensor attributes too; neither operation samples RNG.
        copies = tuple(value.clone().cpu() for value in arguments)
        torch_module.save(copies, inputs)
        pending.append({"ordinal": ordinal, "input_file": inputs.name,
                        "input_sha256": file_sha256(inputs), "argument_count": len(copies)})
        # Returning None leaves the actual forward arguments untouched.

    def post_forward(_module, _arguments, result):
        if len(pending) != 1:
            raise RuntimeError("Missing paired pre-forward capture")
        record = pending.pop()
        outputs = directory / f"output_{record['ordinal']:05d}.pt"
        torch_module.save({"y_true": result[0].detach().clone().cpu(),
                           "logits": result[1].detach().clone().cpu()}, outputs)
        record.update(output_file=outputs.name, output_sha256=file_sha256(outputs))
        records.append(record)
        write_json(directory / "index.json", {
            "schema_version": 1, "captured_batches": len(records), "records": records,
            "input_contract": "Full immutable CPU-cloned actual positional arguments, before upstream in-place graph edits",
            "output_contract": "Native returned y_true and logits from that same forward; no additional forward",
            "pairing": "Use saved tensors, not seed-based reconstruction; native BN is in train mode and evolves in ordinal order",
        })
        # Returning None preserves the native result object.

    model.register_forward_pre_hook(pre_forward)
    model.register_forward_hook(post_forward)


def install_observers(trainer_class, *, output: Path, phase: str, upstream: Path) -> None:
    """Wrap init/eval for IO only; return native values without model changes."""
    original_init, original_eval = trainer_class.__init__, trainer_class.do_eval
    events = []

    def scalar(value):
        return value.detach().cpu().item() if hasattr(value, "detach") else float(value)

    def bn_state(model):
        return {name: {"training": module.training,
                       "num_batches_tracked": int(module.num_batches_tracked.item())}
                for name, module in model.named_modules()
                if getattr(module, "num_batches_tracked", None) is not None}

    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        write_json(output / "effective_params.json", self.parameter)
        write_json(output / "upstream_imports.json", verify_imports(upstream))
        (output / "model.txt").write_text(str(self.model) + "\n")
        write_json(output / "model_contract.json", {
            "training": self.model.training, "batch_norm": bn_state(self.model),
            "aggregations": {name: {"aggr_string": module.aggr,
                                    "aggr_module": type(module.aggr_module).__name__}
                             for name, module in self.model.named_modules()
                             if hasattr(module, "aggr_module")},
            "requires_grad": {name: parameter.requires_grad
                              for name, parameter in self.model.named_parameters()},
        })
        if phase in ("smoke", "train"):
            # state_dict()/torch.save do not sample RNG or mutate parameter/buffer values.
            torch = importlib.import_module("torch")
            torch.save({"model": self.model.state_dict()}, output / "initial_model.ckpt")
        elif phase == "eval":
            install_episode_capture(self.model, output=output,
                                    torch_module=importlib.import_module("torch"))

    def observed_eval(self, dataloader, *args, **kwargs):
        split = next((name for name in ("test", "val", "train_val", "train")
                      if getattr(self, name + "_dataloader", None) is dataloader), "unknown")
        before = bn_state(self.model)
        training = self.model.training
        started = time.time()
        result = original_eval(self, dataloader, *args, **kwargs)
        loss, accuracy, batch_std, auxiliary, ranks = result
        events.append({
            "index": len(events), "split": split, "model_training_before": training,
            "model_training_after": self.model.training, "batch_norm_before": before,
            "batch_norm_after": bn_state(self.model), "batches": len(dataloader),
            "episodes": len(dataloader) * self.batch_size,
            "nominal_query_predictions": len(dataloader) * self.batch_size * self.ways * self.parameter["n_query"],
            "loss": scalar(loss), "accuracy_fraction": scalar(accuracy),
            "native_batch_accuracy_std": scalar(batch_std), "auxiliary_loss": scalar(auxiliary),
            "ranks": ranks, "elapsed_seconds": time.time() - started,
        })
        write_json(output / "eval_metrics.json", {
            "schema_version": 1, "phase": phase, "events": events,
            "primary_eval": events[0] if phase == "eval" else None,
            "uncertainty_note": "Native std is across batches/episodes, not a training-seed confidence interval.",
        })
        return result

    trainer_class.__init__, trainer_class.do_eval = observed_init, observed_eval


def execute(plan: dict) -> None:
    upstream, root, output = map(Path, (plan["upstream"], plan["root"], plan["output"]))
    verification = verify_upstream(upstream)
    receipts = verify_receipts(root, plan["required_asset_receipts"])
    runtime = runtime_versions()
    checkpoint_hash = None
    if plan["checkpoint"]:
        checkpoint = Path(plan["checkpoint"])
        if not checkpoint.is_file() or checkpoint.is_symlink():
            raise ValueError(f"Missing regular nominated checkpoint: {checkpoint}")
        checkpoint_hash = file_sha256(checkpoint)
    # No data/model imports above this point. Refuse existing output even after a race.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    (output / "wandb").mkdir()
    plan.update(dry_run=False, upstream_verification=verification, asset_receipts=receipts,
                runtime=runtime, checkpoint_sha256=checkpoint_hash, started_unix=time.time())
    write_json(output / "protocol.json", plan)
    os.environ.update(plan["environment"])
    sys.dont_write_bytecode = True
    verify_imports(upstream)
    sys.path.insert(0, str(upstream))
    os.chdir(upstream)
    sys.argv = plan["command"][1:]
    try:
        trainer = importlib.import_module("experiments.trainer")
        dataloader_module = importlib.import_module("data.dataloader")
        write_json(output / "python_compatibility.json",
                   install_sampler_compat(dataloader_module.MulticlassTask))
        write_json(output / "upstream_imports.json", verify_imports(upstream))
        install_observers(trainer.TrainerFS, output=output, phase=plan["phase"], upstream=upstream)
        runpy.run_path(plan["command"][1], run_name="__main__")
    except BaseException as error:
        write_json(output / "execution_status.json", {
            "status": "failed", "error_type": type(error).__name__, "error": str(error),
            "finished_unix": time.time(),
        })
        raise
    else:
        write_json(output / "execution_status.json", {"status": "complete", "finished_unix": time.time()})


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("smoke", "train", "eval"))
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(2, 3))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(upstream=args.upstream, root=args.root, output=args.output,
                      phase=args.phase, gpu=args.gpu, seed=args.seed, checkpoint=args.checkpoint)
    plan["upstream_verification"] = verify_upstream(Path(plan["upstream"]))
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        execute(plan)


if __name__ == "__main__":
    main()
