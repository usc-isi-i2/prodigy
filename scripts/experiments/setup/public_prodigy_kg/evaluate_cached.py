"""Replay the fixed public KG evaluation and gate quality before role tests.

Dry run by default. Loads only saved episodic tensors, never a dataset/whole
graph or sampler. The native architecture is constructed from pinned upstream
code; complete model state is loaded strictly. No optimizer is constructed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

if __package__:
    from . import run_native as native
else:
    import run_native as native


EPISODES = 500
REPLAY_ATOL = 1e-6
REPLAY_RTOL = 1e-5


def read_json(path):
    return json.loads(Path(path).read_text())


def require(mapping, expected, label):
    for key, value in expected.items():
        if mapping.get(key) != value:
            raise ValueError(f"{label}: expected {key}={value!r}, got {mapping.get(key)!r}")


def validate_bundle(native_eval: Path, train_run: Path, *, verify_hashes=True) -> dict:
    """Validate finished native capture/provenance before loading any pickle."""
    ep, tp = read_json(native_eval / "protocol.json"), read_json(train_run / "protocol.json")
    require(ep, {"phase": "eval", "dry_run": False, "dataset": "FB15K-237", "seed": 0,
                 "upstream_commit": native.UPSTREAM_COMMIT}, "native eval protocol")
    require(tp, {"phase": "train", "dry_run": False, "dataset": "Wiki", "seed": 0,
                 "upstream_commit": native.UPSTREAM_COMMIT}, "full train protocol")
    require(read_json(native_eval / "execution_status.json"), {"status": "complete"}, "native eval")
    params = read_json(native_eval / "effective_params.json")
    train_params = read_json(train_run / "effective_params.json")
    common = {"layers": "S2,UX,M2", "emb_dim": 256, "input_dim": 770, "dropout": 0,
              "text_features_dropout": 0, "ignore_label_embeddings": True,
              "zero_label_embeddings": False, "meta_gnn_pos_only": True,
              "has_final_back": False, "no_bn_encoder": False, "no_bn_metagraph": False,
              "skip_path": False, "zero_shot": False, "not_freeze_learned_label_embedding": False,
              "reset_after_layer": None, "n_shots": 3, "n_query": 4, "seed": 0}
    require(params, dict(common, dataset="FB15K-237", n_way=20, batch_size=1,
                         test_len_cap=500, train_cap=None, no_split_labels=True,
                         task_name="multiway_classification", eval_only=True,
                         augmentation="", augment_test=False, all_test=True), "eval parameters")
    require(train_params, dict(common, dataset="Wiki", n_way=15, batch_size=10,
                               dataset_len_cap=10010, epochs=1, task_name="cls_nm",
                               augmentation="ND0.5,NZ0.5", augment_test=True,
                               attr_regression_weight=1000), "training parameters")
    if params.get("label_set") != list(map(str, range(200))):
        raise ValueError("Native relation candidate IDs must be exactly 0..199")
    checkpoint = native.absolute_path(Path(ep["checkpoint"]), "checkpoint")
    expected_checkpoint = train_run / "state" / train_params["exp_name"] / "checkpoint" / native.NOMINATED_CHECKPOINT
    if checkpoint != expected_checkpoint.resolve():
        raise ValueError("Nominated checkpoint is not from the specified full training run")
    initial = train_run / "initial_model.ckpt"
    for path in (checkpoint, initial):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing regular model state: {path}")
    contract = read_json(native_eval / "model_contract.json")
    if contract.get("training") is not True or not contract.get("batch_norm"):
        raise ValueError("Missing native train-mode BN contract")
    if any(value.get("training") is not True for value in contract["batch_norm"].values()):
        raise ValueError("Native BN mode differs from the nominated train-mode protocol")
    native_metrics = read_json(native_eval / "eval_metrics.json")
    if len(native_metrics.get("events", [])) != 1:
        raise ValueError("Expected one native eval-only test stream")
    require(native_metrics["events"][0], {"split": "test", "batches": 500,
                                         "model_training_before": True}, "native eval metrics")
    index_path = native_eval / "episode_capture" / "index.json"
    index = read_json(index_path)
    records = index.get("records", [])
    if index.get("captured_batches") != EPISODES or len(records) != EPISODES:
        raise ValueError("Exactly 500 completed native captured batches are required")
    entries = []
    for ordinal, record in enumerate(records):
        require(record, {"ordinal": ordinal, "input_file": f"batch_{ordinal:05d}.pt",
                         "output_file": f"output_{ordinal:05d}.pt"}, "capture index")
        paths = {}
        for kind in ("input", "output"):
            path = index_path.parent / record[f"{kind}_file"]
            digest = record.get(f"{kind}_sha256", "")
            if (len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest)
                    or path.is_symlink() or not path.is_file()):
                raise ValueError(f"Invalid captured {kind}: {path}")
            if verify_hashes and native.file_sha256(path) != digest:
                raise ValueError(f"Captured {kind} SHA256 mismatch: {path}")
            paths[kind] = path
        entries.append(paths)
    checkpoint_sha = native.file_sha256(checkpoint)
    if checkpoint_sha != ep.get("checkpoint_sha256"):
        raise ValueError("Checkpoint differs from the checkpoint used by native evaluation")
    identity = {"upstream_commit": native.UPSTREAM_COMMIT,
                "native_protocol_sha256": native.file_sha256(native_eval / "protocol.json"),
                "native_effective_params_sha256": native.file_sha256(native_eval / "effective_params.json"),
                "train_protocol_sha256": native.file_sha256(train_run / "protocol.json"),
                "capture_index_sha256": native.file_sha256(index_path),
                "checkpoint_sha256": checkpoint_sha, "initial_model_sha256": native.file_sha256(initial)}
    return {"params": params, "entries": entries, "checkpoint": checkpoint,
            "initial": initial, "identity": identity, "native_metrics": native_metrics}


def validate_quality_gate(quality_run: Path, identity: dict) -> dict:
    if __package__:
        from . import decision
    else:
        import decision
    require(read_json(quality_run / "execution_status.json"), {"status": "complete"}, "quality execution")
    result = read_json(quality_run / "quality.json")
    if (result.get("identity") != identity or result.get("phase") != "quality"
            or result.get("completed_episodes") != EPISODES
            or result.get("replay", {}).get("passed") is not True
            or result.get("decision", {}).get("gate_passed") is not True):
        raise ValueError("Roles require the completed, matching quality AND replay gates")
    # A saved pass flag alone is not the evidence: recompute the frozen decision
    # from the actual paired episode vectors before any role forwards.
    try:
        recomputed = decision.decide_quality(
            result["native"]["episode_accuracy"], result["raw"]["episode_accuracy"],
            result["untrained"]["episode_accuracy"])
        replay = result["replay"]
        replay_records = replay["per_episode"]
        replay_complete = (replay["atol"] == REPLAY_ATOL and replay["rtol"] == REPLAY_RTOL
                           and len(replay_records) == EPISODES
                           and all(record.get("passed") is True for record in replay_records)
                           and result["native"]["replay"] == replay)
    except (KeyError, TypeError) as error:
        raise ValueError("Quality gate is missing its episode-level evidence") from error
    if not replay_complete or recomputed != result["decision"] or not recomputed["gate_passed"]:
        raise ValueError("Quality gate does not reconcile with its paired episode evidence")
    return result


def ensure_gpu_idle(gpu: int) -> None:
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
        "--format=csv,noheader,nounits"], text=True).strip()
    if output:
        raise RuntimeError(f"Selected physical GPU {gpu} already has compute processes: {output}")


def import_runtime(upstream: Path):
    """Remove foreign source search roots before importing native namespaces."""
    native.verify_imports(upstream)
    kept = []
    for entry in sys.path:
        path = Path(entry or os.getcwd()).resolve()
        if path != upstream and any((path / name).exists() for name in native.PROJECT_PACKAGES):
            continue
        kept.append(entry)
    sys.path[:] = [str(upstream), *kept]
    # Several upstream modules append the characters of a relative path to
    # sys.path, including '.'. Keep that native quirk inside the pinned clone.
    os.chdir(upstream)
    torch = importlib.import_module("torch")
    layers = importlib.import_module("experiments.layers")
    general = importlib.import_module("models.general_gnn")
    native.verify_imports(upstream)
    return torch, layers, general


def construct_model(params, state, device):
    """Exact model-only constructor from upstream TrainerFS; no BERT/data loader."""
    import torch
    from experiments.layers import get_module_list
    from models.general_gnn import SingleLayerGeneralGNN
    label_mlp = torch.nn.Linear(768, params["emb_dim"])
    text_dropout = torch.nn.Dropout(params["text_features_dropout"])
    layers = get_module_list(params["layers"], params["emb_dim"], edge_attr_dim=768,
                             input_dim=params["input_dim"], dropout=params["dropout"],
                             reset_after_layer=params["reset_after_layer"],
                             attention_mask_scheme=params["attention_mask_scheme"],
                             has_final_back=params["has_final_back"],
                             msg_pos_only=params["meta_gnn_pos_only"],
                             batch_norm_encoder=not params["no_bn_encoder"],
                             batch_norm_metagraph=not params["no_bn_metagraph"], gnn_use_relu=True)
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers), initial_label_mlp=label_mlp,
                                  params=dict(params), text_dropout=text_dropout).to(device)
    for parameter in model.learned_label_embedding.parameters():
        parameter.requires_grad = False
    model.load_state_dict(state, strict=True)
    if (len(model.layer_list) != 3 or len(model.layer_list[0].module_list) != 2
            or type(model.layer_list[1]).__name__ != "BgGraphToSupernodePropagatorCat"
            or model.layer_list[2].num_gnn_layers != 2
            or not isinstance(model.layer_list[2].gnn_non_linear, torch.nn.ReLU)):
        raise ValueError("Unexpected native S2,UX,M2 architecture")
    for layer in model.layer_list[0].module_list:
        if type(layer.aggr_module).__name__ not in ("SumAggregation", "AddAggregation"):
            raise ValueError("Native sum aggregation must remain intact")
    return model


def load_state(path, torch):
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {"model"}:
        raise ValueError("Expected native complete {'model': state_dict} checkpoint")
    return payload["model"]


def rng_snapshot(torch, numpy, seed=0):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return (random.getstate(), numpy.random.get_state(), torch.get_rng_state().clone(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(snapshot, torch, numpy):
    random.setstate(snapshot[0])
    numpy.random.set_state(snapshot[1])
    torch.set_rng_state(snapshot[2])
    if snapshot[3] is not None:
        torch.cuda.set_rng_state_all(snapshot[3])


def clone_arguments(arguments, device):
    return tuple(value.clone().to(device) for value in arguments)


def accuracy(y_true, logits):
    import torch
    if y_true.shape != logits.shape or logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Invalid native query outputs")
    return float((y_true.argmax(-1) == logits.argmax(-1)).double().mean().item())


def compare_native(y_true, logits, reference) -> dict:
    import torch
    y_true, logits = y_true.detach().cpu(), logits.detach().cpu()
    expected_y, expected = reference["y_true"], reference["logits"]
    shape_ok = y_true.shape == expected_y.shape and logits.shape == expected.shape
    if not shape_ok or logits.dtype != torch.float32 or expected.dtype != torch.float32:
        return {"passed": False, "shape_or_dtype_mismatch": True}
    labels_equal = torch.equal(y_true, expected_y)
    argmax_equal = torch.equal(logits.argmax(-1), expected.argmax(-1))
    close = torch.allclose(logits, expected, atol=REPLAY_ATOL, rtol=REPLAY_RTOL)
    error = (logits.double() - expected.double()).abs()
    return {"passed": bool(labels_equal and argmax_equal and close), "labels_equal": labels_equal,
            "argmax_equal": argmax_equal, "allclose": close,
            "max_abs_error": float(error.max()),
            "max_tolerance_ratio": float((error / (REPLAY_ATOL + REPLAY_RTOL * expected.double().abs())).max())}


def episode_iterator(entries, torch):
    for entry in entries:
        yield torch.load(entry["input"], map_location="cpu"), torch.load(entry["output"], map_location="cpu")


STATE_STAGES = {
    "pre_meta_points": "layer_list[1].proj output: S2+UX point vectors before either metagraph layer",
    "final_points": "layer_list[2] output point rows: after both native M layers, before Identity final_input_mlp",
    "final_queries": "final_points rows selected by is_query in ascending point order",
    "final_labels": "layer_list[2] output label rows: before Identity final_label_mlp and cosine normalization",
    "ordering": "Point rows match input y_true_matrix; label rows match episode-local class columns 0..ways-1",
}


def final_state_payload(model, captured, arguments, info, y_true, logits, *, ordinal, arm):
    """Immutable CPU observations from the existing forward; no extra inference."""
    import torch
    if not isinstance(model.final_input_mlp, torch.nn.Identity) or not isinstance(model.final_label_mlp, torch.nn.Identity):
        raise ValueError("Final-state capture requires the native Identity final heads")
    meta, pre_meta = captured["meta"], captured["descriptor"]
    y, is_query, local_labels = arguments[2], info["is_query"], info["labels"]
    points, ways = y.shape
    if (meta.ndim != 2 or meta.shape[0] != points + ways
            or pre_meta.shape != (points, meta.shape[1])
            or is_query.dtype != torch.bool or is_query.shape != (points,)
            or local_labels.shape != (points,) or info["ways"] != ways
            or not torch.equal(local_labels.cpu(), y.argmax(-1).cpu())):
        raise ValueError("Final-state point/label shape or local-class order mismatch")
    expected_y = y[is_query.to(y.device)].detach().cpu()
    if (not torch.equal(y_true.detach().cpu(), expected_y)
            or logits.shape != expected_y.shape or model.logit_scale.ndim != 0):
        raise ValueError("Final-state query row order or decoder shape mismatch")
    def cpu(value):
        return value.detach().cpu().clone()
    # Every tensor owns its CPU storage: callbacks cannot alter inputs, model
    # parameters, captured forward outputs, or the query-drift reference.
    return {"schema_version": 1, "ordinal": ordinal, "arm": arm,
            "stages": dict(STATE_STAGES), "pre_meta_points": cpu(pre_meta),
            "final_points": cpu(meta[:points]), "final_labels": cpu(meta[points:]),
            "final_queries": cpu(meta[:points][is_query.to(meta.device)]),
            "is_query": cpu(is_query), "local_labels": cpu(local_labels),
            "point_y_true": cpu(y), "query_y_true": cpu(y_true),
            "query_point_indices": cpu(torch.where(is_query)[0]),
            "class_indices": torch.arange(ways, dtype=torch.long, device="cpu"),
            "query_logits": cpu(logits), "logit_scale": cpu(model.logit_scale),
            "cosine_eps": model.cos.eps}


def final_state_sink(output: Path, arm: str, torch_module):
    """Persist private per-episode states and an ordered SHA256 index."""
    if arm not in ("native", "support", "query"):
        raise ValueError("Final-state files are limited to the three planned role arms")
    directory = output / "final_states" / arm
    directory.mkdir(parents=True, exist_ok=False)
    records = []

    def save(ordinal, payload):
        if ordinal != len(records) or payload["ordinal"] != ordinal or payload["arm"] != arm:
            raise ValueError("Final-state sink received an out-of-order/wrong-arm episode")
        if any(value.device.type != "cpu" or value.requires_grad for value in payload.values()
               if torch_module.is_tensor(value)):
            raise ValueError("Only detached CPU final states may be serialized")
        path = directory / f"episode_{ordinal:05d}.pt"
        torch_module.save(payload, path)
        records.append({"ordinal": ordinal, "file": path.name,
                        "sha256": native.file_sha256(path),
                        "points": payload["final_points"].shape[0],
                        "queries": payload["final_queries"].shape[0],
                        "ways": payload["final_labels"].shape[0],
                        "embedding_dim": payload["final_points"].shape[1]})
        native.write_json(directory / "index.json", {
            "schema_version": 1, "arm": arm, "stages": STATE_STAGES,
            "saved_episodes": len(records), "records": records,
            "scope": "Private observations of the already-planned forward stream, not additional interventions or metrics",
        })

    return save


def model_stream(model, state, episodes, *, device, helpers, rng, numpy,
                 arm="native", collect_queries=False, query_reference=None, progress=None, state_sink=None):
    """One ordered stream; tests may supply small synthetic streams, CLI cannot."""
    import torch
    model.load_state_dict(state, strict=True)
    model.train(True)
    restore_rng(rng, torch, numpy)
    captured = {}
    hooks = []
    if state_sink is not None and (arm not in ("native", "support", "query")
                                   or not (collect_queries or query_reference is not None)):
        raise ValueError("State capture is only attached to an existing role-query capture stream")
    if arm == "untrained" or state_sink is not None:
        hooks.append(model.layer_list[1].proj.register_forward_hook(
            lambda _m, _a, out: captured.update(descriptor=out.detach().clone())))
    if collect_queries or query_reference is not None:
        hooks.append(model.layer_list[2].register_forward_hook(
            lambda _m, _a, out: captured.update(meta=out.detach().clone())))
    scores, replays, diagnostics, query_states, drift = [], [], [], [], []
    try:
        with torch.no_grad():
            for ordinal, (arguments, reference) in enumerate(episodes):
                info = helpers.validate_episode(arguments)
                captured.clear()
                if arm in ("support", "query"):
                    arguments, diagnostic = helpers.suppress_background(arguments, arm)
                else:
                    diagnostic = {}
                fresh = clone_arguments(arguments, device)
                y_true, logits, _ = model(*fresh)
                if arm == "native":
                    replays.append(compare_native(y_true, logits, reference))
                elif arm == "untrained":
                    # fresh[0].x was mutated by the full native forward. Read labels/
                    # schema from the untouched cached arguments, never that graph.
                    y_true, logits, diagnostic = helpers.cosine_prototype_logits(captured["descriptor"], arguments)
                if not torch.equal(y_true.detach().cpu(), reference["y_true"]):
                    raise ValueError(f"Query label/order mismatch at episode {ordinal}")
                scores.append(accuracy(y_true, logits))
                diagnostics.append(diagnostic)
                if collect_queries or query_reference is not None:
                    queries = captured["meta"][:len(info["is_query"])][info["is_query"].to(device)].cpu()
                    if collect_queries:
                        query_states.append(queries)
                    if query_reference is not None:
                        delta = queries.double() - query_reference[ordinal].double()
                        drift.append({"mean_query_l2_change": float(delta.norm(dim=1).mean()),
                                      "max_abs_query_change": float(delta.abs().max())})
                if state_sink is not None:
                    state_sink(ordinal, final_state_payload(model, captured, arguments, info, y_true, logits,
                                                           ordinal=ordinal, arm=arm))
                if progress:
                    progress(ordinal + 1)
    finally:
        for hook in hooks:
            hook.remove()
    changed = [name for name, parameter in model.named_parameters()
               if not torch.equal(parameter.detach().cpu(), state[name])]
    if changed:
        raise RuntimeError(f"Weights changed during forward-only stream: {changed}")
    bn = {name: int(module.num_batches_tracked.item()) for name, module in model.named_modules()
          if getattr(module, "num_batches_tracked", None) is not None}
    result = {"episode_accuracy": scores, "diagnostics": diagnostics, "query_drift": drift,
              "weights_unchanged": True, "batch_norm_final_counts": bn}
    if replays:
        result["replay"] = {"passed": all(item["passed"] for item in replays),
                            "atol": REPLAY_ATOL, "rtol": REPLAY_RTOL, "per_episode": replays,
                            "max_abs_error": max((item["max_abs_error"] for item in replays
                                                  if "max_abs_error" in item), default=None)}
    return result, query_states


def raw_stream(entries, torch, helpers, device):
    scores, diagnostics = [], []
    for arguments, reference in episode_iterator(entries, torch):
        fresh = clone_arguments(arguments, device)
        descriptor = helpers.raw_descriptors(fresh)
        y_true, logits, diagnostic = helpers.cosine_prototype_logits(descriptor, fresh)
        if not torch.equal(y_true.detach().cpu(), reference["y_true"]):
            raise ValueError("Raw prototype query ordering differs from native capture")
        scores.append(accuracy(y_true, logits))
        diagnostics.append(diagnostic)
    return {"episode_accuracy": scores, "diagnostics": diagnostics}


def execute(plan):
    upstream, output = Path(plan["upstream"]), Path(plan["output"])
    native.verify_upstream(upstream)
    bundle = validate_bundle(Path(plan["native_eval"]), Path(plan["train_run"]))
    quality = None
    if plan["phase"] == "roles":
        quality = validate_quality_gate(Path(plan["quality_run"]), bundle["identity"])
    runtime = native.runtime_versions()
    ensure_gpu_idle(plan["gpu"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    plan.update(dry_run=False, identity=bundle["identity"], runtime=runtime, started_unix=time.time())
    native.write_json(output / "protocol.json", plan)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(plan["gpu"]), WANDB_MODE="offline", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    try:
        torch, _, _ = import_runtime(upstream)
        numpy = importlib.import_module("numpy")
        if __package__:
            helpers = importlib.import_module(".cached_tensors", __package__)
            decision = importlib.import_module(".decision", __package__)
        else:
            helpers = importlib.import_module("cached_tensors")
            decision = importlib.import_module("decision")
        native.write_json(output / "upstream_imports.json", native.verify_imports(upstream))
        torch.set_num_threads(4)
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda:0")
        checkpoint = load_state(bundle["checkpoint"], torch)
        initial = load_state(bundle["initial"], torch)
        model = construct_model(bundle["params"], checkpoint, device)
        # Validate full initial checkpoint strictly before any quality/role forwards.
        model.load_state_dict(initial, strict=True)
        rng = rng_snapshot(torch, numpy)
        def run(arm, state, **options):
            sink = final_state_sink(output, arm, torch) if plan["phase"] == "roles" else None
            return model_stream(model, state, episode_iterator(bundle["entries"], torch),
                                device=device, helpers=helpers, rng=rng, numpy=numpy, arm=arm,
                                state_sink=sink,
                                progress=lambda completed: native.write_json(output / "progress.json",
                                    {"arm": arm, "completed_episodes": completed, "total": EPISODES}), **options)
        baseline, query_states = run("native", checkpoint, collect_queries=plan["phase"] == "roles")
        native.write_json(output / "native_replay.json", baseline)
        if not baseline["replay"]["passed"]:
            raise RuntimeError("Native replay failed the predeclared logit/argmax gate; no further arms run")
        result = {"phase": plan["phase"], "identity": bundle["identity"], "completed_episodes": EPISODES,
                  "replay": baseline["replay"], "native": baseline}
        if plan["phase"] == "quality":
            raw = raw_stream(bundle["entries"], torch, helpers, device)
            untrained, _ = run("untrained", initial)
            result.update(raw=raw, untrained=untrained,
                          decision=decision.decide_quality(baseline["episode_accuracy"], raw["episode_accuracy"],
                                                           untrained["episode_accuracy"]))
        else:
            support, _ = run("support", checkpoint, query_reference=query_states)
            query, _ = run("query", checkpoint, query_reference=query_states)
            result.update(support=support, query=query, quality_gate=quality["decision"],
                          decision=decision.decide_roles(baseline["episode_accuracy"], support["episode_accuracy"],
                                                         query["episode_accuracy"]))
            result["final_states"] = {}
            for arm in ("native", "support", "query"):
                index_path = output / "final_states" / arm / "index.json"
                result["final_states"][arm] = {"index": str(index_path),
                    "sha256": native.file_sha256(index_path),
                    "saved_episodes": read_json(index_path)["saved_episodes"]}
        native.write_json(output / (plan["phase"] + ".json"), result)
    except BaseException as error:
        native.write_json(output / "execution_status.json", {"status": "failed", "error": str(error),
                          "error_type": type(error).__name__, "finished_unix": time.time()})
        raise
    else:
        native.write_json(output / "execution_status.json", {"status": "complete", "finished_unix": time.time()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("quality", "roles"), required=True)
    for name in ("upstream", "native-eval", "train-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--quality-run", type=Path)
    parser.add_argument("--gpu", type=int, choices=(2, 3), default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if (args.phase == "roles") != (args.quality_run is not None):
        parser.error("--quality-run is required only for roles")
    paths = {name: native.absolute_path(getattr(args, name), name)
             for name in ("upstream", "native_eval", "train_run", "output")}
    if args.quality_run is not None:
        paths["quality_run"] = native.absolute_path(args.quality_run, "quality_run")
    if paths["output"].exists():
        raise FileExistsError("Output must be new")
    for name, path in paths.items():
        if name != "output" and (paths["output"].is_relative_to(path) or path.is_relative_to(paths["output"])):
            raise ValueError("Output must be separate from all input directories")
    plan = {"schema_version": 1, "phase": args.phase, "gpu": args.gpu,
            **{name: str(path) for name, path in paths.items()}, "dry_run": True,
            "upstream_verification": native.verify_upstream(paths["upstream"]),
            "replay_gate": {"atol": REPLAY_ATOL, "rtol": REPLAY_RTOL, "same_argmax": True,
                            "exact_query_labels": True, "scope": "All 500 saved native logits; no tolerance adjustment"},
            "protocol": {"episodes": EPISODES, "model_mode": "train, including BN", "seed": 0,
                         "stream_state": "Strict checkpoint/initial-state reload including BN; identical replay-start Python/NumPy/Torch/CUDA RNG state",
                         "rng_note": "Native initial RNG state was not captured. Allowed forward dropout is zero; exact native replay is independently required.",
                         "query_drift_stage": "Final MetaGNN output query rows, before identity final projection/cosine",
                         "private_final_state_capture": ({
                             "arms": ["native", "support", "query"], "stages": STATE_STAGES,
                             "files": "final_states/<arm>/episode_<ordinal>.pt and SHA256 index.json",
                             "contents": "Detached immutable CPU pre-M points, final-M points/queries/labels, local row/class metadata, native logits and logit_scale",
                             "scope": "Observations of the same planned role forwards; no added forwards, arms, metrics, or gates",
                         } if args.phase == "roles" else None),
                         "quality_gate": "Both native-minus-raw/untrained paired95% bootstrap lower bounds >0",
                         "role_gate": "Requires complete matching quality gate; support-minus-native lower>0 and query-minus-native upper<0",
                         "no_resampling": True, "no_other_checkpoints_or_modes": True}}
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        execute(plan)


if __name__ == "__main__":
    main()
