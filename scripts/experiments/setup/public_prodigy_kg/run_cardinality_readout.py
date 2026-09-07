"""Fixed 128-episode FB15K-237 15/20-way sensitivity test; dry run by default.

Changing ways bundles class count, total examples, and native train-mode BN
context. Equal sampler seeds do not imply matched examples across cardinalities.
"""
import argparse
import json
import shlex
from functools import partial
from pathlib import Path

from . import run_native as native

EPISODES = 128
OFFSET = 300007


def build_plan(*, ways, **kwargs):
    if ways not in (15, 20):
        raise ValueError("Only the nominated 15/20-way comparison is supported")
    plan = native.build_plan(phase="eval", seed=0, **kwargs)
    native.replace_option(plan["command"], "--n_way", str(ways))
    native.replace_option(plan["command"], "-test_cap", str(EPISODES))
    plan["shell_display_only"] = shlex.join(plan["command"])
    plan["cardinality"] = {
        "ways": ways, "episodes": EPISODES, "episode_seed_offset": OFFSET,
        "effective_sampler_seed": sum(map(ord, "test")) + OFFSET,
        "relation_candidate_count": 200, "shots": 3, "queries_per_class": 4,
        "parity_atol": 1e-4, "parity_rtol": 0,
        "rule": "subtract support mean from both roles, row-L2, ridge lambda=1, no intercept, scale=1",
        "query_labels_or_query_mean_used_in_fit": False,
        "scope": "Bundled episode-cardinality sensitivity: class count, total examples and BN context change; not an isolated label-count intervention or matched-example comparison.",
    }
    plan["native_protocol_preserved"]["eval"] = (
        f"FB{ways}way/3shot/4query, batch1, {EPISODES} episodes, "
        f"{EPISODES * ways * 4} queries, workers15; unchanged 200-relation pool")
    plan["native_protocol_preserved"]["scope"] = plan["cardinality"]["scope"]
    plan["deviations_from_released_command"].append(
        f"Nominated cardinality test: {ways} ways, {EPISODES} episodes, test sampler offset {OFFSET}; frozen paired readouts.")
    return plan


def install_online(model, output, device, atol, *, expected=EPISODES):
    import torch
    from .readout_episode import run_episode
    from .role_centering import predict
    from .score_readout import score_task

    directory = output / "paired_episodes"
    directory.mkdir(exist_ok=False)
    active, pending, records = False, [], []

    def before(_module, arguments):
        if active:
            return
        if pending or len(records) >= expected:
            raise ValueError("Nested or excess native forward")
        pending.append({"input": tuple(v.clone().cpu() for v in arguments),
                        "state": native.capture_forward_state(model, torch)})

    def after(_module, _arguments, returned):
        nonlocal active
        if active:
            return
        artifact = pending.pop()
        artifact["output"] = {"y_true": returned[0].detach().clone().cpu(),
                              "logits": returned[1].detach().clone().cpu()}
        active = True
        try:
            result = run_episode(model, artifact, device, atol=atol, compare_post=True)
            if len(result["tasks"]) != 1:
                raise ValueError("Expected one task per public episode")
            group, x, labels = result["tasks"][0], result["geometry"]["pre"], artifact["input"][2]
            s, q = group["support_rows"], group["query_rows"]
            centered = predict(x[s], labels[s], x[q], "support")
            truth, full = labels[q].argmax(1).numpy(), result["native"]["logits"].numpy()
            scores = {"uncentered": score_task(truth, full, result["ridge_logits"].numpy()),
                      "support_centered": score_task(truth, full, centered.numpy())}
            result["support_centered_logits"] = centered
        finally:
            active = False
        path = directory / f"episode_{len(records):05d}.pt"
        torch.save({"native_capture": artifact, "result": result, "scores": scores}, path)
        records.append({"ordinal": len(records), "file": path.name, "sha256": native.file_sha256(path)})
        native.write_json(directory / "index.json", {"records": records, "expected_episodes": expected,
            "completion_authority": "parent execution_status.json plus exact expected record count"})
        print(f"Cardinality episode {len(records)}/{expected} saved", flush=True)

    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)
    return records


def install_observers(trainer_class, *, output, phase, upstream, ways):
    if phase != "eval":
        raise ValueError("Cardinality test is evaluation only")
    native.install_observers(trainer_class, output=output, phase="cardinality_readout", upstream=upstream)
    original = trainer_class.__init__

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if (self.parameter["batch_size"] != 1 or self.parameter["test_len_cap"] != EPISODES
                or self.parameter["n_way"] != ways):
            raise ValueError("Unexpected cardinality protocol")
        seed = sum(map(ord, "test")) + OFFSET
        self.test_dataloader.batch_sampler.rng.seed(seed)
        native.write_json(output / "stream.json", {"split": "test", "native_seed": 448,
            "offset": OFFSET, "effective_seed": seed, "same_data_pools": True,
            "ways": ways, "episodes": EPISODES,
            "scope": "Same seed across ways does not imply matched examples; cardinality changes total episode size and BN context."})
        install_online(self.model, output, self.device, 1e-4)

    trainer_class.__init__ = observed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "root", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--ways", type=int, choices=(15, 20), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(upstream=args.upstream, root=args.root, output=args.output,
                      checkpoint=args.checkpoint, gpu=args.gpu, ways=args.ways)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        native.execute(plan, observer_installer=partial(install_observers, ways=args.ways))
        try:
            index = json.loads((args.output / "paired_episodes/index.json").read_text())
            if len(index["records"]) != EPISODES:
                raise ValueError("Incomplete cardinality capture")
        except Exception as error:
            native.write_json(args.output / "execution_status.json", {"status": "failed",
                "reason": "Cardinality capture verification failed", "error": repr(error)})
            raise


if __name__ == "__main__":
    main()
