"""Frozen support-centered readout on a new public test episode stream."""
import argparse
import json
from functools import partial
from pathlib import Path

from . import run_native as native


def install_online(model, output, device, atol):
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
        if pending:
            raise ValueError("Unexpected nested native forward")
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
                raise ValueError("Expected single-task public episodes")
            group = result["tasks"][0]
            x = result["geometry"]["pre"]
            labels = artifact["input"][2]
            s, q = group["support_rows"], group["query_rows"]
            # Frozen inductive rule: no query distribution or labels used in fit.
            centered = predict(x[s], labels[s], x[q], "support")
            truth = labels[q].argmax(1).numpy()
            full = result["native"]["logits"].numpy()
            scores = {"uncentered": score_task(truth, full, result["ridge_logits"].numpy()),
                      "support_centered": score_task(truth, full, centered.numpy())}
            result["support_centered_logits"] = centered
        finally:
            active = False
        path = directory / f"episode_{len(records):05d}.pt"
        torch.save({"native_capture": artifact, "result": result, "scores": scores}, path)
        records.append({"ordinal": len(records), "file": path.name, "sha256": native.file_sha256(path)})
        native.write_json(directory / "index.json", {"records": records,
                          "completion_authority": "parent execution_status.json"})
        print(f"Replication episode {len(records)}/500 saved", flush=True)

    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)


def install_observers(trainer_class, *, output, phase, upstream, offset, atol):
    if phase != "eval":
        raise ValueError("Replication is evaluation only")
    native.install_observers(trainer_class, output=output, phase="readout_replication", upstream=upstream)
    original = trainer_class.__init__

    def observed(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.parameter["batch_size"] != 1 or self.parameter["test_len_cap"] != 500:
            raise ValueError("Unexpected public episode protocol")
        seed = sum(map(ord, "test")) + offset
        self.test_dataloader.batch_sampler.rng.seed(seed)
        native.write_json(output / "stream.json", {"split": "test", "native_seed": sum(map(ord, "test")),
            "offset": offset, "effective_seed": seed, "same_data_pools": True,
            "scope": "new sampler stream, not a new training seed or independent entity pool"})
        install_online(self.model, output, self.device, atol)

    trainer_class.__init__ = observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "root", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = native.build_plan(upstream=args.upstream, root=args.root, output=args.output,
        checkpoint=args.checkpoint, gpu=args.gpu, seed=0, phase="eval")
    plan["replication"] = {"episode_seed_offset": 100003, "parity_atol": 1e-4,
        "parity_rtol": 0, "episodes": 500, "rule": "subtract support mean from both roles, row-L2, ridge lambda=1, no intercept, scale=1",
        "nominated_before_outcomes": True, "query_labels_or_query_mean_used_in_fit": False,
        "controls": ["native", "uncentered pre-metagraph ridge"],
        "scope": "episode-stream replication only; fixed checkpoint and public target"}
    plan["native_protocol_preserved"]["scope"] = "Same native evaluation except explicit test sampler seed offset and state-preserving paired readouts"
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        native.execute(plan, observer_installer=partial(install_observers, offset=100003, atol=1e-4))


if __name__ == "__main__":
    main()
