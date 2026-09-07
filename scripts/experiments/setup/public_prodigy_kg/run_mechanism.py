"""Native evaluation stream plus isolated, paired interventions; dry-run first."""
import argparse
import json
from functools import partial
from pathlib import Path

from . import run_native as native


def install_mechanism_observers(trainer_class, *, output, phase, upstream, atol=0.0, rtol=0.0):
    if phase != "eval":
        raise ValueError("Mechanism analysis is evaluation only")
    # Retain native metric/provenance observers, but replace disk capture with our
    # guarded capture; replay forwards must not enter the native episode stream.
    native.install_observers(trainer_class, output=output, phase="mechanism", upstream=upstream)
    original_init = trainer_class.__init__

    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        install_online_experiment(self.model, output, self.device, atol=atol, rtol=rtol)

    trainer_class.__init__ = observed_init


def install_online_experiment(model, output, device, *, atol=0.0, rtol=0.0):
    import torch
    from .episode_mechanism import run_episode

    directory = output / "paired_episodes"
    directory.mkdir(exist_ok=False)
    active = False
    pending = []
    records = []

    def before(_module, arguments):
        if active:
            return
        if pending:
            raise ValueError("Unexpected nested native forward")
        pending.append({"input": tuple(value.clone().cpu() for value in arguments),
                        "state": native.capture_forward_state(model, torch)})

    def after(_module, _arguments, returned):
        nonlocal active
        if active:
            return
        artifacts = pending.pop()
        artifacts["output"] = {"y_true": returned[0].detach().clone().cpu(),
                               "logits": returned[1].detach().clone().cpu()}
        active = True
        try:
            result = run_episode(model, artifacts, device, atol=atol, rtol=rtol)
        except Exception as error:
            # Preserve the exact failing input/state for diagnosis, not as an
            # accepted intervention episode or a completed stream record.
            torch.save(artifacts, directory / "failed_native_capture.pt")
            native.write_json(directory / "failure.json", {
                "ordinal": len(records), "error": repr(error),
                "accepted_intervention_result": False})
            raise
        finally:
            active = False
        filename = directory / f"episode_{len(records):05d}.pt"
        torch.save({"native_capture": artifacts, "mechanism": result}, filename)
        records.append({"ordinal": len(records), "file": filename.name,
                        "sha256": native.file_sha256(filename)})
        native.write_json(directory / "index.json", {"schema_version": 1, "records": records,
                          "native_stream_unchanged": "Paired forwards restore post-native buffers, module modes and RNG; original result returned unchanged",
                          "completion_authority": "Parent execution_status.json; index is written incrementally"})
        print(f"Paired mechanism episode {len(records)} saved", flush=True)

    model.register_forward_pre_hook(before)
    model.register_forward_hook(after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("upstream", "root", "output", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(2, 3), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parity-atol", type=float, default=0.0)
    parser.add_argument("--parity-rtol", type=float, default=0.0)
    args = parser.parse_args()
    plan = native.build_plan(upstream=args.upstream, root=args.root, output=args.output,
                             checkpoint=args.checkpoint, gpu=args.gpu, seed=args.seed, phase="eval")
    if args.parity_atol < 0 or args.parity_rtol < 0:
        raise ValueError("Parity tolerances must be nonnegative")
    plan["paired_mechanism"] = {"primary_layer": 0, "parity_atol": args.parity_atol, "parity_rtol": args.parity_rtol,
        "arms": ["native", "support_context_removed", "query_context_removed", "keys", "values", "joint"],
        "decoder_crosses": ["native_queries_changed_references", "changed_queries_native_references"],
        "target_tuning": False, "episode_count": 500}
    plan["native_protocol_preserved"]["scope"] = "Native stream preserved; isolated paired interventions added after each native forward"
    plan["upstream_verification"] = native.verify_upstream(args.upstream)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        native.execute(plan, observer_installer=partial(install_mechanism_observers,
                       atol=args.parity_atol, rtol=args.parity_rtol))


if __name__ == "__main__":
    main()
