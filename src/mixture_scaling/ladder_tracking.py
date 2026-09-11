"""Offline-first W&B tracking with an inspectable JSONL mirror."""
import json
from collections import defaultdict
from contextlib import contextmanager


class LossWindow:
    def __init__(self):
        self.sums = defaultdict(float)
        self.weights = defaultdict(int)

    def add(self, source, loss, examples):
        self.sums[source] += loss * examples
        self.weights[source] += examples

    def flush(self):
        if not self.weights:
            raise ValueError("empty loss window")
        metrics = {"train/loss": sum(self.sums.values()) / sum(self.weights.values()),
                   "train/examples": sum(self.weights.values())}
        for source, count in self.weights.items():
            metrics[f"train/source/{source}/loss"] = self.sums[source] / count
            metrics[f"train/source/{source}/examples"] = count
        self.sums.clear()
        self.weights.clear()
        return metrics


@contextmanager
def tracked_run(run_dir, metadata, args):
    import wandb
    # Explicit mode prevents wandb.init's online default on direct Python launches.
    with wandb.init(project=args.wandb_project, mode=args.wandb_mode,
                    group=args.wandb_group or run_dir.parent.parent.name,
                    name=metadata["run_id"], dir=str(run_dir),
                    config={**metadata, "n_sources": len(metadata["sources"]),
                            "log_interval": args.log_interval}) as run:
        run.define_metric("optimizer_step")
        run.define_metric("train/*", step_metric="optimizer_step")
        run.define_metric("validation/*", step_metric="optimizer_step")
        receipt = {"id": run.id, "mode": args.wandb_mode, "directory": run.dir,
                   "project": args.wandb_project}
        (run_dir / "wandb_run.json").write_text(json.dumps(receipt, indent=2) + "\n")
        with (run_dir / "metrics.jsonl").open("w") as handle:
            def log(step, metrics):
                row = {"optimizer_step": step, **metrics}
                handle.write(json.dumps(row) + "\n")
                handle.flush()
                run.log(row)
            yield run, log
