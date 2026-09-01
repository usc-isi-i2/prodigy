#!/usr/bin/env python3
"""Train one balanced leave-one-target-out MT/NM/NM+MT mixture model."""

from __future__ import annotations

import argparse
import copy
import itertools
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))

from experiments.params import get_params  # noqa: E402
from experiments.run_single_experiment import load_dataset, seed_everything  # noqa: E402
from experiments.trainer import TrainerFS  # noqa: E402
from experiments.task_families import (  # noqa: E402
    TASK_FAMILY_TO_ID,
    resolve_task_family,
)


SOURCES = {
    "covid_political": ("/dataMeR1/phil/data/covid_political/graphs", "retweet_graph.pt"),
    "election2020": ("/dataMeR1/phil/data/election2020/graphs", "retweet_graph.pt"),
    "facebook_page_reference": (
        "/dataMeR1/phil/data/facebook_page_reference/graphs", "page_reference_graph.pt"
    ),
    "twibot20": ("/dataMeR1/phil/data/twibot20/graphs", "retweet_graph.pt"),
    "ukr_rus_suspended": (
        "/dataMeR1/phil/data/ukr_rus_suspended/graphs", "retweet_graph.pt"
    ),
}


class ScheduledLoader:
    """Yield homogeneous minibatches from a fixed balanced donor/objective cycle."""

    def __init__(self, loaders, schedule, steps):
        self.loaders = loaders
        self.schedule = schedule
        self.steps = steps

    def __len__(self):
        return self.steps

    def __iter__(self):
        iterators = {key: iter(loader) for key, loader in self.loaders.items()}
        for key in itertools.islice(itertools.cycle(self.schedule), self.steps):
            try:
                batch = next(iterators[key])
            except StopIteration:
                iterators[key] = iter(self.loaders[key])
                batch = next(iterators[key])
            donor, task = key
            batch[0].task_family_id = torch.tensor(
                TASK_FAMILY_TO_ID[resolve_task_family(task, donor)], dtype=torch.long
            )
            yield batch


def donor_params(base, donor, task):
    params = copy.deepcopy(base)
    root, filename = SOURCES[donor]
    params["dataset"] = donor
    params["root"] = root
    params["graph_filename"] = filename
    params["task_name"] = task
    params["n_way"] = 30 if task == "neighbor_matching" else 2
    return params


def build_train_loader(trainer, dataset, params):
    saved = trainer.parameter
    trainer.parameter = params
    try:
        train, _, _, _ = trainer._build_dataloaders(dataset, params["dataset"])
    finally:
        trainer.parameter = saved
    return train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["MT", "NM", "NM_MT"])
    parser.add_argument("--heldout", required=True, choices=list(SOURCES))
    parser.add_argument("--device", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--task-embedding-dim", type=int, default=0)
    parser.add_argument("--task-embedding-dropout", type=float, default=0.25)
    parser.add_argument("--task-embedding-fusion", choices=["add", "film"], default="add")
    args = parser.parse_args()

    donors = [source for source in SOURCES if source != args.heldout]
    first_root, first_file = SOURCES[donors[0]]
    budget = 12 if args.smoke else 900
    task_tag = (
        f"_task{args.task_embedding_dim}_{args.task_embedding_fusion}"
        if args.task_embedding_dim else ""
    )
    prefix = f"mtpilot_{args.arm}{task_tag}_heldout_{args.heldout}" + ("_smoke" if args.smoke else "")
    seen_families = {"neighbor_matching"}
    seen_families.update(resolve_task_family("classification", donor) for donor in donors)
    base = get_params([
        "--config", str(HERE / "configs" / f"{args.arm}.yaml"),
        "--dataset", donors[0], "--root", first_root, "--graph_filename", first_file,
        "--device", str(args.device), "--dataset_len_cap", str(budget),
        "--checkpoint_step", str(budget), "--prefix", prefix,
        "--task_embedding_dim", str(args.task_embedding_dim),
        "--task_embedding_dropout", str(args.task_embedding_dropout),
        "--task_embedding_fusion", args.task_embedding_fusion,
        "--task_embedding_seen_families", ",".join(sorted(seen_families)),
    ])
    seed_everything(base)
    datasets = {donor: load_dataset(donor_params(base, donor, "neighbor_matching")) for donor in donors}
    trainer = TrainerFS(datasets[donors[0]], base)

    objectives = {
        "MT": ["classification"],
        "NM": ["neighbor_matching"],
        "NM_MT": ["classification", "neighbor_matching"],
    }[args.arm]
    loaders = {}
    for donor in donors:
        for task in objectives:
            params = donor_params(base, donor, task)
            loaders[(donor, task)] = build_train_loader(trainer, datasets[donor], params)
    schedule = [(donor, task) for donor in donors for task in objectives]
    trainer.train_dataloader = ScheduledLoader(loaders, schedule, budget)
    trainer.train()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
