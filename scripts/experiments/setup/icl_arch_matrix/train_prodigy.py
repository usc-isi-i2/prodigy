#!/usr/bin/env python3
"""Run PRODIGY after resetting episode RNG post model initialization."""

from __future__ import annotations

import torch

from experiments.params import get_params
from experiments.run_single_experiment import load_dataset, seed_everything
from experiments.trainer import TrainerFS
from scripts.experiments.setup.icl_arch_matrix.common_protocol import reset_episode_rng


def main() -> int:
    torch.set_num_threads(4)
    params = get_params()
    seed_everything(params)
    dataset = load_dataset(params)
    trainer = TrainerFS(dataset, params)
    reset_episode_rng()
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
