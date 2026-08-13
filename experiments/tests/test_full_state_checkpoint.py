from types import SimpleNamespace
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from data.dataloader import BatchSampler, ParamSampler
from experiments.trainer import TrainerFS


class _Task:
    def __init__(self):
        self.scheduled_episode = 0

    def sample(self, num_label, num_member, num_shot, num_query, rng):
        del num_member, num_shot, num_query
        self.scheduled_episode += 1
        return {rng.randrange(1000): [rng.randrange(1000)] for _ in range(num_label)}


def _trainer(tmp_path, workers=0):
    trainer = TrainerFS.__new__(TrainerFS)
    trainer.parameter = {"workers": workers}
    trainer.device = torch.device("cpu")
    trainer.steps = 10
    trainer.ckpt_dir = str(tmp_path)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.all_saveable_modules = {"model": trainer.model}
    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=0.01)
    sampler = BatchSampler(
        10,
        _Task(),
        ParamSampler(batch_size=1, n_way=1, n_shot=1, n_query=1, n_aug=1),
        seed=17,
    )
    trainer.train_dataloader = SimpleNamespace(batch_sampler=sampler)
    trainer.resume_step = 0
    return trainer


def _one_optimizer_step(trainer):
    trainer.optimizer.zero_grad()
    trainer.model(torch.ones(1, 2)).sum().backward()
    trainer.optimizer.step()


class FullStateCheckpointTest(unittest.TestCase):
    def test_restores_optimizer_rng_and_episode_stream(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            trainer = _trainer(tmp_path)
            _one_optimizer_step(trainer)
            trainer.train_dataloader.batch_sampler.sample()
            trainer.save_checkpoint(1)

            weights_path = tmp_path / "state_dict_1.ckpt"
            full_path = tmp_path / "training_state_1.ckpt"
            self.assertEqual(
                set(torch.load(weights_path, weights_only=False)), {"model"}
            )
            full = torch.load(full_path, weights_only=False)
            self.assertEqual(full["_training_checkpoint"]["completed_steps"], 1)
            self.assertTrue(full["_training_checkpoint"]["optimizer"]["state"])

            expected_python = random.random()
            expected_numpy = np.random.random()
            expected_torch = torch.rand(1)
            expected_episode = trainer.train_dataloader.batch_sampler.sample()

            with torch.no_grad():
                trainer.model.weight.add_(100)
            trainer.optimizer.state.clear()
            trainer.train_dataloader.batch_sampler.sample()

            trainer.load_training_checkpoint(str(full_path))
            self.assertEqual(trainer.resume_step, 1)
            self.assertTrue(trainer.optimizer.state)
            # The trainer restores this immediately after DataLoader iterator
            # construction, which otherwise consumes a global Torch RNG draw.
            trainer._restore_rng_state(trainer._deferred_resume_rng_state)
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(np.random.random(), expected_numpy)
            self.assertTrue(torch.equal(torch.rand(1), expected_torch))
            self.assertEqual(
                trainer.train_dataloader.batch_sampler.sample(), expected_episode
            )

    def test_weights_only_is_not_an_exact_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            trainer = _trainer(tmp_path)
            trainer.save_checkpoint(0)
            with self.assertRaisesRegex(ValueError, "weights-only"):
                trainer.load_training_checkpoint(str(tmp_path / "state_dict_0.ckpt"))

    def test_rejects_multiprocess_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            trainer = _trainer(tmp_path, workers=2)
            trainer.save_checkpoint(0)
            with self.assertRaisesRegex(ValueError, "workers 0"):
                trainer.load_training_checkpoint(
                    str(tmp_path / "training_state_0.ckpt")
                )
