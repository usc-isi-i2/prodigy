from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate_fixed_grid import (  # noqa: E402
    ReplayLoader,
    assert_cpu_batches,
    load_checkpoint_strict,
)


class FakeGraph:
    def __init__(self, x: torch.Tensor):
        self.x = x

    def clone(self):
        return FakeGraph(self.x.clone())

    def to_dict(self):
        return {"x": self.x}


def test_replay_clones_every_mutable_tensor():
    cached = [(FakeGraph(torch.tensor([1.0, 2.0])), torch.tensor([3.0]))]
    assert_cpu_batches(cached)

    first = next(iter(ReplayLoader(cached)))
    first[0].x.add_(10)
    first[1].add_(20)

    assert torch.equal(cached[0][0].x, torch.tensor([1.0, 2.0]))
    assert torch.equal(cached[0][1], torch.tensor([3.0]))
    second = next(iter(ReplayLoader(cached)))
    assert torch.equal(second[0].x, cached[0][0].x)
    assert torch.equal(second[1], cached[0][1])
    assert second[0].x.data_ptr() != cached[0][0].x.data_ptr()
    assert second[1].data_ptr() != cached[0][1].data_ptr()


def test_strict_checkpoint_reload_replaces_prior_model():
    with tempfile.TemporaryDirectory() as directory:
        expected = torch.nn.Linear(2, 1)
        with torch.no_grad():
            expected.weight.fill_(7.0)
            expected.bias.fill_(3.0)
        checkpoint = Path(directory) / "state_dict_2500.ckpt"
        torch.save({"model": expected.state_dict()}, checkpoint)

        actual = torch.nn.Linear(2, 1)
        trainer = SimpleNamespace(device="cpu", all_saveable_modules={"model": actual})
        load_checkpoint_strict(trainer, checkpoint)
        assert torch.equal(actual.weight, expected.weight)
        assert torch.equal(actual.bias, expected.bias)


def test_strict_checkpoint_reload_rejects_missing_module():
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "state_dict_2500.ckpt"
        torch.save({}, checkpoint)
        trainer = SimpleNamespace(
            device="cpu", all_saveable_modules={"model": torch.nn.Linear(2, 1)}
        )
        try:
            load_checkpoint_strict(trainer, checkpoint)
        except KeyError as error:
            assert "missing modules" in str(error)
        else:
            raise AssertionError("missing checkpoint module must fail")
