"""Light CPU model checks; no dataset loading or training."""
import random
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from .run_native import capture_forward_state, install_episode_capture
from .paired_replay import load_record, replay_native, restore_forward_state


def test_snapshot_is_read_only_and_buffers_are_independent():
    model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.4))
    model.train()
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    initial = capture_forward_state(model, torch)
    snapshot = capture_forward_state(model, torch)
    assert torch.equal(initial["torch_rng"], torch.get_rng_state())
    assert initial["python_rng"] == random.getstate()
    assert np.array_equal(initial["numpy_rng"][1], np.random.get_state()[1])
    expected = model(inputs)
    assert snapshot["buffers"]["0.num_batches_tracked"].item() == 0
    assert model[0].num_batches_tracked.item() == 1
    with torch.no_grad():
        for name, buffer in model.named_buffers():
            buffer.copy_(snapshot["buffers"][name])
    torch.set_rng_state(snapshot["torch_rng"])
    actual = model(inputs)
    assert torch.equal(expected, actual)
    assert snapshot["training"] == {"": True, "0": True, "1": True}


def test_actual_capture_replay_and_failure_isolation():
    class EpisodeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(3)
            self.dropout = torch.nn.Dropout(.4)

        def forward(self, features, labels):
            features.add_(1)  # Mimic native in-place graph mutation.
            return labels, self.dropout(self.bn(features)), features

    model = EpisodeModel()
    inputs = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    labels = torch.eye(3)[torch.tensor([0, 1, 2, 0])]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        install_episode_capture(model, output=directory, torch_module=torch)
        with torch.no_grad():
            model(inputs.clone(), labels.clone())
        # Replay on a separate model loaded with the same parameters.
        replay = EpisodeModel()
        replay.load_state_dict(model.state_dict())
        replay.dropout.eval()  # Verify mixed caller modes are preserved.
        capture = directory / "episode_capture"
        record = json.loads((capture / "index.json").read_text())["records"][0]
        artifacts = load_record(capture, record)
        before = capture_forward_state(replay, torch)
        result = replay_native(replay, artifacts, "cpu")
        assert result["max_abs_error"] == 0
        assert torch.equal(artifacts["input"][0], inputs)
        assert torch.equal(before["torch_rng"], torch.get_rng_state())
        assert not replay.dropout.training
        assert replay.bn.num_batches_tracked.item() == 1
        artifacts["output"]["logits"][0, 0] += 1
        try:
            replay_native(replay, artifacts, "cpu")
        except ValueError as error:
            assert "logit parity" in str(error)
        else:
            raise AssertionError("Corrupted logits passed parity")
        assert torch.equal(before["torch_rng"], torch.get_rng_state())
        assert replay.bn.num_batches_tracked.item() == 1
        broken = dict(before, buffers={})
        try:
            restore_forward_state(replay, broken)
        except ValueError as error:
            assert "inventory" in str(error)
        else:
            raise AssertionError("Missing buffers accepted")


if __name__ == "__main__":
    test_snapshot_is_read_only_and_buffers_are_independent()
    test_actual_capture_replay_and_failure_isolation()
    print("Forward-state replay test passed")
