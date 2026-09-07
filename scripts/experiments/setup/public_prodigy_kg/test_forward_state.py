"""Light CPU model checks; no dataset loading or training."""
import random

import numpy as np
import torch

from .run_native import capture_forward_state


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


if __name__ == "__main__":
    test_snapshot_is_read_only_and_buffers_are_independent()
    print("Forward-state replay test passed")
