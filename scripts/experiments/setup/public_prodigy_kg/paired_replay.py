"""Replay immutable native episodes; checkpoint/model construction stays native.

Only load trusted local capture artifacts. A caller must load the checkpoint
whose hash is recorded in protocol.json before using these helpers.
"""
from contextlib import contextmanager
import random

import numpy as np
import torch

from .run_native import capture_forward_state, file_sha256


def restore_forward_state(model, state):
    """Validate inventories before restoring buffers, individual modes and RNG."""
    buffers = dict(model.named_buffers())
    modules = dict(model.named_modules())
    if buffers.keys() != state["buffers"].keys() or modules.keys() != state["training"].keys():
        raise ValueError("Replay model inventory differs from captured model")
    for name, buffer in buffers.items():
        saved = state["buffers"][name]
        if buffer.shape != saved.shape or buffer.dtype != saved.dtype:
            raise ValueError(f"Replay buffer contract differs: {name}")
    if state["cuda_rng"] is not None:
        if not torch.cuda.is_initialized() or len(state["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA replay requires the captured visible-device inventory")
    with torch.no_grad():
        for name, buffer in buffers.items():
            buffer.copy_(state["buffers"][name])
    # train() recursively overwrites child modes, so restore each flag directly.
    for name, module in modules.items():
        module.training = state["training"][name]
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng"])


@contextmanager
def isolated_forward_state(model, state):
    """Keep interventions from advancing the caller's native state stream."""
    caller = capture_forward_state(model, torch)
    try:
        restore_forward_state(model, state)
        yield
    finally:
        restore_forward_state(model, caller)


def load_record(directory, record):
    artifacts = {}
    for kind in ("input", "output", "state"):
        path = directory / record[f"{kind}_file"]
        if path.resolve().parent != directory.resolve():
            raise ValueError("Capture artifact must be directly inside capture directory")
        if file_sha256(path) != record[f"{kind}_sha256"]:
            raise ValueError(f"Capture hash mismatch: {kind}")
        artifacts[kind] = torch.load(path, map_location="cpu")
    return artifacts


def replay_native(model, artifacts, device, *, atol=0.0, rtol=0.0):
    """Require native parity before accepting any intervention on an episode.

    Exact comparison is the default. Any relaxed GPU tolerance must be explicitly
    selected and reported by the caller, not silently absorbed here.
    """
    if atol < 0 or rtol < 0:
        raise ValueError("Parity tolerances must be nonnegative")
    arguments = tuple(value.clone().to(device) for value in artifacts["input"])
    with isolated_forward_state(model, artifacts["state"]), torch.no_grad():
        labels, logits, *_ = model(*arguments)
    labels, logits = labels.detach().cpu(), logits.detach().cpu()
    expected = artifacts["output"]
    if not torch.equal(labels, expected["y_true"]):
        raise ValueError("Native replay labels differ")
    reference = expected["logits"]
    if logits.shape != reference.shape or not torch.isfinite(logits).all():
        raise ValueError("Native replay logits have invalid shape or values")
    error = float((logits - reference).abs().max())
    if not torch.allclose(logits, reference, atol=atol, rtol=rtol):
        raise ValueError(f"Native replay failed logit parity: max_abs_error={error}")
    return {"max_abs_error": error, "atol": atol, "rtol": rtol,
            "y_true": labels, "logits": logits}
