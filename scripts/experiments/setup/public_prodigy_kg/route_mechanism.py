"""Exploratory route localization; not a replication of the first-layer test."""
from contextlib import contextmanager

import torch

from .episode_mechanism import trace_forward, decode_pair
from .paired_replay import replay_native
from .role_interventions import context_donor, query_rows


@contextmanager
def metagraph_input(module, donor=None, support=None):
    """Observe/replace actual pre-M rows, including the native keyword call."""
    records = []

    def hook(_module, args, kwargs):
        if records:
            raise ValueError("Metagraph invoked multiple times")
        x = args[0] if args else kwargs["x"]
        if donor is not None:
            if (x.shape != donor.shape or x.device != donor.device or
                    x.dtype != donor.dtype or support.dtype != torch.bool or
                    support.shape != x.shape[:1]):
                raise ValueError("Pre-metagraph donor contract mismatch")
            x = x.clone()
            x[support] = donor[support]
        records.append(x.detach().clone())
        if args:
            return (x, *args[1:]), kwargs
        return args, {**kwargs, "x": x}

    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield records
        if len(records) != 1:
            raise ValueError("Metagraph not invoked")
    finally:
        handle.remove()


def run_route_episode(model, artifacts, device, *, atol=0.0, rtol=0.0):
    """Same saved state, full support donor, layer-specific support-only swaps.

    The donor second-layer projection comes from the FULL support-context-removal
    forward, not a first-layer swap. Consequently this localizes sufficient
    changed messages, not an additive path-mediated fraction. Pre-M replacement
    includes support residual information but excludes donor query/label rows.
    """
    parity = replay_native(model, artifacts, device, atol=atol, rtol=rtol)
    modules = [m for m in model.layer_list if hasattr(m, "gnn_layers")]
    if len(modules) != 1 or len(modules[0].gnn_layers) != 2:
        raise ValueError("Expected one two-layer native metagraph")
    meta = modules[0]
    arguments = tuple(x.clone().to(device) for x in artifacts["input"])
    state = artifacts["state"]
    queries = query_rows(arguments)
    support = torch.zeros(queries.numel() + arguments[1].shape[0],
                          dtype=torch.bool, device=device)
    support[:queries.numel()] = ~queries

    def trace(args, layer=0, swap=None, prem=None):
        with metagraph_input(meta, prem, support) as inputs:
            result = trace_forward(model, args, state,
                                   meta.gnn_layers[layer].mlp_kqv, swap)
        result["pre_meta"] = inputs[0]
        return result

    native = trace(arguments)
    if not torch.equal(native["y_true"].cpu(), artifacts["output"]["y_true"]):
        raise ValueError("Native instrumentation changed labels")
    if not torch.allclose(native["logits"].cpu(), parity["logits"], atol=atol, rtol=rtol):
        raise ValueError("Native route instrumentation failed parity")
    support_args, receipt = context_donor(arguments, "support")
    donors = [trace(support_args, layer=i) for i in range(2)]
    if not torch.allclose(donors[0]["logits"], donors[1]["logits"], atol=atol, rtol=rtol):
        raise ValueError("Layer-specific donor replays disagree")
    arms = {"native": native, "support_context_removed": donors[0],
            "pre_meta_support": trace(arguments, prem=donors[0]["pre_meta"])}
    for layer in range(2):
        # Null transplant verifies each selected route independently.
        native_layer = trace(arguments, layer=layer)
        null = trace(arguments, layer=layer,
                     swap=(native_layer["projection"], support, "joint"))
        if not torch.allclose(null["logits"], native["logits"], atol=atol, rtol=rtol):
            raise ValueError(f"Layer {layer + 1} null transplant failed parity")
        for component in ("keys", "values", "joint"):
            arms[f"layer{layer + 1}_{component}"] = trace(
                arguments, layer=layer,
                swap=(donors[layer]["projection"], support, component))
    null_pre = trace(arguments, prem=native["pre_meta"])
    if not torch.allclose(null_pre["logits"], native["logits"], atol=atol, rtol=rtol):
        raise ValueError("Pre-metagraph null transplant failed parity")
    results = {}
    for name, arm in arms.items():
        if not torch.equal(arm["y_true"], native["y_true"]) or not torch.equal(arm["edges"], native["edges"]):
            raise ValueError(f"Labels/class mapping changed in {name}")
        reproduced = decode_pair(model, arguments, arm["input_x"], arm["label_x"])
        if not torch.allclose(reproduced, arm["logits"], atol=atol, rtol=rtol):
            raise ValueError(f"Decoder reproduction failed in {name}")
        results[name] = {
            "logits": arm["logits"].cpu(),
            "reference_only_logits": decode_pair(model, arguments, native["input_x"], arm["label_x"]).cpu(),
            "query_only_logits": decode_pair(model, arguments, arm["input_x"], native["label_x"]).cpu(),
            "query_max_abs_change": float((arm["input_x"][queries] - native["input_x"][queries]).abs().max()),
            "reference_max_abs_change": float((arm["label_x"] - native["label_x"]).abs().max()),
        }
    return {"y_true": native["y_true"].cpu(), "arms": results,
            "parity": {k: parity[k] for k in ("atol", "rtol", "max_abs_error")},
            "context_receipts": [receipt], "exploratory": True,
            "donor_definition": "full support context removal, captured separately at each nominated layer",
            "pre_meta_non_support_max_abs_change": float((arms["pre_meta_support"]["pre_meta"][~support] - native["pre_meta"][~support]).abs().max())}
