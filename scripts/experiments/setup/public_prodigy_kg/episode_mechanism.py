"""Paired public KG mechanism experiment on one captured native episode."""
from contextlib import contextmanager, nullcontext

import torch

from .paired_replay import isolated_forward_state, replay_native
from .role_interventions import context_donor, query_rows, transplant_projection


@contextmanager
def capture_decoder(model):
    """Observe the actual decoder inputs without changing its returned logits."""
    original = model.decode
    had_override = "decode" in model.__dict__
    previous = model.__dict__.get("decode")
    records = []

    def observed(input_x, label_x, edges, edgelist_bipartite=False):
        if edgelist_bipartite:
            raise ValueError("Expected native non-bipartite-index decoder")
        if records:
            raise ValueError("Multiple decoder calls in one episode")
        records.append({"input_x": input_x.detach().clone(),
                        "label_x": label_x.detach().clone(),
                        "edges": edges.detach().clone()})
        return original(input_x, label_x, edges, edgelist_bipartite=False)

    model.decode = observed
    try:
        yield records
        if len(records) != 1:
            raise ValueError("Native decoder was not invoked")
    finally:
        if had_override:
            model.decode = previous
        else:
            del model.decode


def trace_forward(model, arguments, state, projection, transplant=None):
    projection_outputs = []

    def observe(_module, _inputs, output):
        projection_outputs.append(output.detach().clone())

    intervention = (transplant_projection(projection, *transplant)
                    if transplant is not None else nullcontext())
    with isolated_forward_state(model, state), torch.no_grad(), intervention:
        # Registered after the transplant, so captures the actual consumed output.
        handle = projection.register_forward_hook(observe)
        try:
            with capture_decoder(model) as decoder:
                labels, logits, *_ = model(*(value.clone() for value in arguments))
        finally:
            handle.remove()
    if len(projection_outputs) != 1:
        raise ValueError("Expected exactly one first-layer projection")
    return {"y_true": labels.detach().clone(), "logits": logits.detach().clone(),
            "projection": projection_outputs[0], **decoder[0]}


def decode_pair(model, arguments, input_x, label_x):
    queries = query_rows(arguments)
    with torch.no_grad():
        return model.decode(input_x, label_x, arguments[3],
                            edgelist_bipartite=False).reshape(arguments[2].shape)[queries]


def run_episode(model, artifacts, device, *, atol=0.0, rtol=0.0):
    """No fitting or target selection. Fail native parity before computing arms.

    Uses the first layer of the single native MetaGNN block. The model must be
    built by the pinned runtime and have the nominated checkpoint loaded.
    """
    parity = replay_native(model, artifacts, device, atol=atol, rtol=rtol)
    meta = [layer for layer in model.layer_list if hasattr(layer, "gnn_layers")]
    if len(meta) != 1 or len(meta[0].gnn_layers) != 2:
        raise ValueError("Expected the nominated two-layer native metagraph")
    projection = meta[0].gnn_layers[0].mlp_kqv
    arguments = tuple(value.clone().to(device) for value in artifacts["input"])
    state = artifacts["state"]
    native = trace_forward(model, arguments, state, projection)
    if not torch.allclose(native["logits"].cpu(), parity["logits"], atol=atol, rtol=rtol):
        raise ValueError("Instrumentation changed native logits")
    queries = query_rows(arguments)
    support = torch.zeros(native["projection"].shape[0], dtype=torch.bool, device=device)
    if support.numel() != queries.numel() + arguments[1].shape[0]:
        raise ValueError("Metagraph projection rows do not match examples plus labels")
    support[:queries.numel()] = ~queries
    support_args, support_receipt = context_donor(arguments, "support")
    query_args, query_receipt = context_donor(arguments, "query")
    arms = {"native": native,
            "support_context_removed": trace_forward(model, support_args, state, projection),
            "query_context_removed": trace_forward(model, query_args, state, projection)}
    donor = arms["support_context_removed"]["projection"]
    for component in ("keys", "values", "joint"):
        arms[component] = trace_forward(model, arguments, state, projection,
                                        (donor, support, component))
    results = {}
    for name, arm in arms.items():
        if not torch.equal(arm["y_true"], native["y_true"]):
            raise ValueError(f"Episode labels changed in {name}")
        if not torch.equal(arm["edges"], native["edges"]):
            raise ValueError(f"Decoder class mapping changed in {name}")
        reproduced = decode_pair(model, arguments, arm["input_x"], arm["label_x"])
        if not torch.allclose(reproduced, arm["logits"], atol=atol, rtol=rtol):
            raise ValueError(f"Final representations do not reproduce logits in {name}")
        # Crossed decoder interventions: native queries + changed references,
        # and changed queries + native references. Neither is assumed additive.
        reference_only = decode_pair(model, arguments, native["input_x"], arm["label_x"])
        query_only = decode_pair(model, arguments, arm["input_x"], native["label_x"])
        results[name] = {
            "logits": arm["logits"].cpu(), "reference_only_logits": reference_only.cpu(),
            "query_only_logits": query_only.cpu(),
            "query_max_abs_change": float((arm["input_x"][queries] - native["input_x"][queries]).abs().max()),
            "reference_max_abs_change": float((arm["label_x"] - native["label_x"]).abs().max()),
        }
    return {"y_true": native["y_true"].cpu(), "arms": results,
            "parity": {key: parity[key] for key in ("atol", "rtol", "max_abs_error")},
            "context_receipts": [support_receipt, query_receipt],
            "primary_projection": "first metagraph block, first layer, mlp_kqv"}
