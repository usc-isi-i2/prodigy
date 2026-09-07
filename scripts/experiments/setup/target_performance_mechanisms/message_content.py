"""Scoped inference controls for the actual background aggregation operator.

Do not change production training code or reinterpret these controls as trained
mean-aggregation models. All changes are restored when the context exits.
"""
from contextlib import contextmanager

import torch
from torch_geometric.nn.aggr import MeanAggregation

from .role_topology import assert_background_attributes_unused


def aggregation_audit(model):
    assert_background_attributes_unused(model)
    rows = []
    for i, layer in enumerate(model.layer_list[0].module_list):
        values = layer.aggregate(torch.ones(3, 1), torch.tensor([0, 1, 1]), dim_size=3)
        rows.append({"layer": i, "configured_aggr": layer.aggr,
                     "aggregation_module": type(layer.aggr_module).__name__,
                     "unit_message_probe": values.flatten().tolist()})
    return rows


@contextmanager
def message_control(model, graph, condition):
    assert_background_attributes_unused(model)
    if condition not in ("intact", "actual_mean", "no_message_bias", "bias_only", "mean_message", "zero_messages"):
        raise ValueError("unknown message-content control")
    if model.training or len(model.layer_list[0].module_list) != 1:
        raise ValueError("single frozen S layer required")
    layer = model.layer_list[0].module_list[0]
    original_aggregate = layer.aggr_module
    handles = []
    try:
        if condition == "actual_mean":
            layer.aggr_module = MeanAggregation().eval()
        elif condition != "intact":
            def intervene(module, args, values):
                if condition == "zero_messages":
                    return torch.zeros_like(values)
                if condition == "no_message_bias":
                    return values - module.bias
                if condition == "bias_only":
                    return module.bias.expand_as(values)
                real = graph.global_node_ids >= 0
                group = graph.batch
                count = torch.bincount(group[real], minlength=len(graph.ptr)-1).clamp_min(1)
                total = values.new_zeros((len(count), values.shape[1]))
                total.index_add_(0, group[real], values[real])
                return (total / count[:, None])[group]
            handles.append(layer.lin_x.register_forward_hook(intervene))
        yield
    finally:
        for handle in handles:
            handle.remove()
        layer.aggr_module = original_aggregate
