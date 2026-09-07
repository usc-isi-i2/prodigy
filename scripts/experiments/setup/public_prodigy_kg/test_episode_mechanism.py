"""Tiny fixture using the real pinned decoder and two-layer MetaGNN.

Pass the pinned upstream checkout as the sole argument. Background and pooling
are deliberately tiny test components, not a reproduction of the full recipe.
"""
import sys
import json
import tempfile
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data

from .episode_mechanism import run_episode
from .run_native import capture_forward_state
from .paired_replay import restore_forward_state
from .run_mechanism import install_online_experiment


def test_pinned_metagraph(upstream, device="cpu"):
    sys.path.insert(0, str(Path(upstream).resolve()))
    from models.general_gnn import SingleLayerGeneralGNN
    from models.metaGNN import MetaGNN
    from models.layer_classes import BackgroundGNNLayer, SupernodeAggrLayer

    class Background(torch.nn.Module, BackgroundGNNLayer):
        def __init__(self):
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(8)

        def forward(self, original, x, edges, *unused):
            changed = x.clone()
            changed.index_add_(0, edges[1], x[edges[0]])
            return self.bn(changed)

    class Pool(torch.nn.Module, SupernodeAggrLayer):
        def forward(self, x, edges, supernodes, batch):
            pooled = torch.zeros_like(x)
            pooled.index_add_(0, edges[1], x[edges[0]])
            return pooled[supernodes]

    torch.manual_seed(17)
    model = SingleLayerGeneralGNN(torch.nn.ModuleList([
        Background(), Pool(), MetaGNN(2, 8, heads=2, n_layers=2, msg_pos_only=True)]),
        params={"emb_dim": 8, "ignore_label_embeddings": True,
                "zero_label_embeddings": False, "skip_path": False, "zero_shot": False})
    graphs = [Data(x=torch.randn(4, 8),
                   edge_index=torch.tensor([[0, 2, 1], [2, 1, 2]]),
                   edge_attr=torch.randn(3, 2), supernode=torch.tensor([3]),
                   edge_index_supernode=torch.tensor([[0, 1], [3, 3]]))
              for _ in range(4)]
    labels = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
    query_mask = torch.tensor([False, False, True, True, False, False, True, True])
    edge_index = torch.stack((torch.arange(4).repeat_interleave(2), torch.tensor([4, 5] * 4)))
    edge_attr = torch.stack((query_mask.float(), (2 * labels.flatten() - 1) * ~query_mask), dim=1)
    args = (Batch.from_data_list(graphs), torch.randn(2, 8), labels, edge_index, edge_attr, query_mask)
    model.to(device)
    args = tuple(value.to(device) for value in args)
    state = capture_forward_state(model, torch)
    with torch.no_grad():
        y, logits, _ = model(*(value.clone() for value in args))
    artifacts = {"input": tuple(value.clone().cpu() for value in args), "state": state,
                 "output": {"y_true": y.clone().cpu(), "logits": logits.clone().cpu()}}
    before = capture_forward_state(model, torch)
    result = run_episode(model, artifacts, device)
    assert result["parity"]["max_abs_error"] == 0
    assert set(result["arms"]) == {"native", "support_context_removed", "query_context_removed", "keys", "values", "joint"}
    for arm in result["arms"].values():
        for field in ("logits", "reference_only_logits", "query_only_logits"):
            assert arm[field].shape == (2, 2)
            assert torch.isfinite(arm[field]).all()
    assert result["arms"]["native"]["query_max_abs_change"] == 0
    assert torch.equal(result["arms"]["native"]["reference_only_logits"], logits.cpu())
    assert result["arms"]["support_context_removed"]["query_max_abs_change"] > 0
    assert torch.equal(before["torch_rng"], torch.get_rng_state())
    for name, value in model.named_buffers():
        assert torch.equal(value.cpu(), before["buffers"][name])
    assert "decode" not in model.__dict__
    assert not model.layer_list[2].gnn_layers[0].mlp_kqv._forward_hooks
    stream_state = capture_forward_state(model, torch)
    with torch.no_grad():
        expected_stream = [model(*(value.clone() for value in args))[1].clone() for _ in range(2)]
    expected_state = capture_forward_state(model, torch)
    restore_forward_state(model, stream_state)
    with tempfile.TemporaryDirectory() as temporary:
        install_online_experiment(model, Path(temporary), device)
        with torch.no_grad():
            actual_stream = [model(*(value.clone() for value in args))[1].clone() for _ in range(2)]
        assert all(torch.equal(a, b) for a, b in zip(expected_stream, actual_stream))
        index = json.loads((Path(temporary) / "paired_episodes/index.json").read_text())
        assert len(index["records"]) == 2  # Recursive replay must not add episodes.
    for name, value in model.named_buffers():
        assert torch.equal(value.cpu(), expected_state["buffers"][name])
    assert torch.equal(torch.get_rng_state(), expected_state["torch_rng"])
    print("Pinned decoder/metagraph episode mechanism fixture passed")


if __name__ == "__main__":
    test_pinned_metagraph(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "cpu")
