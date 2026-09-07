import json
from pathlib import Path
import tempfile
import unittest
import torch

from .test_readout_episode import fixture
from .run_readout_replication import install_online
from .run_native import capture_forward_state
from .paired_replay import restore_forward_state


class ReplicationTests(unittest.TestCase):
    def test_online_replay_preserves_native_stream(self):
        class Meta(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gnn_layers = torch.nn.ModuleList([])
                self.bn = torch.nn.BatchNorm1d(2)
            def forward(self, *, x, start_right):
                return self.bn(x)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer_list = torch.nn.ModuleList([Meta()])
            def decode(self, input_x, label_x, edges, edgelist_bipartite=False):
                return input_x
            def forward(self, x, unused, labels, edges, attrs, query):
                z = self.layer_list[0](x=x, start_right=len(x))
                q = query.reshape(len(x), 2)[:, 0]
                return labels[q], self.decode(z, unused, edges)[q], x
        model = Model()
        x, y, edges, query = fixture(2, 1)
        args = (x, torch.zeros(1), y, edges, torch.zeros(1), query)
        initial = capture_forward_state(model, torch)
        with torch.no_grad():
            expected = [model(*args)[1].clone() for _ in range(2)]
        terminal = capture_forward_state(model, torch)
        restore_forward_state(model, initial)
        with tempfile.TemporaryDirectory() as directory:
            install_online(model, Path(directory), "cpu", 0)
            with torch.no_grad():
                actual = [model(*args)[1].clone() for _ in range(2)]
            records = json.loads((Path(directory) / "paired_episodes/index.json").read_text())["records"]
            self.assertEqual(len(records), 2)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b)
        for name, value in model.named_buffers():
            self.assertTrue(torch.equal(value, terminal["buffers"][name]))
        self.assertTrue(torch.equal(torch.get_rng_state(), terminal["torch_rng"]))


if __name__ == "__main__":
    unittest.main()
