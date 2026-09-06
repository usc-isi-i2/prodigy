import unittest

import torch
from torch_geometric.data import Data, Batch

from experiments.layers import get_module_list
from models.general_gnn import SingleLayerGeneralGNN
from .replay import batch_hash, bn_mode, clone_batch, episode_probe, intervene, trace_stages


def fixture():
    torch.manual_seed(21)
    graphs = []
    for i in range(8):
        graphs.append(Data(x=torch.randn(4, 8), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                           edge_attr=torch.zeros(4, 1), supernode=torch.tensor([3]),
                           edge_index_supernode=torch.tensor([[0], [3]]),
                           global_node_ids=torch.tensor([i * 3, i * 3 + 1, i * 3 + 2, -1])))
    graph = Batch.from_data_list(graphs)
    graph.task_id_per_sample = torch.arange(2).repeat_interleave(4)
    graph.task_label_map = torch.tensor([[0, 1], [1, 0]])
    labels = torch.tensor([0, 1, 0, 1] * 2)
    query = torch.tensor([False, False, True, True] * 2)
    edge_index = torch.stack((torch.arange(8).repeat_interleave(2),
                             8 + graph.task_id_per_sample.repeat_interleave(2) * 2 + torch.arange(2).repeat(8)))
    onehot = torch.nn.functional.one_hot(labels, 2).float()
    edge_attr = torch.stack((query.repeat_interleave(2).float(),
                            ((onehot * 2 - 1) * ~query[:, None]).reshape(-1)), 1)
    batch = [graph, torch.zeros(4, 8), onehot, edge_index, edge_attr,
             query.repeat_interleave(2), torch.empty(0), torch.empty(0), torch.empty(0)]
    params = {"emb_dim": 8, "zero_shot": False, "skip_path": False,
              "ignore_label_embeddings": False, "zero_label_embeddings": False}
    layers = get_module_list("S,U,M", 8, 1, 8, 0, None, None, False, False)
    model = SingleLayerGeneralGNN(torch.nn.ModuleList(layers), params=params).eval()
    return model, batch


class ReplayTest(unittest.TestCase):
    def test_trace_exact_parity_and_restore(self):
        model, batch = fixture()
        digest = batch_hash(batch)
        with torch.no_grad():
            _, reference, _ = model(*clone_batch(batch))
            with trace_stages(model, batch[0]) as traces:
                _, result, _ = model(*clone_batch(batch))
        torch.testing.assert_close(reference, result, rtol=0, atol=0)
        self.assertEqual(set(traces), {"S0_conv_center", "S0_pool", "U1_pre_meta", "M2_post_meta", "final_input"})
        self.assertTrue(all(x.shape == (8, 8) for x in traces.values()))
        self.assertEqual(batch_hash(batch), digest)
        self.assertNotIn("forward", model.layer_list[1].__dict__)
        with self.assertRaises(RuntimeError):
            with trace_stages(model, batch[0]):
                raise RuntimeError("intentional")
        self.assertNotIn("forward", model.layer_list[1].__dict__)

    def test_probes_ignore_query_labels(self):
        _, batch = fixture()
        x = torch.randn(8, 8)
        changed = clone_batch(batch)
        query = batch[5].reshape(-1, 2)[:, 0]
        changed[2][query] = changed[2][query].flip(1)
        for method in ("ridge", "prototype"):
            torch.testing.assert_close(episode_probe(x, batch, method), episode_probe(x, changed, method))

    def test_feature_shuffle_invariants(self):
        _, batch = fixture()
        changed = clone_batch(batch)
        intervene(changed, "shuffle_context", 16)
        centers = batch[0].ptr[:-1]
        torch.testing.assert_close(batch[0].x[centers], changed[0].x[centers])
        torch.testing.assert_close(batch[0].edge_index, changed[0].edge_index)
        tasks = batch[0].task_id_per_sample[batch[0].batch]
        for task in (0, 1):
            torch.testing.assert_close(batch[0].x[tasks == task].sort(dim=0).values,
                                       changed[0].x[tasks == task].sort(dim=0).values)
        self.assertNotEqual(batch_hash(batch), batch_hash(changed))

    def test_bn_diagnostic_does_not_modify_buffers(self):
        model, batch = fixture()
        state = {k: v.clone() for k, v in model.state_dict().items()}
        with torch.no_grad(), bn_mode(model, "bn_batch_all"):
            model(*clone_batch(batch))
        for k, v in model.state_dict().items():
            torch.testing.assert_close(state[k], v, rtol=0, atol=0)
        self.assertTrue(all(not m.training for m in model.modules()))


if __name__ == "__main__":
    unittest.main()
