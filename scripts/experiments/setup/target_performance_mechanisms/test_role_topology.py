import unittest

import torch
from torch_geometric.data import Data, Batch

from .test_replay import fixture
from .replay import batch_hash, clone_batch
from .role_context import query_mask
from .role_topology import swap_edges, topology_batch, encode, factorial_predictions, stable_seed, assert_background_attributes_unused


def cycle_fixture():
    model, batch = fixture()
    graphs = []
    for gi in range(8):
        forward = [(i, (i + 1) % 9) for i in range(9)]
        edges = forward + [(v, u) for u, v in forward]
        graphs.append(Data(x=torch.randn(10, 8), edge_index=torch.tensor(edges).T.contiguous(),
                           edge_attr=torch.zeros(18, 1), supernode=torch.tensor([9]),
                           edge_index_supernode=torch.tensor([[0], [9]]),
                           global_node_ids=torch.tensor([gi * 9 + i for i in range(9)] + [-1])))
    g = Batch.from_data_list(graphs)
    g.task_id_per_sample = batch[0].task_id_per_sample
    g.task_label_map = batch[0].task_label_map
    batch[0] = g
    return model, batch


class RoleTopologyTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_directed_degree_and_self_loops_exact(self):
        edges = torch.tensor([(i, (i+1) % 12) for i in range(12)] + [(0, 0)]).T
        changed, audit = swap_edges(edges, seed=7)
        self.assertFalse(audit["reciprocal"])
        self.assertGreater(audit["changed_edges"], 0)
        for axis in (0, 1):
            torch.testing.assert_close(torch.bincount(edges[axis]), torch.bincount(changed[axis]), rtol=0, atol=0)
        self.assertEqual(int((changed[0] == changed[1]).sum()), 1)

    def test_reciprocal_degree_exact_and_deterministic(self):
        _, b = cycle_fixture()
        edges = b[0].edge_index[:, :18]
        rng = torch.get_rng_state().clone()
        first, audit = swap_edges(edges, seed=19)
        second, again = swap_edges(edges, seed=19)
        self.assertEqual(audit, again)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertGreater(audit["changed_edges"], 0)
        pairs = set(map(tuple, first.T.tolist()))
        self.assertTrue(all((v, u) in pairs for u, v in pairs))

    def test_rigid_and_empty_graphs_are_reported(self):
        for edges in (torch.empty((2, 0), dtype=torch.long), torch.tensor([[0, 0, 0], [1, 2, 3]])):
            result, audit = swap_edges(edges, seed=1)
            torch.testing.assert_close(result, edges, rtol=0, atol=0)
            self.assertEqual(audit["accepted_swaps"], 0)
            self.assertEqual(audit["changed_edges"], 0)

    def test_parallel_edges_rejected(self):
        with self.assertRaises(ValueError):
            swap_edges(torch.tensor([[0, 0], [1, 1]]), seed=1)

    def test_role_scope_labels_features_pooling_and_inputs_preserved(self):
        _, b = cycle_fixture()
        before = batch_hash(b)
        altered, audits = topology_batch(b, "rewired", seed=91, role="support")
        g, a = b[0], altered[0]
        mask = query_mask(b)[g.batch[g.edge_index[0]]]
        torch.testing.assert_close(a.edge_index[:, mask], g.edge_index[:, mask], rtol=0, atol=0)
        for key in ("x", "edge_index_supernode", "global_node_ids", "ptr", "batch", "edge_attr"):
            torch.testing.assert_close(a[key], g[key], rtol=0, atol=0)
        for i in range(1, len(b)):
            torch.testing.assert_close(altered[i], b[i], rtol=0, atol=0)
        self.assertEqual(before, batch_hash(b))
        self.assertTrue(all(torch.equal(a.batch[a.edge_index[0]], a.batch[a.edge_index[1]]) for _ in [0]))
        self.assertGreater(sum(r.get("changed_edges", 0) for r in audits), 0)

    def test_label_blind_null_and_descriptors(self):
        model, b = cycle_fixture()
        other = clone_batch(b)
        other[2] = other[2].flip(1)
        a, _ = topology_batch(b, "rewired", seed=7)
        c, _ = topology_batch(other, "rewired", seed=7)
        torch.testing.assert_close(a[0].edge_index, c[0].edge_index, rtol=0, atol=0)
        other = clone_batch(b)
        other[2][query_mask(b)] = other[2][query_mask(b)].flip(1)
        with torch.no_grad():
            enc = {name: encode(model, topology_batch(b, name, seed=7)[0])[0] for name in ("intact", "removed", "rewired")}
            values, desc = factorial_predictions(model, b, enc)
            changed, desc2 = factorial_predictions(model, other, enc)
        self.assertEqual(desc, desc2)
        for key in values:
            torch.testing.assert_close(values[key], changed[key], rtol=0, atol=0)

    def test_all_nine_combinations_match_direct_forward(self):
        model, b = cycle_fixture()
        seed = stable_seed("fixture", 7)
        with torch.no_grad():
            enc = {name: encode(model, topology_batch(b, name, seed=seed)[0])[0] for name in ("intact", "removed", "rewired")}
            values, _ = factorial_predictions(model, b, enc)
            for (q, s), logits in values.items():
                changed, _ = topology_batch(b, q, seed=seed, role="query")
                changed, _ = topology_batch(changed, s, seed=seed, role="support")
                _, direct = encode(model, changed)
                torch.testing.assert_close(direct, logits, rtol=0, atol=1e-6)

    def test_nested_deletion_dose_and_reciprocity(self):
        _, b = cycle_fixture()
        sets = []
        for condition in ("intact", "drop25", "drop50", "drop75", "removed"):
            changed, _ = topology_batch(b, condition, seed=4)
            pairs = set(map(tuple, changed[0].edge_index.T.tolist()))
            self.assertTrue(all((v, u) in pairs for u, v in pairs))
            sets.append(pairs)
        self.assertTrue(all(after < before for before, after in zip(sets, sets[1:])))

    def test_nonconstant_attributes_and_training_mode_rejected(self):
        model, b = cycle_fixture()
        b[0].edge_attr[0] = 2
        with self.assertRaises(ValueError):
            topology_batch(b, "rewired", seed=1)
        # The general fixture uses edge_attr_dim=1; production uses None.
        with self.assertRaises(ValueError):
            assert_background_attributes_unused(model)
        for layer in model.layer_list[0].module_list:
            layer.lin_edge_attr = None
        assert_background_attributes_unused(model)
        allowed, _ = topology_batch(b, "rewired", seed=1, edge_attributes_unused=True)
        torch.testing.assert_close(allowed[0].edge_attr, b[0].edge_attr, rtol=0, atol=0)
        model.layer_list[0].module_list[0].lin_edge_attr = torch.nn.Linear(1, 8)
        with self.assertRaises(ValueError):
            assert_background_attributes_unused(model)
        model.train()
        with self.assertRaises(ValueError):
            factorial_predictions(model, b, {})


if __name__ == "__main__":
    unittest.main()
