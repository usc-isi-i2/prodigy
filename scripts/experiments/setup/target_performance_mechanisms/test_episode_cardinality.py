import unittest

import torch
from torch_geometric.utils import add_self_loops

from models.metaGNN import MetaGNN, MetaGNNLayer
from .episode_cardinality import (VARIANTS, edge_counts, multiplicity_hooks,
                                  cardinality_mode, cached_meta_forward)
from .replay import clone_batch, trace_stages
from .test_replay import fixture


def geometry(shots=10, queries=2):
    count = 2 * shots + queries
    edges = torch.stack((torch.arange(count).repeat_interleave(2),
                         count + torch.arange(2).repeat(count)))
    query = torch.arange(count).repeat_interleave(2) >= 2 * shots
    y = torch.arange(count) % 2
    values = ((torch.nn.functional.one_hot(y, 2) * 2 - 1) * ~query.reshape(-1, 2)).reshape(-1)
    attrs = torch.stack((query.float(), values.float()), dim=1)
    return edges, attrs, query, count


class CardinalityTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(731)
        torch.set_num_threads(1)

    def test_production_count_factors(self):
        edges, attrs, query, count = geometry()
        edges = torch.cat((edges[:, ~query], edges.flip(0)), 1)
        attrs = torch.cat((attrs[~query], attrs))
        edges, attrs = add_self_loops(edges, attrs, fill_value=torch.zeros(2), num_nodes=count + 2)
        way, shot, roles, audit = edge_counts(edges, attrs, count, count + 2)
        self.assertEqual((audit["eval_ways"], audit["eval_shots"]), (2, 10))
        for role, value in {"label_positive": .3, "label_negative": 8.7,
                            "query_labels": 15., "support_positive": 1.,
                            "support_negative": 29., "label_self": 1., "query_self": 1., "support_self": 1.}.items():
            torch.testing.assert_close((way * shot)[roles[role]], torch.full_like(way[roles[role]], value))
        with self.assertRaises(ValueError):
            edge_counts(edges[:, :-1], attrs[:-1], count, count + 2)
        bad = attrs.clone()
        bad[roles["query_labels"], 1] = 1
        with self.assertRaises(ValueError):
            edge_counts(edges, bad, count, count + 2)

    def test_integer_virtual_replication_and_identity(self):
        layer = MetaGNNLayer(2, 8, heads=2, batch_norm=True).double().eval()
        x = torch.randn(3, 8, dtype=torch.float64)
        edges = torch.tensor([[0, 1, 2, 1], [0, 0, 1, 2]])
        attrs = torch.randn(4, 2, dtype=torch.float64)
        weights = torch.tensor([1., 3., 2., 4.], dtype=torch.float64)
        with torch.no_grad():
            reference = layer(x, edges, edge_attr=attrs)
            with multiplicity_hooks(layer, torch.ones_like(weights), adjust_bias=True):
                torch.testing.assert_close(reference, layer(x, edges, edge_attr=attrs), rtol=0, atol=0)
            repeated = layer(x, edges.repeat_interleave(weights.long(), dim=1), edge_attr=attrs.repeat_interleave(weights.long(), dim=0))
            with multiplicity_hooks(layer, weights, adjust_bias=True):
                weighted = layer(x, edges, edge_attr=attrs)
            torch.testing.assert_close(repeated, weighted, rtol=1e-12, atol=1e-12)
            torch.testing.assert_close(reference, layer(x, edges, edge_attr=attrs), rtol=0, atol=0)

    def test_attention_does_not_rescale_bias(self):
        layer = MetaGNNLayer(2, 8, heads=2, batch_norm=False).eval()
        with torch.no_grad():
            layer.out_proj.weight.zero_()
            layer.out_proj.bias.fill_(1)
            x = torch.zeros(2, 8)
            edges = torch.tensor([[0, 1, 0], [0, 0, 1]])
            attrs = torch.zeros(3, 2)
            weights = torch.tensor([2., 5., .25])
            reference = layer(x, edges, edge_attr=attrs)
            with multiplicity_hooks(layer, weights):
                torch.testing.assert_close(reference, layer(x, edges, edge_attr=attrs), rtol=0, atol=0)
            with multiplicity_hooks(layer, weights, adjust_bias=True):
                expected = torch.tensor([7., .25])[:, None].expand(-1, 8)
                torch.testing.assert_close(expected, layer(x, edges, edge_attr=attrs))

    def test_cached_production_suffix_and_query_label_independence(self):
        model, batch = fixture()
        batch[1].normal_()
        with torch.no_grad(), trace_stages(model, batch[0]) as traces:
            _, reference, _ = model(*clone_batch(batch))
        changed = clone_batch(batch)
        query = batch[5].reshape(-1, 2)[:, 0]
        changed[2][query] = changed[2][query].flip(1)
        with torch.no_grad():
            with cardinality_mode(model.layer_list[2], "baseline") as audit:
                x, logits = cached_meta_forward(model, batch, traces["U1_pre_meta"])
            torch.testing.assert_close(x, traces["M2_post_meta"], rtol=0, atol=0)
            torch.testing.assert_close(logits, reference, rtol=0, atol=0)
            self.assertEqual(len(audit), 1)
            for variant in VARIANTS:
                with cardinality_mode(model.layer_list[2], variant):
                    _, first = cached_meta_forward(model, batch, traces["U1_pre_meta"])
                    _, second = cached_meta_forward(model, changed, traces["U1_pre_meta"])
                torch.testing.assert_close(first, second, rtol=0, atol=0)
            _, restored = cached_meta_forward(model, batch, traces["U1_pre_meta"])
            torch.testing.assert_close(reference, restored, rtol=0, atol=0)

    def test_added_query_cannot_change_existing_outputs(self):
        meta = MetaGNN(2, 8, heads=2, batch_norm=True).double().eval()
        edges, attrs, query, count = geometry(shots=1, queries=2)
        x = torch.randn(count + 2, 8, dtype=torch.float64)
        new_x = torch.cat((x[:count], torch.randn(1, 8, dtype=x.dtype), x[count:]))
        new_edges = edges.clone()
        new_edges[1] += 1
        new_edges = torch.cat((new_edges, torch.tensor([[count, count], [count + 1, count + 2]])), 1)
        new_attrs = torch.cat((attrs, torch.tensor([[1., 0.], [1., 0.]]))).double()
        new_query = torch.cat((query, torch.tensor([True, True])))
        with torch.no_grad():
            for variant in VARIANTS:
                with cardinality_mode(meta, variant):
                    before = meta(x, edges, attrs.double(), query, count)
                    after = meta(new_x, new_edges, new_attrs, new_query, count + 1)
                torch.testing.assert_close(before[:count], after[:count], rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(before[count:], after[count+1:], rtol=1e-12, atol=1e-12)

    def test_hooks_removed_on_exception(self):
        meta = MetaGNN(2, 8, heads=2).eval()
        layer = meta.gnn_layers[0]
        with self.assertRaises(RuntimeError):
            with cardinality_mode(meta, "count_joint_attention"):
                raise RuntimeError("test")
        self.assertEqual(len(layer._forward_pre_hooks), 0)
        self.assertEqual(len(layer.att_mlp._forward_hooks), 0)
        self.assertEqual(len(layer.out_proj._forward_hooks), 0)
        with self.assertRaises(ValueError):
            with multiplicity_hooks(layer, torch.tensor([0.])):
                pass


if __name__ == "__main__":
    unittest.main()
