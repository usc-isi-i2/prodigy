import unittest
import random
import pandas as pd

import torch
from torch_geometric.data import Data, Batch

from experiments.layers import get_module_list
from models.general_gnn import SingleLayerGeneralGNN
from .replay import batch_hash, bn_mode, clone_batch, episode_probe, intervene, trace_stages, meta_bias_mode
from models.metaGNN import MetaGNNLayer
from .probe_cached_inputs import standardized_probe, input_summaries, detailed_summaries
from .probe_source_coverage import select_references, nearest_cosine, double_center, correlation
from .probe_metadata import metadata_features, FIELDS
from .audit_source_episodes import member_variants, raw_probe_stats


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
    def test_metadata_excludes_label_and_marks_missing(self):
        frame = pd.DataFrame({**{field: [3., None] for field in FIELDS},
                              "verified": ["True", "unknown"], "label_suspended": [1, 0]})
        values, missing = metadata_features(frame)
        changed = frame.copy()
        changed["label_suspended"] = [0, 1]
        other, _ = metadata_features(changed)
        torch.testing.assert_close(values, other)
        self.assertEqual(values.shape, (2, 12))
        self.assertTrue(torch.isfinite(values).all())
        self.assertEqual(float(values[0, -1]), 1.)
        torch.testing.assert_close(missing[1], torch.ones(12))

    def test_coverage_normalization_and_residualization(self):
        reference, selected = select_references(torch.eye(4) * 3, torch.ones(4), 4, "uniform_unique", 1)
        score = nearest_cosine(torch.eye(4) * 2, reference, chunk=2)
        torch.testing.assert_close(score, torch.ones(4))
        self.assertEqual(len(selected), 4)
        random_state = torch.get_rng_state().clone()
        select_references(torch.eye(4), torch.tensor([100, 1, 1, 1]), 4, "exposure_weighted", 3)
        torch.testing.assert_close(random_state, torch.get_rng_state(), rtol=0, atol=0)
        additive = torch.arange(4.)[:, None] + torch.arange(8.)[None, :]
        torch.testing.assert_close(double_center(additive), torch.zeros_like(additive))
        self.assertIsNone(correlation(torch.ones(5), torch.arange(5)))

    def test_detailed_inputs_exclude_center_and_supernode(self):
        _, batch = fixture()
        graph = batch[0]
        graph.x.zero_()
        for start in graph.ptr[:-1].tolist():
            graph.x[start] = 4
            graph.x[start + 1, 0] = 2
            graph.x[start + 3] = 100
        raw, topology, text = input_summaries(batch)
        detail = detailed_summaries(batch, raw, topology, text)
        torch.testing.assert_close(detail["context_nonzero_fraction"], torch.full((8,), .5))
        torch.testing.assert_close(detail["context_avg_norm"], torch.ones(8))
        torch.testing.assert_close(detail["context_mean_norm"], torch.ones(8))
        torch.testing.assert_close(detail["context_coherence"], torch.ones(8))
        torch.testing.assert_close(detail["context_pairwise_cosine"], torch.zeros(8))
        self.assertTrue(all(torch.isfinite(x).all() for x in detail.values()))
        for start in graph.ptr[:-1].tolist():
            graph.x[start + 2, 0] = 2
        raw, topology, text = input_summaries(batch)
        detail = detailed_summaries(batch, raw, topology, text)
        torch.testing.assert_close(detail["context_pairwise_cosine"], torch.ones(8))

    def test_meta_projection_bias_is_degree_scaled_and_control_removes_only_excess(self):
        layer = MetaGNNLayer(2, 8, heads=2, batch_norm=False).eval()
        with torch.no_grad():
            layer.out_proj.weight.zero_()
            layer.out_proj.bias.fill_(1)
            x = torch.zeros(3, 8)
            edges = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 1, 1, 2, 2, 2]])
            attrs = torch.zeros(6, 2)
            before = layer(x, edges, edge_attr=attrs)
            torch.testing.assert_close(before, torch.tensor([1., 2., 3.])[:, None].expand(-1, 8))
            with meta_bias_mode(layer, "meta_bias_normalized"):
                result = layer(x, edges, edge_attr=attrs)
            torch.testing.assert_close(result, torch.ones_like(result))
            torch.testing.assert_close(before, layer(x, edges, edge_attr=attrs), rtol=0, atol=0)

    def test_member_policies_hold_walks_and_selected_sets_fixed(self):
        walk = torch.arange(12).repeat(6)
        policies = member_variants({99: list(range(7))}, {99: walk}, random.Random(41))
        self.assertEqual(policies["production_sorted"].tolist(), [list(range(7))])
        self.assertEqual(set(policies["same_members_shuffled_roles"][0].tolist()), set(range(7)))
        self.assertEqual(len(set(policies["uniform_walk_endpoints"][0].tolist())), 7)
        with self.assertRaises(RuntimeError):
            member_variants({99: list(range(1, 8))}, {99: walk}, random.Random(41))

    def test_nm_raw_probe_known_perfect_case(self):
        features = torch.eye(30)[:, None, :].expand(-1, 7, -1)
        result = raw_probe_stats(features)
        self.assertEqual(result["raw_prototype_nm_accuracy"], 1.0)
        self.assertEqual(result["raw_support_query_cosine"], 1.0)
        self.assertEqual(result["raw_prototype_margin"], 1.0)

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
        torch.testing.assert_close(standardized_probe(x, batch), standardized_probe(x, changed))
        torch.testing.assert_close(standardized_probe(x, batch, x), standardized_probe(x, changed, x))

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
