"""Synthetic tensor-only checks; no native model, dataset or GPU required."""

import unittest

import torch
from torch_geometric.data import Batch, Data

try:
    from .cached_tensors import (
        cosine_prototype_logits, raw_descriptors, suppress_background, validate_episode,
    )
except ImportError:
    from cached_tensors import (
        cosine_prototype_logits, raw_descriptors, suppress_background, validate_episode,
    )


def make_episode(ways=2, empty_edges=False):
    """Use genuine PyG batching and the pinned upstream metagraph ordering."""
    graphs = []
    labels = torch.arange(ways).repeat_interleave(7)
    is_query = torch.tensor([False] * 3 + [True] * 4).repeat(ways)
    for graph_idx in range(len(labels)):
        x = torch.zeros(4, 770)
        x[0, int(labels[graph_idx])] = graph_idx + 1.0
        x[1, 767 - int(labels[graph_idx])] = graph_idx + 2.0
        x[0, -1] = 1
        x[1, -2] = 1
        edges = torch.empty((2, 0), dtype=torch.long) if empty_edges else torch.tensor(
            [[0, 1, 2], [1, 2, 0]])
        graphs.append(Data(
            x=x, edge_index=edges,
            edge_attr=torch.full((edges.shape[1], 768), float(graph_idx)),
            supernode=torch.tensor([3]),
            edge_index_supernode=torch.tensor([[0, 1, 2], [3, 3, 3]]),
            edge_index_from_supernode=torch.tensor([[3, 3, 3], [0, 1, 2]]),
            sample_id=torch.tensor([graph_idx]),
        ))
    graph = Batch.from_data_list(graphs)
    y = torch.nn.functional.one_hot(labels, ways).float()
    count = len(labels)
    meta_index = torch.stack((torch.arange(count).repeat_interleave(ways),
                              torch.arange(ways).repeat(count) + count))
    query_mask = is_query.repeat_interleave(ways)
    meta_attr = torch.stack((query_mask.float(),
                             (2 * y.flatten() - 1) * ~query_mask), dim=1)
    # Sequence values are intentionally opaque: helpers must preserve them.
    seqs = tuple(torch.arange(i + 2).reshape(1, -1) for i in range(3))
    return (graph, torch.zeros(ways, 768), y, meta_index, meta_attr, query_mask, *seqs)


class CachedTensorTests(unittest.TestCase):
    def assert_tensors_equal(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(torch.equal(actual, expected))

    def assert_non_edges_equal(self, actual, expected):
        self.assertEqual(set(actual[0].keys), set(expected[0].keys))
        for key in expected[0].keys:
            if key not in ("edge_index", "edge_attr"):
                self.assert_tensors_equal(actual[0][key], expected[0][key])
        for new, old in zip(actual[1:], expected[1:]):
            self.assert_tensors_equal(new, old)

    def test_public_default_and_local_labels(self):
        episode = validate_episode(make_episode(20))
        self.assertEqual(episode["ways"], 20)
        self.assertEqual(episode["is_query"].shape, (140,))
        self.assertEqual(episode["is_query"].dtype, torch.bool)
        self.assertEqual(int(episode["is_query"].sum()), 80)
        self.assertEqual(episode["labels"].dtype, torch.long)
        self.assertEqual(episode["labels"].tolist(), torch.arange(20).repeat_interleave(7).tolist())

    def test_small_fixture_requires_explicit_validate_ways(self):
        args = make_episode()
        with self.assertRaisesRegex(ValueError, "20-way"):
            validate_episode(args)
        self.assertEqual(validate_episode(args, expected_ways=2)["ways"], 2)

    def test_bad_onehot_is_rejected(self):
        args = make_episode()
        args[2][0, 1] = 1
        with self.assertRaisesRegex(ValueError, "exactly one-hot"):
            validate_episode(args, expected_ways=2)

    def test_edge_mask_is_not_a_graph_mask(self):
        args = make_episode()
        args[5][1] = True
        with self.assertRaisesRegex(ValueError, "inconsistent roles"):
            validate_episode(args, expected_ways=2)

    def test_noncanonical_metagraph_is_rejected(self):
        args = make_episode()
        args[3][:, [0, 1]] = args[3][:, [1, 0]]
        with self.assertRaisesRegex(ValueError, "canonical"):
            validate_episode(args, expected_ways=2)

    def test_wrong_metagraph_support_sign_is_rejected(self):
        args = make_episode()
        args[4][0, 1] *= -1
        with self.assertRaisesRegex(ValueError, "attributes disagree"):
            validate_episode(args, expected_ways=2)

    def test_per_class_counts_are_enforced(self):
        args = make_episode()
        args[5][:2] = True
        args[4][:2, 0] = 1
        args[4][:2, 1] = 0
        with self.assertRaisesRegex(ValueError, "3 supports"):
            validate_episode(args, expected_ways=2)

    def test_background_edge_alignment_and_cross_graph_checks(self):
        args = make_episode()
        args[0].edge_attr = args[0].edge_attr[:-1]
        with self.assertRaisesRegex(ValueError, "not aligned"):
            suppress_background(args)
        args = make_episode()
        args[0].edge_index[1, 0] = args[0].ptr[1]
        with self.assertRaisesRegex(ValueError, "crosses subgraphs"):
            suppress_background(args)

    def test_raw_head_tail_order_and_flag_exclusion(self):
        args = make_episode()
        actual = raw_descriptors(args)
        self.assertEqual(actual.shape, (14, 1536))
        self.assert_tensors_equal(actual[:, :768], args[0].x[args[0].ptr[:-1], :768])
        self.assert_tensors_equal(actual[:, 768:], args[0].x[args[0].ptr[:-1] + 1, :768])
        self.assertEqual(int(torch.count_nonzero(actual[0])), 2)
        actual[0, 0] = -99
        self.assertEqual(float(args[0].x[0, 0]), 1.0)

    def test_wrong_head_tail_or_other_flag_is_rejected(self):
        for node in (0, 2):
            with self.subTest(node=node):
                args = make_episode()
                args[0].x[node, -2] = 1
                with self.assertRaisesRegex(ValueError, "head/tail flags"):
                    raw_descriptors(args)

    def test_cosine_readout_uses_example_then_prototype_normalization(self):
        args = make_episode()
        descriptors = args[2].double().clone()
        descriptors[0] = torch.tensor([2.0, 0.0])
        descriptors[1] = torch.tensor([0.0, 10.0])
        descriptors[2] = torch.tensor([2.0, 0.0])
        truth, logits, diagnostic = cosine_prototype_logits(descriptors, args)
        # Normalize each example first: the mean direction is [2, 1], not [4, 10].
        self.assertAlmostEqual(float(logits[0, 0]), 2 / (5 ** 0.5))
        self.assertEqual(logits.dtype, torch.float64)
        self.assertEqual(logits.shape, (8, 2))
        self.assert_tensors_equal(truth, args[2][args[5].reshape(14, 2)[:, 0]])
        self.assertEqual(diagnostic["zero_descriptor_count"], 0)
        self.assertEqual(diagnostic["zero_prototype_count"], 0)

    def test_query_labels_do_not_affect_prototypes_or_logits(self):
        args = make_episode()
        descriptors = args[2].clone()
        original_truth, original_logits, _ = cosine_prototype_logits(descriptors, args)
        query_rows = args[5].reshape(14, 2)[:, 0]
        args[2][query_rows] = args[2][query_rows].flip(1)
        truth, logits, _ = cosine_prototype_logits(descriptors, args)
        self.assert_tensors_equal(logits, original_logits)
        self.assertFalse(torch.equal(truth, original_truth))

    def test_zero_norms_are_explicit_and_remain_finite(self):
        args = make_episode()
        descriptors = args[2].clone()
        descriptors[0] = torch.tensor([1.0, 0.0])
        descriptors[1] = torch.tensor([-1.0, 0.0])
        descriptors[2] = 0
        descriptors[3] = 0
        _, logits, diagnostic = cosine_prototype_logits(descriptors, args)
        self.assertEqual(diagnostic["zero_descriptor_count"], 2)
        self.assertEqual(diagnostic["zero_support_descriptor_count"], 1)
        self.assertEqual(diagnostic["zero_query_descriptor_count"], 1)
        self.assertEqual(diagnostic["zero_prototype_count"], 1)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue((logits[:, 0] == 0).all())
        self.assertTrue((logits[0] == 0).all())

    def test_nonfinite_descriptors_and_norm_overflow_fail(self):
        for value in (float("nan"), float("inf"), torch.finfo(torch.float32).max):
            with self.subTest(value=value):
                args = make_episode()
                descriptors = args[2].clone()
                descriptors[0] = value
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    cosine_prototype_logits(descriptors, args)

    def test_suppression_is_role_specific_and_input_is_immutable(self):
        for role in ("support", "query", "intact"):
            with self.subTest(role=role):
                args = make_episode()
                snapshot = tuple(value.clone() for value in args)
                result, diagnostic = suppress_background(args, role)
                role_mask = validate_episode(args, expected_ways=2)["is_query"]
                query_edges = role_mask[args[0].batch[args[0].edge_index[0]]]
                keep = query_edges if role == "support" else ~query_edges
                if role == "intact":
                    keep = torch.ones_like(query_edges)
                self.assert_non_edges_equal(result, args)
                self.assert_tensors_equal(result[0].edge_index, args[0].edge_index[:, keep])
                self.assert_tensors_equal(result[0].edge_attr, args[0].edge_attr[keep])
                self.assert_non_edges_equal(args, snapshot)
                self.assert_tensors_equal(args[0].edge_index, snapshot[0].edge_index)
                self.assert_tensors_equal(args[0].edge_attr, snapshot[0].edge_attr)
                self.assertEqual(diagnostic["edges_after"], int(keep.sum()))
                self.assertEqual(diagnostic["edges_removed"], int((~keep).sum()))
                validate_episode(result, expected_ways=2)
                # All surviving tensors, not just filtered edges, must be fresh.
                for key in args[0].keys:
                    if args[0][key].numel() and result[0][key].numel():
                        self.assertNotEqual(args[0][key].data_ptr(), result[0][key].data_ptr())
                for old, new in zip(args[1:], result[1:]):
                    self.assertNotEqual(old.data_ptr(), new.data_ptr())
                result[0].x[0, 0] = -999
                self.assert_tensors_equal(args[0].x, snapshot[0].x)

    def test_no_background_edges_is_valid(self):
        args = make_episode(empty_edges=True)
        for role in ("support", "query", "intact"):
            result, diagnostic = suppress_background(args, role)
            self.assertEqual(result[0].edge_index.shape, (2, 0))
            self.assertEqual(result[0].edge_attr.shape, (0, 768))
            self.assertEqual(diagnostic["edges_removed"], 0)

    def test_unknown_role_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown suppression role"):
            suppress_background(make_episode(), "both")


if __name__ == "__main__":
    unittest.main()
