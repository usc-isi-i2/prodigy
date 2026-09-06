import unittest

import torch
from torch_geometric.data import Batch

from .fixed_support_context import (draw_seed, graph_hash, support_draw, replace_support_contexts,
    verify_replacement, context_comparison, score_context_draws)
from .replay import batch_hash, clone_batch
from .role_context import query_mask, changed_embedding
from .episode_cardinality import cached_meta_forward
from .test_natural_support import support_fixture


class ToyDataset:
    def __init__(self, batch):
        self.graphs = {int(g.global_node_ids[0]): g.clone() for g in batch[0].to_data_list()}

    def __getitem__(self, center):
        g = self.graphs[center].clone()
        g.x[1:-1] += torch.rand_like(g.x[1:-1])
        return g


class FixedSupportTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_original_context_roundtrip_preserves_all_tensor_fields(self):
        _, b = support_fixture()
        b[0].extra_metadata = torch.arange(3)
        q = query_mask(b)
        supports = Batch.from_data_list([g for i, g in enumerate(b[0].to_data_list()) if not q[i]])
        self.assertEqual(batch_hash(b), batch_hash(replace_support_contexts(b, supports)))

    def test_draw_is_query_blind_and_restores_global_rng(self):
        _, b = support_fixture()
        dataset, before = ToyDataset(b), torch.get_rng_state()
        a = support_draw(dataset, b, 923)
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        changed = clone_batch(b)
        q = query_mask(b)
        changed[0].x[q[changed[0].batch]] = 999
        changed[2][q] = changed[2][q].flip(1)
        other = support_draw(dataset, changed, 923)
        self.assertEqual(graph_hash(a), graph_hash(other))
        self.assertNotEqual(graph_hash(a), graph_hash(support_draw(dataset, b, 924)))

    def test_sampled_contexts_keep_centers_queries_and_truth(self):
        _, b = support_fixture()
        original = batch_hash(b)
        changed = replace_support_contexts(b, support_draw(ToyDataset(b), b, 3))
        self.assertTrue(all(verify_replacement(b, changed).values()))
        self.assertEqual(original, batch_hash(b))
        self.assertNotEqual(original, batch_hash(changed))
        changed[2][0] = changed[2][0].flip(0)
        with self.assertRaises(AssertionError):
            verify_replacement(b, changed)

    def test_full_forward_query_vectors_and_suffix_match(self):
        model, b = support_fixture()
        q = query_mask(b)
        with torch.no_grad():
            pre, baseline = changed_embedding(model, b, "baseline")
            changed = replace_support_contexts(b, support_draw(ToyDataset(b), b, 82))
            new_pre, direct = changed_embedding(model, changed, "baseline")
            torch.testing.assert_close(new_pre[q], pre[q], rtol=0, atol=0)
            base_x, _ = cached_meta_forward(model, b, pre)
            x, z = cached_meta_forward(model, b, new_pre)
            torch.testing.assert_close(x[q], base_x[q], rtol=0, atol=0)
            torch.testing.assert_close(z, direct, rtol=0, atol=0)

    def test_seed_grid_is_fixed_and_distinct(self):
        targets = ("covid_political", "election2020", "twibot20", "ukr_rus_suspended", "facebook_page_reference")
        seeds = [draw_seed(s, t, b, d) for s in ("original", "fresh") for t in targets for b in range(32) for d in range(1, 8)]
        self.assertEqual(len(set(seeds)), 2240)

    def test_zero_variation_accounting(self):
        z = torch.tensor([[1., -1.], [-1., 1.], [1., -1.], [-1., 1.]])[None].repeat(8, 1, 1)
        labels = dict(local_y=torch.tensor([0, 1, 0, 1]), mapping=torch.tensor([[0, 1]]*4),
            episode_ids=torch.tensor([0, 0, 1, 1]), use_global=True)
        rows, ep, cohorts = score_context_draws(z, labels, torch.tensor([0, 1, 1, 2]), {})
        self.assertEqual(len(rows), 9)
        self.assertEqual(len(ep), 16)
        for r in cohorts:
            self.assertEqual(r["mean_probability_variance"], 0)
            self.assertEqual(r["correctness_flips"], 0)


if __name__ == "__main__":
    unittest.main()
