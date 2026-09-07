import unittest

import torch

from .support_dose import DOSES, dose_masks, masked_batch
from .test_role_topology import cycle_fixture
from .replay import batch_hash, clone_batch
from .role_context import query_mask, intervene_roles
from .role_topology import encode


class SupportDoseTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.batch = cycle_fixture()

    def test_nested_reciprocal_deterministic_and_label_blind(self):
        before = batch_hash(self.batch)
        rng = torch.get_rng_state().clone()
        masks, stats = dose_masks(self.batch, seed=91)
        other = clone_batch(self.batch)
        other[2] = other[2].flip(1)
        again, stats2 = dose_masks(other, seed=91)
        self.assertEqual(stats, stats2)
        for dose in DOSES:
            torch.testing.assert_close(masks[dose], again[dose], rtol=0, atol=0)
            pairs = set(map(tuple, self.batch[0].edge_index[:, masks[dose]].T.tolist()))
            self.assertTrue(all((v, u) in pairs for u, v in pairs))
        for a, b in zip(DOSES, DOSES[1:]):
            self.assertFalse(torch.any(masks[b] & ~masks[a]))
        self.assertEqual(before, batch_hash(self.batch))
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)

    def test_endpoints_match_existing_role_intervention_and_query_encodings(self):
        masks, _ = dose_masks(self.batch, seed=12)
        self.assertEqual(batch_hash(masked_batch(self.batch, masks[0])), batch_hash(self.batch))
        self.assertEqual(batch_hash(masked_batch(self.batch, masks[100])),
                         batch_hash(intervene_roles(self.batch, "edges_support")))
        with torch.no_grad():
            original, _ = encode(self.model, self.batch)
            for dose in DOSES:
                pre, _ = encode(self.model, masked_batch(self.batch, masks[dose]))
                q = query_mask(self.batch)
                torch.testing.assert_close(pre[q], original[q], rtol=0, atol=0)

    def test_self_loops_are_suppressed_at_complete_endpoint(self):
        g = self.batch[0]
        g.edge_index = torch.cat([g.edge_index, g.ptr[:-1].repeat(2, 1)], dim=1)
        g.edge_attr = torch.cat([g.edge_attr, torch.ones(len(g.ptr)-1, 1)], dim=0)
        masks, stats = dose_masks(self.batch, seed=8)
        self.assertEqual(stats[100]["retained_support_edges"], 0)
        self.assertEqual(stats[100]["support_self_loops"], 4)
        self.assertEqual(batch_hash(masked_batch(self.batch, masks[100])),
                         batch_hash(intervene_roles(self.batch, "edges_support")))

    def test_directed_empty_and_attribute_alignment(self):
        g = self.batch[0]
        keep = g.edge_index[0] < g.edge_index[1]
        g.edge_index = g.edge_index[:, keep]
        g.edge_attr = torch.arange(g.edge_index.shape[1]).float()[:, None]
        masks, stats = dose_masks(self.batch, seed=13)
        for dose in DOSES:
            changed = masked_batch(self.batch, masks[dose])[0]
            torch.testing.assert_close(changed.edge_attr, g.edge_attr[masks[dose]], rtol=0, atol=0)
            self.assertEqual(stats[dose]["support_edges"], stats[dose]["support_units"])
        g.edge_index = g.edge_index[:, :0]
        g.edge_attr = g.edge_attr[:0]
        _, stats = dose_masks(self.batch, seed=13)
        self.assertTrue(all(s["nonempty_support_subgraphs"] == 0 for s in stats.values()))


if __name__ == "__main__":
    unittest.main()
