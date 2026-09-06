import unittest
import torch

from .inspect_examples import CONDITIONS, localized_intervention, stratum
from .test_replay import fixture
from .replay import batch_hash


class ExampleAuditTest(unittest.TestCase):
    def test_strata(self):
        self.assertEqual(stratum(1, 1, 0, 0), "ukr_right_hk_wrong")
        self.assertEqual(stratum(0, 1, 0, 0), "hk_right_ukr_wrong")
        self.assertEqual(stratum(1, 0, 0, 1), "both_wrong_raw_right")
        self.assertEqual(stratum(0, 0, 0, 1), "both_right_raw_wrong")
        self.assertIsNone(stratum(1, 1, 1, 1))

    def test_localized_features_preserve_other_samples_and_labels(self):
        _, b = fixture()
        digest = batch_hash(b)
        for condition in CONDITIONS:
            changed = localized_intervention(b, 2, condition)
            selected = torch.tensor([0, 1]) if condition.startswith("support") else torch.tensor([2])
            outside = ~torch.isin(b[0].batch, selected)
            torch.testing.assert_close(changed[0].x[outside], b[0].x[outside], rtol=0, atol=0)
            for i in range(1, len(b)):
                torch.testing.assert_close(changed[i], b[i], rtol=0, atol=0)
            torch.testing.assert_close(changed[0].global_node_ids, b[0].global_node_ids)
            if condition.endswith("edges_removed"):
                self.assertFalse(torch.isin(changed[0].batch[changed[0].edge_index[0]], selected).any())
            else:
                torch.testing.assert_close(changed[0].edge_index, b[0].edge_index)
        self.assertEqual(batch_hash(b), digest)

    def test_zero_neighbor_query_context_is_noop(self):
        _, b = fixture()
        start, end = b[0].ptr[2:4].tolist()
        b[0].global_node_ids[start+1:end] = -1
        changed = localized_intervention(b, 2, "query_context_replaced_by_center")
        self.assertEqual(batch_hash(b), batch_hash(changed))

    def test_support_cannot_be_selected_as_query(self):
        _, b = fixture()
        with self.assertRaises(ValueError):
            localized_intervention(b, 0, CONDITIONS[0])


if __name__ == "__main__":
    unittest.main()
