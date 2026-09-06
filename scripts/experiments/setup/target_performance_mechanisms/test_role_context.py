import unittest
import torch

from .inspect_examples import localized_intervention
from .role_context import VARIANTS, intervene_roles, query_mask, role_predictions
from .replay import batch_hash, clone_batch, trace_stages
from .test_replay import fixture
from .run_role_context import paired_counts, cohorts
from .check_support_path_seeds import inspect_path


class RoleContextTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_composed_suffix_matches_every_whole_forward(self):
        model, b = fixture()
        digest = batch_hash(b)
        with torch.no_grad():
            with trace_stages(model, b[0]) as trace:
                model(*clone_batch(b))
            predictions, checks = role_predictions(model, b, trace["U1_pre_meta"], verify_direct=True)
            self.assertEqual(set(predictions), set(VARIANTS))
            self.assertEqual(len(checks), 7)
            for variant in VARIANTS:
                _, direct, _ = model(*intervene_roles(b, variant))
                torch.testing.assert_close(direct, predictions[variant], rtol=0, atol=1e-6)
        self.assertEqual(batch_hash(b), digest)

    def test_roles_leave_other_feature_rows_and_labels_fixed(self):
        _, b = fixture()
        q = query_mask(b)
        for role in ("query", "support"):
            changed = intervene_roles(b, "features_" + role)
            selected = q if role == "query" else ~q
            untouched = ~selected[b[0].batch]
            torch.testing.assert_close(changed[0].x[untouched], b[0].x[untouched], rtol=0, atol=0)
            for i in range(1, len(b)):
                torch.testing.assert_close(changed[i], b[i], rtol=0, atol=0)

    def test_joint_query_changes_equal_localized_change_for_selected_query(self):
        model, b = fixture()
        pairs = {"features_query": "query_context_replaced_by_center", "edges_query": "query_edges_removed",
                 "center_zero_query": "query_center_zeroed"}
        with torch.no_grad():
            for variant, local in pairs.items():
                _, all_queries, _ = model(*intervene_roles(b, variant))
                for qi, s in enumerate(query_mask(b).nonzero().flatten().tolist()):
                    _, one_query, _ = model(*localized_intervention(b, s, local))
                    torch.testing.assert_close(all_queries[qi], one_query[qi], rtol=0, atol=1e-6)

    def test_reject_train_mode_or_transductive_bn(self):
        model, b = fixture()
        pre = torch.zeros(8, 8)
        model.train()
        with self.assertRaises(ValueError):
            role_predictions(model, b, pre)
        model.eval()
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm1d):
                module.track_running_stats = False
                break
        with self.assertRaises(ValueError):
            role_predictions(model, b, pre)

    def test_paired_counts_include_harm_as_well_as_rescue(self):
        y = torch.tensor([0, 1, 0, 1])
        base = torch.tensor([[2., 0.], [2., 0.], [2., 0.], [2., 0.]])
        changed = torch.tensor([[0., 2.], [0., 2.], [2., 0.], [0., 2.]])
        r = paired_counts(base, changed, y, torch.ones(4, dtype=torch.bool))
        self.assertEqual((r["errors_fixed"], r["errors_introduced"]), (2, 1))
        self.assertEqual(r["delta_accuracy"], .25)
        self.assertIsNone(paired_counts(base, changed, y, torch.zeros(4, dtype=torch.bool))["delta_accuracy"])

    def test_cohorts_do_not_use_truth_and_exclude_empty_context_agreement(self):
        meta = {"context_nodes": torch.tensor([0, 3, 4]), "incoming": torch.tensor([0, 2, 0]),
                "outgoing": torch.tensor([0, 0, 2]), "raw_center_prediction": torch.tensor([0, 1, 1]),
                "raw_context_prediction": torch.tensor([0, 1, 0])}
        masks = cohorts(meta)
        self.assertEqual(masks["raw_center_context_agree"].tolist(), [False, True, False])
        self.assertEqual(masks["raw_center_context_disagree"].tolist(), [False, False, True])

    def test_support_edges_change_label_side_only_in_single_meta_layer(self):
        model, b = fixture()
        digest = batch_hash(b)
        with torch.no_grad():
            values, audit = inspect_path(model, b)
        self.assertEqual(set(values), {"baseline", "edges_support"})
        self.assertTrue(audit["query_pre_and_post_bit_exact"])
        self.assertTrue(audit["label_only_recomposition_bit_exact"])
        self.assertGreater(audit["label_max_abs_change"], 0)
        self.assertEqual(batch_hash(b), digest)


if __name__ == "__main__":
    unittest.main()
