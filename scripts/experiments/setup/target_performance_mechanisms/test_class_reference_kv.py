"""Synthetic checks for the actual-production support key/value factorial."""
import unittest

import torch

from models.metaGNN import MetaGNNLayer
from .class_reference_kv import kv_forward
from .message_scale import radial_swap
from .replay import batch_hash, clone_batch
from .role_context import query_mask
from .test_role_topology import cycle_fixture
from .verify_member_training import model_digest


class ClassReferenceKVTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, self.batch = cycle_fixture()
        self.model.layer_list[0].module_list[0].lin_edge_attr = None

    def test_joint_matches_removed_and_queries_stay_exact(self):
        before = batch_hash(self.batch)
        digest = model_digest(self.model.state_dict())
        a, ta, _, _, aa = kv_forward(self.model, self.batch)
        b, tb, _, _, ab = kv_forward(self.model, self.batch, alpha=0.)
        donor = ab['kqv_native'].clone()
        saved = donor.clone()
        q = query_mask(self.batch)
        support = aa['support_mask']
        dim = self.model.params['emb_dim']
        for k, v in ((donor, None), (None, donor), (donor, donor)):
            logits, trace, _, _, audit = kv_forward(self.model, self.batch, key_donor=k, value_donor=v)
            torch.testing.assert_close(trace['final_input'][q], ta['final_input'][q], rtol=0, atol=0)
            torch.testing.assert_close(audit['kqv_used'][:, :dim], aa['kqv_native'][:, :dim], rtol=0, atol=0)
            torch.testing.assert_close(audit['kqv_used'][~support], aa['kqv_native'][~support], rtol=0, atol=0)
            for donor_part, start in ((k, dim), (v, 2 * dim)):
                expected = aa['kqv_native'] if donor_part is None else donor
                torch.testing.assert_close(audit['kqv_used'][support, start:start + dim], expected[support, start:start + dim], rtol=0, atol=0)
            if k is None:
                torch.testing.assert_close(audit['attention'], aa['attention'], rtol=0, atol=0)
            if k is not None and v is not None:
                torch.testing.assert_close(logits, b, rtol=0, atol=0)
                torch.testing.assert_close(audit['final_labels'], ab['final_labels'], rtol=0, atol=0)
                # Supports retain intact residuals: only label and query outputs
                # need endpoint equivalence, and no second M pass may consume them.
                torch.testing.assert_close(audit['meta_input'], aa['meta_input'], rtol=0, atol=0)
            self.assertEqual(audit['max_attention_value_error'], 0.)
        torch.testing.assert_close(donor, saved, rtol=0, atol=0)
        torch.testing.assert_close(ta['final_input'][q], tb['final_input'][q], rtol=0, atol=0)
        self.assertEqual(before, batch_hash(self.batch))
        self.assertEqual(digest, model_digest(self.model.state_dict()))
        restored, _, _, _, _ = kv_forward(self.model, self.batch)
        torch.testing.assert_close(a, restored, rtol=0, atol=0)

    def test_identity_donors_and_attention_audit(self):
        a, _, _, _, audit = kv_forward(self.model, self.batch)
        b, _, _, _, swapped = kv_forward(self.model, self.batch,
            key_donor=audit['kqv_native'], value_donor=audit['kqv_native'])
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(audit['attention'], swapped['attention'], rtol=0, atol=0)
        torch.testing.assert_close(audit['attention_sum'], torch.ones_like(audit['attention_sum']))
        self.assertTrue(all(value == 1 for value in audit['capture_calls'].values()))

    def test_query_outcomes_do_not_affect_the_factorial(self):
        _, _, _, _, removed = kv_forward(self.model, self.batch, alpha=0.)
        changed = clone_batch(self.batch)
        changed[2][query_mask(changed)] = changed[2][query_mask(changed)].flip(1)
        options = {'key_donor': removed['kqv_native'], 'value_donor': removed['kqv_native']}
        a, _, _, _, _ = kv_forward(self.model, self.batch, **options)
        b, _, _, _, _ = kv_forward(self.model, changed, **options)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_masks_and_donors_rejected_and_hooks_removed(self):
        _, _, _, _, audit = kv_forward(self.model, self.batch)
        for donor in (torch.zeros(1, 3), audit['kqv_native'].double(), audit['kqv_native'] * float('nan')):
            with self.assertRaises(ValueError):
                kv_forward(self.model, self.batch, key_donor=donor)
        for mode in ('short', 'mixed', 'nonbinary', 'all_query', 'all_support'):
            changed = clone_batch(self.batch)
            if mode == 'short':
                changed[5] = changed[5][:-1]
            elif mode == 'mixed':
                changed[5][0] = ~changed[5][0]
            elif mode == 'nonbinary':
                changed[5] = changed[5].float()
                changed[5][0] = .5
            else:
                changed[5].fill_(mode == 'all_query')
            with self.assertRaises(ValueError):
                kv_forward(self.model, changed)
        layer = self.model.layer_list[2].gnn_layers[0]
        for module in (layer, layer.mlp_kqv, layer.att_mlp, layer.out_proj, self.model.final_label_mlp):
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)

    def test_non_single_layer_or_changed_model_contract_rejected(self):
        for variant in ('training', 'second_meta', 'back', 'projection', 'bn', 'dropout'):
            model, batch = cycle_fixture()
            model.layer_list[0].module_list[0].lin_edge_attr = None
            meta = model.layer_list[2]
            if variant == 'training':
                model.train()
            elif variant == 'second_meta':
                meta.gnn_layers.append(MetaGNNLayer(2, 8).eval())
                meta.num_gnn_layers = 2
            elif variant == 'back':
                meta.gnn_layers_back = MetaGNNLayer(2, 8).eval()
            elif variant == 'projection':
                model.final_label_mlp = torch.nn.Linear(8, 8).eval()
            elif variant == 'bn':
                meta.gnn_layers[0].bn.track_running_stats = False
            else:
                meta.gnn_layers[0].dropout = .1
            with self.assertRaises(ValueError):
                kv_forward(model, batch)

    def test_radial_factorial_cross_term_survives_an_affine_map(self):
        a = torch.tensor([[3., 4.], [0., 2.]], dtype=torch.float64)
        b = torch.tensor([[0., 2.], [4., 3.]], dtype=torch.float64)
        direction, _ = radial_swap(b, a)
        norm, _ = radial_swap(a, b)
        ra, rb = a.norm(dim=1, keepdim=True), b.norm(dim=1, keepdim=True)
        cross = (rb - ra) * (b / rb - a / ra)
        torch.testing.assert_close(b - direction - norm + a, cross)
        weight = torch.tensor([[2., -1.], [.5, 3.]], dtype=torch.float64)
        bias = torch.tensor([7., -4.], dtype=torch.float64)
        affine = lambda x: x @ weight.T + bias
        observed = affine(b) - affine(direction) - affine(norm) + affine(a)
        torch.testing.assert_close(observed, cross @ weight.T)
        self.assertGreater(float(observed.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
