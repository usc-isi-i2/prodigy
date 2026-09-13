import unittest

import torch

from models.general_gnn import SingleLayerGeneralGNN
from models.metaGNN import MetaGNN
from .metagraph import replay, compact_trace, radial_value_donors


def fixture():
    torch.manual_seed(709)
    ways, dim, per_class = 3, 8, 3
    n = ways * per_class
    model = SingleLayerGeneralGNN(torch.nn.ModuleList([
        torch.nn.Identity(), torch.nn.Identity(),
        MetaGNN(emb_dim=dim, edge_attr_dim=2, n_layers=1, heads=2,
                dropout=0, has_final_back=False, msg_pos_only=False),
    ]), params=dict(emb_dim=dim, zero_shot=False, skip_path=False)).eval()
    pre, label = torch.randn(n, dim), torch.randn(ways, dim)
    truth = torch.arange(n) // per_class
    query = torch.arange(n) % per_class == per_class - 1
    source = torch.arange(n).repeat_interleave(ways)
    dest = torch.arange(ways).repeat(n) + n
    edges = torch.stack([source, dest])
    roles = query.repeat_interleave(ways)
    signs = (torch.nn.functional.one_hot(truth, ways).flatten() * 2 - 1) * ~roles
    attrs = torch.stack([roles, signs], dim=1).float()
    return model, (pre, label, edges, attrs, roles)


class MetagraphTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_joint_endpoint_and_independent_paths(self):
        model, args = fixture()
        saved = [v.clone() for v in args]
        a, aa = replay(model, args)
        new = [v.clone() for v in args]
        new[0][:2] += torch.randn(2, 8)
        b, bb = replay(model, new)
        donor = bb['native_kqv'].clone()
        donor_saved = donor.clone()
        for k, v in [(donor, None), (None, donor), (donor, donor)]:
            z, audit = replay(model, args, changed_rows=[0, 1], key_donor=k, value_donor=v)
            self.assertTrue(torch.equal(audit['queries'], aa['queries']))
            self.assertTrue(torch.equal(audit['used_kqv'][:, :8], aa['native_kqv'][:, :8]))
            if k is None:
                self.assertTrue(torch.equal(audit['attention'], aa['attention']))
            else:
                # Changed support Q only affects their own incoming attention.
                labels = audit['edges'][1] >= 9
                self.assertTrue(torch.equal(audit['attention'][labels], bb['attention'][labels]))
            if k is not None and v is not None:
                self.assertTrue(torch.equal(audit['labels'], bb['labels']))
                self.assertTrue(torch.equal(z[aa['query_mask']], b[aa['query_mask']]))
            trace = compact_trace(audit, 2, 0, float(model.logit_scale.exp()))
            torch.testing.assert_close(sum(trace['terms'].values()), z[2], atol=1e-5, rtol=0)
        self.assertTrue(torch.equal(donor, donor_saved))
        for old, new in zip(saved, args):
            self.assertTrue(torch.equal(old, new))
        restored, _ = replay(model, args)
        self.assertTrue(torch.equal(a, restored))

    def test_reject_query_transplant_and_training(self):
        model, args = fixture()
        _, a = replay(model, args)
        with self.assertRaises(ValueError):
            replay(model, args, changed_rows=[2], value_donor=a['native_kqv'])
        model.train()
        with self.assertRaises(ValueError):
            replay(model, args)
        model.eval()
        layer = model.layer_list[2].gnn_layers[0]
        for module in [layer, layer.mlp_kqv, layer.att_mlp, layer.bn]:
            self.assertEqual(len(module._forward_hooks), 0)
            self.assertEqual(len(module._forward_pre_hooks), 0)

    def test_radial_factorial_changes_only_intended_value_geometry(self):
        # Two orthogonal directions with different lengths, and an untouched row.
        a = torch.tensor([[7., 8., 9., 10., 2., 0.], [9., 10., 11., 12., 0., 4.]])
        b = torch.tensor([[11., 12., 13., 14., 0., 3.], [13., 14., 15., 16., 5., 0.]])
        direction, radius = radial_value_donors(a, b, [0], 2)
        torch.testing.assert_close(direction, torch.tensor([[7., 8., 9., 10., 0., 2.], [9., 10., 11., 12., 0., 4.]]))
        torch.testing.assert_close(radius, torch.tensor([[7., 8., 9., 10., 3., 0.], [9., 10., 11., 12., 0., 4.]]))
        with self.assertRaises(ValueError):
            radial_value_donors(a, b * 0, [0], 2)


if __name__ == '__main__':
    unittest.main()
