"""Synthetic end-to-end tests, runnable without private data or research code."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch_geometric.data import Data, Batch

from graph_role.model import make_model, PARAM_KEYS
from graph_role.io import pack_batch, unpack_batch, checked_path, sha, labels_for
from graph_role.replay import batch_hash, clone_batch
from graph_role.role_context import query_mask
from graph_role.role_topology import topology_batch, swap_edges
from graph_role.support_dose import dose_masks
from graph_role.predict import predict_batch, expected_conditions
from graph_role.message_content import aggregation_audit, message_control
from graph_role.run import verify_code


def fixture():
    torch.manual_seed(21)
    graphs = []
    for i in range(8):
        cycle = [(j, (j+1) % 9) for j in range(9)]
        edges = torch.tensor(cycle + [(b, a) for a, b in cycle]).T.contiguous()
        graphs.append(Data(x=torch.randn(10, 8), edge_index=edges, edge_attr=torch.zeros(18, 1),
            supernode=torch.tensor([9]), edge_index_supernode=torch.tensor([[0], [9]]),
            global_node_ids=torch.tensor([i*9+j for j in range(9)]+[-1])))
    g = Batch.from_data_list(graphs)
    g.task_id_per_sample = torch.arange(2).repeat_interleave(4)
    g.task_label_map = torch.tensor([[0, 1], [1, 0]])
    labels = torch.tensor([0, 1, 0, 1]*2)
    query = torch.tensor([False, False, True, True]*2)
    edges = torch.stack((torch.arange(8).repeat_interleave(2),
        8 + g.task_id_per_sample.repeat_interleave(2)*2 + torch.arange(2).repeat(8)))
    onehot = torch.nn.functional.one_hot(labels, 2).float()
    attrs = torch.stack((query.repeat_interleave(2).float(), ((onehot*2-1)*~query[:, None]).reshape(-1)), 1)
    batch = [g, torch.randn(4, 8), onehot, edges, attrs, query.repeat_interleave(2),
             torch.empty(0), torch.empty(0), torch.empty(0)]
    params = {'layers': 'S,U,M', 'emb_dim': 8, 'gnn_type': 'sage', 'dropout': 0., 'reset_after_layer': None,
              'has_final_back': False, 'meta_gnn_pos_only': False, 'no_bn_metagraph': False, 'no_bn_encoder': False,
              'text_features_dropout': 0., 'zero_shot': False, 'skip_path': False,
              'ignore_label_embeddings': False, 'zero_label_embeddings': False, 'task_name': 'classification'}
    return make_model({'params': params, 'feature_dim': 8, 'label_dim': 8}), batch


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_package_hashes(self):
        self.assertEqual(len(verify_code()), 64)

    def test_full_panel_roundtrip_and_roles(self):
        model, batch = fixture()
        portable = unpack_batch(pack_batch(batch))
        out, checks = predict_batch(model, portable, target='synthetic', stream='original', batch_index=0, check_direct=True)
        other, _ = predict_batch(model, batch, target='synthetic', stream='original', batch_index=0)
        self.assertEqual(set(out), expected_conditions())
        self.assertEqual(len(out), 45)
        self.assertEqual(checks, {'query_pre_exact': 9, 'query_post_exact': 14, 'direct_role_checks': 37})
        for k in out:
            torch.testing.assert_close(out[k], other[k], rtol=0, atol=0)
        for role, qc, sc in [('query', 'removed', 'intact'), ('support', 'intact', 'removed'), ('both', 'removed', 'removed')]:
            torch.testing.assert_close(out[f'message/zero_messages/{role}'], out[f'topology/{qc}/{sc}/0'], rtol=0, atol=0)

    def test_query_labels_do_not_enter_predictions(self):
        model, batch = fixture()
        changed = clone_batch(batch)
        q = query_mask(batch)
        changed[2][q] = changed[2][q].flip(1)
        a, _ = predict_batch(model, batch, target='synthetic', stream='original', batch_index=0)
        b, _ = predict_batch(model, changed, target='synthetic', stream='original', batch_index=0)
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)

    def test_tensor_only_serialization(self):
        _, batch = fixture()
        with tempfile.TemporaryDirectory() as name:
            p = Path(name)/'input.pt'
            torch.save(pack_batch(batch), p)
            restored = unpack_batch(torch.load(p, weights_only=True))
            self.assertEqual(batch_hash(restored), batch_hash(unpack_batch(pack_batch(batch))))

    def test_nested_label_blind_doses_and_query_edges(self):
        _, batch = fixture()
        masks, stats = dose_masks(batch, seed=81)
        qedges = query_mask(batch)[batch[0].batch[batch[0].edge_index[0]]]
        self.assertTrue(all(m[qedges].all() for m in masks.values()))
        self.assertTrue(all(not torch.any(masks[b] & ~masks[a]) for a,b in zip((0,25,50,75),(25,50,75,100))))
        self.assertEqual(stats[100]['retained_support_edges'], 0)
        changed = clone_batch(batch)
        changed[2] = changed[2].flip(1)
        other, _ = dose_masks(changed, seed=81)
        for d in masks:
            torch.testing.assert_close(masks[d], other[d], rtol=0, atol=0)

    def test_degree_preserving_null_empty_rigid_and_parallel(self):
        _, batch = fixture()
        edge = batch[0].edge_index[:, :18]
        changed, receipt = swap_edges(edge, seed=9)
        self.assertGreater(receipt['changed_edges'], 0)
        for axis in (0,1):
            torch.testing.assert_close(edge[axis].sort().values, changed[axis].sort().values, rtol=0, atol=0)
        for edge in (torch.empty(2,0,dtype=torch.long), torch.tensor([[0,0,0],[1,2,3]])):
            _, receipt = swap_edges(edge, seed=9)
            self.assertEqual(receipt['accepted_swaps'], 0)
        with self.assertRaises(ValueError):
            swap_edges(torch.tensor([[0,0],[1,1]]), seed=0)

    def test_operator_restored_after_exception(self):
        model, batch = fixture()
        before = aggregation_audit(model)
        with self.assertRaises(RuntimeError):
            with message_control(model, batch[0], 'actual_mean'):
                self.assertEqual(aggregation_audit(model)[0]['unit_message_probe'], [1.,1.,0.])
                raise RuntimeError('test')
        self.assertEqual(aggregation_audit(model), before)

    def test_tampered_input_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)
            p=root/'value'; p.write_text('original')
            r={'path':'value','sha256':sha(p)}
            self.assertEqual(checked_path(root,r),p.resolve())
            p.write_text('changed')
            with self.assertRaises(ValueError): checked_path(root,r)
            with self.assertRaises(ValueError): checked_path(root,{'path':'../escape','sha256':'x'})

    def test_bad_recipe_rejected(self):
        model, _ = fixture()
        params=dict(model.params, has_final_back=True)
        with self.assertRaises(ValueError): make_model({'params':params,'feature_dim':8,'label_dim':8})


if __name__ == '__main__':
    unittest.main()
