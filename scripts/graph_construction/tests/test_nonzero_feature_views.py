import importlib.util
from pathlib import Path
import unittest
import torch
from torch_geometric.data import Data

spec = importlib.util.spec_from_file_location('views', Path(__file__).parents[1] / 'nonzero_feature_views.py')
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)

class NonzeroViewsTest(unittest.TestCase):
    def test_induced_all_fields_and_isolate(self):
        x = torch.tensor([[0., 0.], [1., 0.], [0., 2.], [0., 0.], [1e-20, 0.]])
        edges = torch.tensor([[0, 1, 2, 1, 3], [1, 2, 1, 3, 4]])
        attrs = torch.arange(5).float().reshape(-1, 1)
        y = torch.tensor([-1, 0, 1, -1, 0])
        mask = torch.tensor([False, True, False, False, True])
        graph = dict(x=x, edge_index=edges, edge_attr=attrs, y=y,
                     user_ids=['a', 'b', 'c', 'd', 'e'], u2i=dict(zip('abcde', range(5))),
                     handles=list('ABCDE'), node_targets={'f': torch.arange(5.)},
                     split_masks={'train': mask}, edge_index_views={'history': edges[:, :3]},
                     edge_attr_views={'history': attrs[:3]}, target_edge_index_views={'future': edges[:, 3:]},
                     static_split_stats={'total_edges': 5},
                     data=Data(x=x, edge_index=edges, edge_attr=attrs, y=y, train_mask=mask))
        ids = v.nonzero_ids(x)
        self.assertEqual(ids.tolist(), [1, 2, 4])
        out = v.induced(graph, ids)
        v.verify_induced(graph, out, ids)
        self.assertEqual(out['edge_index'].tolist(), [[0, 1], [1, 0]])
        self.assertEqual(out['edge_attr'].flatten().tolist(), [1., 2.])
        self.assertEqual(out['user_ids'], ['b', 'c', 'e'])
        self.assertEqual(out['u2i'], {'b': 0, 'c': 1, 'e': 2})
        self.assertEqual(out['data'].train_mask.tolist(), [True, False, True])
        self.assertEqual(out['node_targets']['f'].tolist(), [1., 2., 4.])
        self.assertEqual(out['target_edge_index_views']['future'].shape[1], 0)
        self.assertEqual(v.statistics(out)['isolated_nodes'], 1)
        self.assertEqual(graph['static_split_stats']['total_edges'], 5)
        self.assertNotIn('static_split_stats', out)
        self.assertEqual(graph['x'].shape[0], 5)

    def test_nonfinite_fails_explicitly(self):
        with self.assertRaises(ValueError):
            v.nonzero_ids(torch.tensor([[float('nan')]]))

    def test_unknown_tensor_fails(self):
        with self.assertRaises(ValueError):
            v.induced({'x': torch.ones(3, 2), 'mystery': torch.ones(4)}, torch.tensor([0, 2]))

if __name__ == '__main__':
    unittest.main()
