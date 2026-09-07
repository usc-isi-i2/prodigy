import unittest
import torch
from torch_geometric.data import Data
from .task_switch_batch import build_batch, relabel_batch


class TaskSwitchBatchTests(unittest.TestCase):
    def setUp(self):
        self.graphs = [Data(x=torch.full((3, 768), float(i)),
                            y=torch.tensor([1, 0, -1]),
                            edge_index=torch.tensor([[0, 1], [1, 0]]),
                            edge_index_supernode=torch.tensor([[0, 1], [2, 2]]),
                            supernode=torch.tensor([2]),
                            global_node_ids=torch.tensor([i, i + 176, -1]),
                            center_node_idx=torch.tensor([i]),
                            graph_id=torch.tensor([7, 7, 7])) for i in range(176)]
        self.labels = torch.tensor(([0] * 10 + [1] * 10 + [0] * 12 + [1] * 12) * 4)
        self.table = torch.arange(1536, dtype=torch.float).reshape(2, 768)

    def test_native_layout_and_sequences(self):
        b = build_batch(self.graphs, self.labels, self.table)
        self.assertEqual(len(b), 9)
        self.assertEqual(tuple(b[3].shape), (2, 352))
        self.assertEqual(int(b[5].sum()), 192)
        self.assertTrue(torch.equal(b[4][b[5]], torch.tensor([1., 0.]).repeat(192, 1)))
        for i in range(176):
            e = i // 44
            self.assertEqual(b[3][:, 2*i:2*i+2].tolist(), [[i, i], [176+2*e, 177+2*e]])
            if i % 44 < 20:
                self.assertEqual(b[4][2*i:2*i+2, 1].tolist(),
                                 [1., -1.] if self.labels[i] == 0 else [-1., 1.])
        self.assertEqual(b[6][0, :4].tolist(), [0, 176, 1, 176])
        self.assertEqual(b[7][0, :4].tolist(), [20, 184, 21, 184])
        self.assertEqual(b[8][0, -2:].tolist(), [43, 177])
        self.assertEqual(b[0].source_id_per_task.tolist(), [7]*4)
        self.assertTrue(torch.equal(b[1], self.table.repeat(4, 1)))

    def test_graph_identity_sanitization_and_no_mutation(self):
        a = build_batch(self.graphs, self.labels, self.table)
        b = relabel_batch(a, 1 - self.labels)
        for key, value in a[0]:
            if isinstance(value, torch.Tensor):
                self.assertTrue(torch.equal(value, b[0][key]), key)
        for original, rebuilt in zip(self.graphs, a[0].to_data_list()):
            for key, value in original:
                self.assertTrue(torch.equal(rebuilt[key], torch.zeros_like(value) if key == 'y' else value), key)
            self.assertEqual(original.y.tolist(), [1, 0, -1])
        b[0].x.zero_()
        b[1].zero_()
        self.assertGreater(float(a[0].x.sum()), 0)
        self.assertGreater(float(self.table.sum()), 0)
        self.assertTrue(torch.equal(a[2].argmax(1), self.labels))

    def test_query_relabel_only_changes_scoring_and_gt(self):
        a = build_batch(self.graphs, self.labels, self.table)
        changed = self.labels.clone()
        changed[20] = 1 - changed[20]
        b = relabel_batch(a, changed, require_balance=False)
        for i in (1, 3, 4, 5, 6, 7):
            self.assertTrue(torch.equal(a[i], b[i]), i)
        self.assertFalse(torch.equal(a[2], b[2]))
        self.assertFalse(torch.equal(a[8], b[8]))

    def test_sequences_match_native_collator_identity_permutation(self):
        from data.dataloader import linearize
        b = build_batch(self.graphs, self.labels, self.table)
        mask = b[5][::2].reshape(4, 44)
        selected = b[3][:, b[2].flatten() == 1].reshape(2, 4, 44)
        inputs, targets = selected[0], selected[1]
        support, _ = linearize(~mask, inputs, targets, torch.arange(20).repeat(4, 1))
        query, _ = linearize(mask, inputs, torch.ones_like(targets) * (b[3].max() + 1),
                             torch.arange(24).repeat(4, 1))
        gt, _ = linearize(mask, inputs, targets, torch.arange(24).repeat(4, 1))
        for actual, expected in zip(b[6:], (support, query, gt)):
            self.assertTrue(torch.equal(actual, expected))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError): build_batch(self.graphs[:-1], self.labels, self.table)
        with self.assertRaises(ValueError): build_batch(self.graphs, self.labels.float(), self.table)
        with self.assertRaises(ValueError): build_batch(self.graphs, torch.zeros(176, dtype=torch.long), self.table)
        labels = self.labels.clone(); labels[0] = 2
        with self.assertRaises(ValueError): build_batch(self.graphs, labels, self.table)


if __name__ == '__main__':
    unittest.main()
