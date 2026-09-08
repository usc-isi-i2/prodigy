import unittest
import numpy as np
import torch
from .test_role_topology import cycle_fixture
from .support_boundary_episode import evaluate_episode, outcomes
from .support_boundary_folds import hide_support_labels
from .replay import batch_hash, clone_batch
from .verify_member_training import model_digest


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model, b = cycle_fixture()
        self.model.layer_list[0].module_list[0].lin_edge_attr = None
        b[0].task_id_per_sample = torch.zeros(8, dtype=torch.long)
        b[0].task_label_map = torch.tensor([[0, 1]])
        b[1] = b[1][:2]
        y = torch.tensor([0, 0, 0, 1, 1, 1, 0, 1])
        q = torch.tensor([False] * 6 + [True] * 2)
        b[2] = torch.nn.functional.one_hot(y, 2).float()
        b[3] = torch.stack((torch.arange(8).repeat_interleave(2), torch.tensor([8, 9] * 8)))
        b[4] = torch.stack((q.repeat_interleave(2).float(), ((b[2]*2-1)*~q[:, None]).flatten()), 1)
        b[5] = q.repeat_interleave(2)
        self.batch = b

    def test_end_to_end_and_query_label_blindness(self):
        before, weights = batch_hash(self.batch), model_digest(self.model.state_dict())
        a = evaluate_episode(self.model, self.batch)
        other = clone_batch(self.batch)
        other[2][6:] = other[2][6:].flip(1)
        b = evaluate_episode(self.model, other)
        self.assertEqual(len(a['logits']), 6)
        for key in a['logits']:
            self.assertEqual(tuple(a['logits'][key].shape), (2, 2))
            self.assertTrue(torch.isfinite(a['logits'][key]).all())
            torch.testing.assert_close(a['logits'][key], b['logits'][key], rtol=0, atol=0)
        self.assertEqual(a['calibration'], b['calibration'])
        self.assertTrue(all(np.isfinite(v).all() for v in a['support_oof_margins'].values()))
        self.assertEqual(before, batch_hash(self.batch))
        self.assertEqual(weights, model_digest(self.model.state_dict()))

    def test_actual_forward_hidden_label_invariance(self):
        other = clone_batch(self.batch)
        other[2][[0, 3]] = other[2][[0, 3]].flip(1)
        for name, value in outcomes(self.model, hide_support_labels(self.batch, [0, 3])).items():
            changed = outcomes(self.model, hide_support_labels(other, [0, 3]))[name]
            torch.testing.assert_close(value, changed, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
