import unittest
import numpy as np
from .analyze_twibot_cue_alignment import episode_alignment


class CueAlignmentTests(unittest.TestCase):
    def test_rank_and_episode_local_label_flip_invariance(self):
        x = np.tile(np.array([[0., -2], [0, -1], [0, 1], [0, 2]]), (4, 1))
        matched = episode_alignment(x, 3*x)
        self.assertTrue(all(abs(r["spearman"] - 1) < 1e-12 for r in matched))
        self.assertTrue(all(r["decision_agreement"] == 1 for r in matched))
        inverse = episode_alignment(x, -x)
        self.assertTrue(all(abs(r["spearman"] + 1) < 1e-12 for r in inverse))
        a, b = x.copy(), 3*x.copy()
        a[:4] = a[:4, ::-1]
        b[:4] = b[:4, ::-1]
        self.assertEqual(episode_alignment(a, b), matched)

    def test_constants_counted_as_missing_not_zero_agreement(self):
        x = np.zeros((16, 2))
        self.assertEqual(episode_alignment(x, x), [None]*4)
        with self.assertRaises(ValueError):
            episode_alignment(x, x[:8])


if __name__ == "__main__":
    unittest.main()
