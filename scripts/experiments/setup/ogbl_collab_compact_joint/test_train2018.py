import unittest
import numpy as np
from train2018 import grid


class Train2018Tests(unittest.TestCase):
    def test_grid_complete_and_baseline_control(self):
        panel = {'bp':np.array([2., .5]), 'bn':np.ones(100),
                 'official_bp':np.array([3., 2.]), 'official_bn':np.ones(100)}
        scores = {'p':np.array([1., -1.]), 'n':np.zeros(100)}
        rows = grid(panel,scores)
        self.assertEqual(len(rows),40)
        self.assertEqual(rows[0],dict(base='frozen2015',alpha=0.,hits=.5))
        self.assertEqual(rows[20],dict(base='validation2018',alpha=0.,hits=1.))
        np.testing.assert_array_equal(scores['p'],[1.,-1.])


if __name__=='__main__':
    unittest.main()
