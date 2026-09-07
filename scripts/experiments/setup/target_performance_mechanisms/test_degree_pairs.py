import unittest
import numpy as np
from .decompose_degree_pairs import pair_accounting


class DegreePairsTests(unittest.TestCase):
    def test_partition_and_ties(self):
        rows, a, b = pair_accounting([1,1,0,0], [0,1,0,2], [1,0,1,2],
                                     [1,1,0,0], [0,0,0,0])
        self.assertEqual(sum(r["pairs"] for r in rows), 4)
        self.assertEqual((a,b), (1,.5))
        self.assertEqual(sum(r["change_contribution"] for r in rows), -.5)
        self.assertEqual(sum(r["lost_contribution"] for r in rows), .5)

    def test_class_orientation_invariance(self):
        y=np.array([0,1,0,1]); d=np.array([0,2,1,0]); c=np.array([0,.5,.2,-.3])
        a=np.array([.1,.3,.4,.6]); b=np.array([.5,.2,.1,.8])
        first=pair_accounting(y,d,c,a,b)
        second=pair_accounting(1-y,d,-c,1-a,1-b)
        self.assertEqual(first,second)

    def test_different_degree_cue_ties_are_retained(self):
        rows,_,_=pair_accounting([0,1],[0,1],[0,0],[0,1],[1,0])
        self.assertEqual(rows[-1]["mass"],1)
        self.assertEqual(rows[-1]["change_contribution"],-1)

    def test_invalid_inputs(self):
        for y,d,c,a,b in [([1,1],[0,1],[0,1],[0,1],[0,1]),
                           ([0,1],[0,.5],[0,1],[0,1],[0,1]),
                           ([0,1],[0,1],[0,1],[0,float("nan")],[0,1])]:
            with self.assertRaises(ValueError): pair_accounting(y,d,c,a,b)


if __name__ == "__main__": unittest.main()
