import unittest
from unittest.mock import patch
import numpy as np
import candidate


class CandidateTests(unittest.TestCase):
    def test_negative_sampler_deterministic_and_excludes_validation(self):
        n=10000
        pos=np.array([[1,2],[3,4]])
        blocked=candidate.joint.gate.negatives(pos,n,20262017,count=100100)[:20]
        a=candidate.fresh_negatives(pos,blocked,n,2017)
        b=candidate.fresh_negatives(pos,blocked,n,2017)
        np.testing.assert_array_equal(a,b)
        keys=candidate.joint.gate.keys
        self.assertEqual(len(a),100000)
        self.assertEqual(len(np.intersect1d(keys(a,n),keys(blocked,n))),0)
        self.assertEqual(len(np.intersect1d(keys(a,n),keys(pos,n))),0)

    def test_contract_does_not_select_on_test(self):
        c=candidate.contract()
        self.assertFalse(c['test_selection'])
        self.assertEqual(c['select_validation_year'],2018)
        self.assertLess(c['total_inference_scalars'],1000000)


if __name__=='__main__': unittest.main()
