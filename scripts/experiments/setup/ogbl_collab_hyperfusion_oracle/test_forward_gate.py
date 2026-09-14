import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from forward_gate import percentile
class TestForward(unittest.TestCase):
    def test_frozen_cdf(self):
        np.testing.assert_array_equal(percentile(np.array([1.,3.]),np.array([2.,4.]),np.array([0.,1.,2.5,5.])),np.array([0.,.25,.5,1.]))
if __name__=="__main__": unittest.main()
