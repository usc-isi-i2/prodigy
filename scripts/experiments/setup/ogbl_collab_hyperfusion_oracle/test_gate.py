import importlib.util
from pathlib import Path
import sys
import unittest, numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SPEC = importlib.util.spec_from_file_location("pair_gate", HERE / "gate.py")
MOD = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(MOD)
fold_ids, features, FOLDS = MOD.fold_ids, MOD.features, MOD.FOLDS

class GateTest(unittest.TestCase):
    def test_fold_is_undirected_and_bounded(self):
        a=np.array([[1,9],[3,4]]); b=a[:,::-1]
        np.testing.assert_array_equal(fold_ids(a,10),fold_ids(b,10))
        self.assertTrue(((fold_ids(a,10)>=0)&(fold_ids(a,10)<FOLDS)).all())

    def test_feature_shape(self):
        p=np.array([[1.,2.],[2.,1.],[3.,4.]])
        n=np.array([[0.],[1.],[2.]])
        panel={"pfeatures":np.zeros((2,13)),"nfeatures":np.zeros((1,13)),
               "psymmetric":np.zeros((2,14)),"nsymmetric":np.zeros((1,14))}
        xp,xn=features(p,n,panel); self.assertEqual(xp.shape,(2,33)); self.assertEqual(xn.shape,(1,33))

if __name__=="__main__": unittest.main()
