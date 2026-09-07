import unittest
import torch
from .run import assemble,CONDITIONS


class GILTCrossoverTest(unittest.TestCase):
    def test_exhaustive_ownership_fixed_projection(self):
        a={"feature_projection":torch.ones(2,2),"encoder.bn.running_mean":torch.ones(2),"predictor.weight":torch.ones(2)*2}
        b={k:v.clone() for k,v in a.items()};b["encoder.bn.running_mean"].fill_(9);b["predictor.weight"].fill_(8)
        hybrid,parts=assemble(a,b,100,900)
        self.assertTrue(torch.equal(hybrid["encoder.bn.running_mean"],a["encoder.bn.running_mean"]))
        self.assertTrue(torch.equal(hybrid["predictor.weight"],b["predictor.weight"]))
        self.assertEqual(sum(len(v) for v in parts.values()),3)
        with self.assertRaises(ValueError):assemble(a|{"extra":torch.ones(1)},b|{"extra":torch.ones(1)},100,900)
        with self.assertRaises(ValueError):assemble(a,{k:v for k,v in b.items() if k!="predictor.weight"},100,900)
        with self.assertRaises(ValueError):assemble(a,b|{"predictor.weight":b["predictor.weight"].double()},100,900)
        with self.assertRaises(ValueError):assemble(a,b,300,900)
        with self.assertRaises(ValueError):assemble(a,b,100,300)
        b["feature_projection"].fill_(0)
        with self.assertRaises(ValueError):assemble(a,b,100,900)
        self.assertEqual(CONDITIONS,((100,100),(100,900),(900,100),(900,900)))


if __name__=="__main__":unittest.main()
