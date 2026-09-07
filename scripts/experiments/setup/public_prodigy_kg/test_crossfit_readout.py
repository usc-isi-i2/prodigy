import unittest
import torch
from .crossfit_readout import hide_support
from .test_readout_episode import fixture


class HideTests(unittest.TestCase):
    def test_all_label_paths_and_no_mutation(self):
        x, y, edges, query = fixture(20, 1)
        attrs = torch.stack([query.float(), (2*y.flatten()-1)*(~query)], 1)
        args = (x, torch.ones(20,20), y, edges, attrs, query, torch.ones(1,120))
        held = torch.arange(20)*7
        changed = hide_support(args, held)
        self.assertTrue(changed[5].reshape(140,20)[held].all())
        self.assertTrue((changed[4].reshape(140,20,2)[held,:,1]==0).all())
        self.assertTrue((changed[2][held]==0).all())
        self.assertTrue((changed[6]==0).all())
        self.assertTrue((y[held].sum(1)==1).all())
        self.assertFalse(query.reshape(140,20)[held].any())
        keep = torch.ones(140,dtype=torch.bool); keep[held]=False
        torch.testing.assert_close(changed[4].reshape(140,20,2)[keep], attrs.reshape(140,20,2)[keep])
        with self.assertRaises(ValueError):
            hide_support(args, held+3)


if __name__ == "__main__":
    unittest.main()
