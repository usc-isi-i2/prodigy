import unittest
import torch
from .support_exceptions import exception_labels


class ExceptionTests(unittest.TestCase):
    def test_nested_balanced_query_fixed(self):
        cells=torch.cat((torch.arange(4).repeat_interleave(5),torch.arange(4).repeat_interleave(6))).repeat(4)
        clean,zero=exception_labels(cells,0,191)
        a,one=exception_labels(cells,1,191)
        b,two=exception_labels(cells,2,191)
        self.assertFalse(zero.any())
        self.assertTrue((~one|two).all())
        for labels,mask,count in ((a,one,1),(b,two,2)):
            self.assertEqual(int(mask.sum()),16*count)
            self.assertTrue(torch.equal(labels.reshape(4,44)[:,20:],clean.reshape(4,44)[:,20:]))
            self.assertTrue((labels.reshape(4,44)[:,:20].sum(1)==10).all())
            for ep in range(4):
                for cell in range(4):
                    self.assertEqual(int(mask[ep*44:ep*44+20][cells[ep*44:ep*44+20]==cell].sum()),count)
        self.assertTrue(torch.equal(cells//2,clean))

    def test_bad_cells(self):
        with self.assertRaises(ValueError): exception_labels(torch.zeros(176,dtype=torch.long),1,1)


if __name__=='__main__': unittest.main()
