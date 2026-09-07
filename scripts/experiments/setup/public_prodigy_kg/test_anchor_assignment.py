import unittest
import torch
from .run_anchor_assignment import assignment,EPISODES,WAYS


class AssignmentTest(unittest.TestCase):
    def test_confined_and_restored_even_on_error(self):
        table=torch.nn.Parameter(torch.arange(90,dtype=torch.float32).reshape(30,3))
        original=table.detach().clone()
        with assignment(table,False):
            self.assertTrue(torch.equal(table,original))
        with self.assertRaises(RuntimeError):
            with assignment(table,True):
                self.assertTrue(torch.equal(table[:20],original[:20].flip(0)))
                self.assertTrue(torch.equal(table[20:],original[20:]))
                raise RuntimeError("test")
        self.assertTrue(torch.equal(table,original))
        self.assertTrue(table.requires_grad)
        self.assertEqual((EPISODES,WAYS),(32,20))


if __name__=="__main__":
    unittest.main()
