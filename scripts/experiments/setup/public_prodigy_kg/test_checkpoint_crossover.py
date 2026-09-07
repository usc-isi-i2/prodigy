import unittest
import torch
from .run_checkpoint_crossover import partition,assemble,CONDITIONS,EPISODES


class CrossoverTest(unittest.TestCase):
    def test_exhaustive_partition_and_buffers(self):
        keys=["layer_list.0.bn.running_mean","layer_list.1.weight","initial_input_mlp.weight",
              "layer_list.2.bn.running_var","initial_label_mlp.weight","learned_label_embedding.weight",
              "final_input_mlp.weight","final_label_mlp.weight","logit_scale"]
        early={k:torch.tensor([2.]) for k in keys}
        late={k:torch.tensor([8.]) for k in keys}
        parts=partition(early)
        self.assertEqual(set(sum(parts.values(),[])),set(keys))
        hybrid=assemble(early,late)
        for k in parts["encoder"]:
            self.assertEqual(hybrid[k].item(),2)
        for k in parts["inference"]:
            self.assertEqual(hybrid[k].item(),8)
        hybrid[keys[0]].fill_(0)
        self.assertEqual(early[keys[0]].item(),2)
        with self.assertRaises(ValueError):
            partition(early|{"unclassified.weight":torch.ones(1)})
        with self.assertRaises(ValueError):
            assemble(early,late|{"extra":torch.ones(1)})
        self.assertEqual(len(CONDITIONS),4)
        self.assertEqual(EPISODES,128)


if __name__=="__main__":
    unittest.main()
