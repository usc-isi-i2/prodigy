import unittest
import torch
from .evaluate_matched_relations import metrics


class MatchedMetricsTests(unittest.TestCase):
    def test_semantic_tie_breaking_and_episode_auc(self):
        labels=dict(local_y=torch.tensor([0,1]),mapping=torch.tensor([[1,0],[1,0]]),episode_ids=torch.tensor([0,0]))
        got=metrics(torch.zeros(2,2),labels)
        self.assertEqual(got['accuracy'],.5)
        self.assertEqual(got['f1'],0)
        self.assertEqual(got['mean_episode_auc'],.5)
        self.assertEqual(got['roc_auc'],.5)

    def test_perfect_reversed_semantic_mapping(self):
        labels=dict(local_y=torch.tensor([0,1]),mapping=torch.tensor([[1,0],[1,0]]),episode_ids=torch.tensor([0,0]))
        got=metrics(torch.tensor([[2.,0.],[0.,2.]]),labels)
        for key in ('accuracy','f1','mean_episode_auc','roc_auc'): self.assertEqual(got[key],1)


if __name__=='__main__': unittest.main()
