import unittest
import torch
from torch_geometric.data import Data, Batch
from .cache_matched_relations import rebuild_batch


def sample(i, extra=False):
    ids=torch.tensor([i, -1]) if not extra else torch.tensor([i,2,-1])
    n=len(ids)
    x=torch.zeros(n,2); x[0]=i+1
    y=torch.zeros(n,dtype=torch.long); y[0]=i
    return Data(x=x,y=y,global_node_ids=ids,edge_index=torch.empty(2,0,dtype=torch.long),
        edge_attr=torch.empty(0,1),center_node_idx=i,num_nodes=n,
        supernode=torch.tensor([n-1]),edge_index_supernode=torch.tensor([[0],[n-1]]),
        edge_index_from_supernode=torch.tensor([[n-1],[0]]))


class MatchedCacheTests(unittest.TestCase):
    def batch(self):
        g=Batch.from_data_list([sample(0),sample(1)])
        for k,v in dict(task_id_per_sample=[0,0],task_id_per_query=[0],source_id_per_task=[-1],
                        task_label_map=[[0,1]]).items(): g[k]=torch.tensor(v)
        return [g,torch.tensor([1.,2.]),torch.eye(2)]

    def test_preserves_tasks_when_neighborhood_size_changes(self):
        old=self.batch()
        new=rebuild_batch(old,{0:sample(0,True),1:sample(1,True)})
        self.assertEqual(new[0].ptr.tolist(),[0,3,6])
        for a,b in zip(old[1:],new[1:]): self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(new[0].task_label_map,old[0].task_label_map))

    def test_changed_center_or_unknown_metadata_rejected(self):
        old=self.batch()
        with self.assertRaises(ValueError): rebuild_batch(old,{0:sample(1),1:sample(0)})
        old[0].unknown=torch.tensor([1])
        with self.assertRaises(ValueError): rebuild_batch(old,{0:sample(0),1:sample(1)})


if __name__ == '__main__': unittest.main()
