"""Regression for fresh support graphs mixed with unbatched canonical graphs."""
import torch
from torch_geometric.data import Data, Batch
from .run import align_scalar_attributes

def test_mixed_scalar_metadata():
    graphs=[Data(x=torch.tensor([[float(i)],[0.]]),edge_index=torch.tensor([[0],[1]]),
                 center_node_idx=i,global_node_ids=torch.tensor([i,-1])) for i in range(3)]
    original=Batch.from_data_list(graphs)
    parts=original.to_data_list()
    parts[1]=align_scalar_attributes(graphs[1],parts[1])
    restored=Batch.from_data_list(parts)
    for key,value in original:
        if isinstance(value,torch.Tensor):torch.testing.assert_close(restored[key],value,rtol=0,atol=0)
    changed=graphs[1].clone();changed.center_node_idx=99;changed.global_node_ids[0]=99
    parts[1]=align_scalar_attributes(changed,parts[1])
    replaced=Batch.from_data_list(parts)
    assert replaced.center_node_idx.tolist()==[0,99,2]
    assert changed.center_node_idx==99
