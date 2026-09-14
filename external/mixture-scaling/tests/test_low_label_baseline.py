import numpy as np
from mixture_scaling.probe_low_label_baseline import labeled
class D: pass
def test_labeled_filters_negative_labels():
    import torch
    graph=D(); graph.data=D(); graph.data.x=torch.arange(12).reshape(4,3).float(); graph.data.y=torch.tensor([0,-1,1,-1])
    x,y=labeled(graph); assert x.shape==(2,3); assert np.array_equal(y,np.array([0,1]))
