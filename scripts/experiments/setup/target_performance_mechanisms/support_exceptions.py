"""Nested, outcome-blind support flips balanced in each joint rule cell."""
import numpy as np
import torch


def exception_labels(cells, count, seed):
    if cells.shape != (176,) or count not in (0,1,2):
        raise ValueError('Four44-point episodes and0/1/2 exceptions per cell required')
    labels = cells // 2
    altered = labels.clone()
    mask = torch.zeros(176,dtype=torch.bool)
    rng = np.random.default_rng(seed)
    for ep in range(4):
        for cell in range(4):
            candidates = torch.where(cells[ep*44:ep*44+20] == cell)[0].numpy()
            if len(candidates)!=5:
                raise ValueError('Five supports per joint cell required')
            selected = rng.permutation(candidates)[:count]+ep*44
            mask[selected] = True
    altered[mask] = 1-altered[mask]
    return altered,mask
