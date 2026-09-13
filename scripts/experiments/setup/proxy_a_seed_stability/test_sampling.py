"""Protect the scientific distinction between repeatability and ID bias."""
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'scripts/experiments/analysis/graphs/structure/graph_divergence'))
sys.path.insert(0,str(ROOT/'scripts/experiments/analysis/graphs/structure_features/path_feature_coupling'))
from compute_graph_divergence import sample_feature_rows
from analyze_neighbor_augmented_features import sample_nonmissing_node_ids


def test_uniform_sampling_removes_sorted_id_bias_and_is_repeatable():
    x=torch.ones((100000,2)); x[::5]=0
    _,legacy,_=sample_feature_rows(x,len(x),4000,np.random.default_rng(0))
    ids,rows=sample_nonmissing_node_ids(x,len(x),4000,np.random.default_rng(0))
    repeat,_=sample_nonmissing_node_ids(x,len(x),4000,np.random.default_rng(0))
    other,_=sample_nonmissing_node_ids(x,len(x),4000,np.random.default_rng(1))
    assert len(ids)==len(np.unique(ids))==4000
    assert np.all(rows!=0) and np.array_equal(ids,repeat)
    assert not np.array_equal(ids,other)
    assert .47 < np.mean(ids/len(x)) < .53
    assert np.mean(legacy/len(x)) < .08
