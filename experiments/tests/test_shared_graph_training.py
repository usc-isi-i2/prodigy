"""Correctness gates for shared-data training; no local model training."""
import copy
from types import SimpleNamespace
import random

import numpy as np
import unittest
import tempfile
from pathlib import Path
import torch
from torch_geometric.data import Data

from data.dataset import SubgraphDataset
from data.dataloader import NeighborTask, ParamSampler
from experiments.sampler import NeighborSampler
from experiments.params import get_params
from experiments.run_shared_graph import (REPO, make_plan, prepare_shared_dataset,
                                          shared_storage_report, validate_configs,
                                          validate_disjoint_sources)


def tiny_dataset():
    # Two separate 12-node cliques, enough distinct positives for real 3+4 episodes.
    edges = [(i, j) for offset in (0, 12) for i in range(offset, offset+12)
             for j in range(offset, offset+12) if i != j]
    graph = Data(x=torch.arange(24*4).reshape(24,4).float(),
                 edge_index=torch.tensor(edges).T.contiguous(), y=torch.zeros(24).long(),
                 graph_id=torch.tensor([0]*12+[1]*12), num_nodes=24,
                 source_graph_names=['a', 'b'], user_ids=[str(i) for i in range(24)])
    return SubgraphDataset(graph, NeighborSampler(graph, 2, hop_sizes=[9,9], limit=101, walk_hops=1))


def _inspect_spawned(dataset, queue):
    queue.put((shared_storage_report(dataset), dataset.graph.x[0].tolist()))


def ladder_config():
    return str(REPO/'scripts/experiments/setup/nm_ladder_nhop2/configs/train_ordA_r2.yaml')


class SharedGraphTests(unittest.TestCase):
    def test_shared_source_pools_preserve_episode_stream_and_boundaries(self):
        dataset = tiny_dataset()
        old_graph = dataset.graph
        prepare_shared_dataset(dataset)
        assert old_graph.user_ids == [str(i) for i in range(24)]
        assert dataset.graph.user_ids == []
        assert all(shared_storage_report(dataset).values())
        for source in (0,1):
            array = dataset.source_node_pools[source].numpy()
            common = dict(neighbor_sampler=dataset.neighbor_sampler, size=24,
                          direction='inout', confine_to_single_stratum=True, stratum_weighting='balanced')
            legacy = NeighborTask(strata=[array.tolist()], **common)
            shared = NeighborTask(strata=[array], **common)
            assert shared.strata[0] is array
            for seed in range(4):
                torch.manual_seed(seed)
                first = legacy.sample(2,7,3,4,random.Random(seed))
                torch.manual_seed(seed)
                second = shared.sample(2,7,3,4,random.Random(seed))
                assert first == second
                for members in second.values():
                    for member in members:
                        sampled = dataset[member]
                        assert set(sampled.graph_id[:-1].tolist()) == {source}



    def test_spawned_process_receives_shared_graph_without_copy(self):
        dataset = prepare_shared_dataset(tiny_dataset())
        ctx = torch.multiprocessing.get_context('spawn')
        queue = ctx.Queue()
        child = ctx.Process(target=_inspect_spawned, args=(dataset, queue))
        child.start()
        try:
            report, features = queue.get(timeout=30)
            assert all(report.values())
            assert features == dataset.graph.x[0].tolist()
        finally:
            child.join(timeout=30)
            if child.is_alive():
                child.terminate()
                child.join()
            queue.close()
        assert child.exitcode == 0


    def test_cross_source_edge_rejected(self):
        graph = tiny_dataset().graph
        graph.edge_index = torch.cat([graph.edge_index, torch.tensor([[0],[12]])], dim=1)
        with self.assertRaisesRegex(ValueError, 'cross-source'):
            validate_disjoint_sources(graph, chunk_size=5)



    def test_plan_enforces_worker_budget_and_preserves_training_budget(self):
        tmp_path = Path(tempfile.gettempdir()) / "shared-plan-only"
        args = SimpleNamespace(configs=[ladder_config()]*8, gpus=[2], models_per_gpu=8,
                               worker_budget=32, workers_per_model=None, smoke_steps=0,
                               run_dir=tmp_path)
        plan, workers = make_plan(args, [])
        assert workers == 4
        assert len({p['exp_name'] for p in plan}) == 8
        assert all(p['epochs']*p['dataset_len_cap'] == 40000 for p in plan)
        assert all(str(p['device']) == 'cuda:2' and not p['detect_anomaly'] for p in plan)
        args.workers_per_model = 16
        with self.assertRaisesRegex(ValueError, 'budget'):
            make_plan(args, [])


    def test_mismatched_graph_or_new_unknown_data_option_rejected(self):
        p = get_params(['--config', ladder_config()])
        for key, value in [('graph_filename','different.pt'), ('edge_view','static_background'),
                           ('some_future_data_option', True)]:
            other = copy.deepcopy(p)
            other[key] = value
            with self.assertRaisesRegex(ValueError, 'settings'):
                validate_configs([p, other])
        other = copy.deepcopy(p)
        other.update(prefix='different', seed=9, neighbor_sampling_source_subset='covid')
        validate_configs([p, other])


if __name__ == "__main__":
    unittest.main()
