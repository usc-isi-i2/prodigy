from __future__ import annotations

import random

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    N_QUERY,
    N_SHOT,
    N_WAY,
    iter_episodes,
    new_fingerprint,
    reset_episode_rng,
    update_episode_fingerprint,
)


def _synthetic_batch():
    data_list = []
    labels = []
    query = []
    for task in range(2):
        for label in range(N_WAY):
            for member in range(N_SHOT + N_QUERY):
                data_list.append(
                    Data(
                        x=torch.randn(4, 768),
                        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
                        global_node_ids=torch.tensor([1000 + len(data_list), 2, 3, -1]),
                        center_node_idx=torch.tensor(1000 + len(data_list)),
                        num_nodes=4,
                    )
                )
                labels.append(label)
                query.append(member >= N_SHOT)
    graphs = Batch.from_data_list(data_list)
    samples_per_task = N_WAY * (N_SHOT + N_QUERY)
    graphs.task_id_per_sample = torch.arange(2).repeat_interleave(samples_per_task)
    label_tensor = torch.nn.functional.one_hot(torch.tensor(labels), N_WAY).float()
    edge_mask = torch.tensor(query).repeat_interleave(N_WAY)
    filler = torch.empty(0)
    return (graphs, filler, label_tensor, filler, filler, edge_mask, filler, filler, filler)


def _synthetic_classification_batch():
    n_way, n_shot, n_query = 2, 10, 1
    data_list, labels, query = [], [], []
    for label in range(n_way):
        for member in range(n_shot + n_query):
            data_list.append(
                Data(
                    x=torch.randn(4, 768),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
                    global_node_ids=torch.tensor([2000 + len(data_list), 2, 3, -1]),
                    center_node_idx=torch.tensor(2000 + len(data_list)),
                    num_nodes=4,
                )
            )
            labels.append(label)
            query.append(member >= n_shot)
    graphs = Batch.from_data_list(data_list)
    graphs.task_id_per_sample = torch.zeros(len(data_list), dtype=torch.long)
    label_tensor = torch.nn.functional.one_hot(torch.tensor(labels), n_way).float()
    edge_mask = torch.tensor(query).repeat_interleave(n_way)
    filler = torch.empty(0)
    batch = (graphs, filler, label_tensor, filler, filler, edge_mask, filler, filler, filler)
    return batch, n_way, n_shot, n_query


def test_episode_extraction_removes_pooling_nodes_and_preserves_counts():
    episodes = list(iter_episodes(_synthetic_batch()))
    assert len(episodes) == 2
    samples_per_task = N_WAY * (N_SHOT + N_QUERY)
    for task, episode in enumerate(episodes):
        assert episode.x.shape == (N_WAY * (N_SHOT + N_QUERY) * 3, 768)
        assert episode.centers.numel() == N_WAY * (N_SHOT + N_QUERY)
        assert int(episode.support_mask.sum()) == N_WAY * N_SHOT
        assert int(episode.query_mask.sum()) == N_WAY * N_QUERY
        assert (episode.centers >= 0).all()
        expected = torch.arange(
            1000 + task * samples_per_task,
            1000 + (task + 1) * samples_per_task,
        )
        assert torch.equal(episode.global_centers, expected)
        assert episode.global_node_ids is not None
        assert (episode.global_node_ids >= 0).all()


def test_episode_rng_reset_restores_all_sampling_streams():
    reset_episode_rng()
    first = (random.random(), np.random.random(), torch.rand(1).item())

    random.random()
    np.random.random()
    torch.rand(100)
    reset_episode_rng()
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second


def test_episode_fingerprint_uses_global_sampled_node_ids():
    episode, *_ = iter_episodes(_synthetic_batch())
    first = new_fingerprint()
    update_episode_fingerprint(first, episode)
    episode.global_node_ids[0] += 1
    second = new_fingerprint()
    update_episode_fingerprint(second, episode)
    assert first.hexdigest() != second.hexdigest()


def test_episode_extraction_supports_downstream_classification_shape():
    batch, n_way, n_shot, n_query = _synthetic_classification_batch()
    episode, = iter_episodes(batch, n_way=n_way, n_shot=n_shot, n_query=n_query)
    assert episode.n_way == n_way
    assert episode.n_shot == n_shot
    assert episode.n_query == n_query


def test_episode_extraction_allows_frequency_weighted_query_counts():
    n_way, n_shot, n_query = 2, 10, 2
    data_list, labels, query = [], [], []
    for label, label_queries in enumerate((3, 1)):
        for member in range(n_shot + label_queries):
            data_list.append(
                Data(
                    x=torch.randn(4, 768),
                    edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
                    global_node_ids=torch.tensor([3000 + len(data_list), 2, 3, -1]),
                    center_node_idx=torch.tensor(3000 + len(data_list)),
                    num_nodes=4,
                )
            )
            labels.append(label)
            query.append(member >= n_shot)
    graphs = Batch.from_data_list(data_list)
    graphs.task_id_per_sample = torch.zeros(len(data_list), dtype=torch.long)
    label_tensor = torch.nn.functional.one_hot(torch.tensor(labels), n_way).float()
    edge_mask = torch.tensor(query).repeat_interleave(n_way)
    filler = torch.empty(0)
    batch = (graphs, filler, label_tensor, filler, filler, edge_mask, filler, filler, filler)
    episode, = iter_episodes(
        batch,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        equal_query_counts=False,
    )
    assert int(episode.query_mask.sum()) == n_way * n_query
