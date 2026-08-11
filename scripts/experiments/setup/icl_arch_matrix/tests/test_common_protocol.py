from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from scripts.experiments.setup.icl_arch_matrix.common_protocol import (
    N_QUERY,
    N_SHOT,
    N_WAY,
    iter_episodes,
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
    for episode in episodes:
        assert episode.x.shape == (N_WAY * (N_SHOT + N_QUERY) * 3, 768)
        assert episode.centers.numel() == N_WAY * (N_SHOT + N_QUERY)
        assert int(episode.support_mask.sum()) == N_WAY * N_SHOT
        assert int(episode.query_mask.sum()) == N_WAY * N_QUERY
        assert (episode.centers >= 0).all()


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
