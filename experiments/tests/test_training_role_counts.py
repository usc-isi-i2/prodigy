from types import SimpleNamespace

import torch

from experiments.training_role_counts import TrainingRoleCounter


def test_counts_anchor_support_and_query_roles(tmp_path):
    graph = SimpleNamespace(
        center_node_idx=torch.tensor([4, 5, 6, 7, 8, 9]),
        task_label_map=torch.tensor([[1, 2]]),
    )
    # Two metagraph edges per sample. The first column is sufficient to recover
    # the sample-level mask, matching TrainerFS's existing decode path.
    query_by_sample = torch.tensor([False, False, True, False, False, True])
    edge_query_mask = query_by_sample.repeat_interleave(2)
    batch = [graph, None, None, None, None, edge_query_mask]

    counter = TrainingRoleCounter(num_nodes=10)
    counter.observe_batch(batch)

    assert counter.steps == 1
    assert counter.anchor.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert counter.support.tolist() == [0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    assert counter.query.tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 1]

    output = tmp_path / "counts.csv"
    counter.write_csv(output, graph_id=torch.tensor([0] * 5 + [1] * 5))
    lines = output.read_text().splitlines()
    assert lines[0] == (
        "node_id,graph_id,anchor_count,support_count,query_count,total_count"
    )
    assert lines[2] == "1,0,1,0,0,1"


def test_role_counts_round_trip_state():
    original = TrainingRoleCounter(num_nodes=3)
    original.anchor[:] = [1, 2, 3]
    original.support[:] = [4, 5, 6]
    original.query[:] = [7, 8, 9]
    original.steps = 11

    restored = TrainingRoleCounter(num_nodes=3)
    restored.load_state_dict(original.state_dict())

    assert restored.steps == 11
    assert restored.anchor.tolist() == [1, 2, 3]
    assert restored.support.tolist() == [4, 5, 6]
    assert restored.query.tolist() == [7, 8, 9]
