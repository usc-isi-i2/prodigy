import torch

from experiments.nm_overlap import overlap_aware_support_keep_mask


def test_masks_only_training_adjacent_rival_support_messages():
    # Undirected training view: node 4 is adjacent to both class anchors 0 and 1;
    # node 5 is adjacent only to its own anchor 1.
    neighbors = [[4], [4, 5], [], [], [0, 1], [1]]
    col = torch.tensor([value for row in neighbors for value in row])
    rowptr = torch.tensor([0, 1, 3, 3, 3, 5, 6])
    centers = torch.tensor([4, 6, 5, 6])
    task_ids = torch.zeros(4, dtype=torch.long)
    anchors = torch.tensor([[0, 1]])
    source = torch.arange(4).repeat_interleave(2)
    destination = torch.arange(2).repeat(4) + 4
    edges = torch.stack([source, destination])
    query = torch.tensor([False, True, False, True]).repeat_interleave(2)
    labels = torch.tensor([0, 0, 1, 1])
    signs = (torch.nn.functional.one_hot(labels, 2).reshape(-1) * 2 - 1) * (~query)
    attrs = torch.stack([query, signs], dim=1)

    keep = overlap_aware_support_keep_mask(
        rowptr, col, centers, task_ids, anchors, edges, attrs
    )

    assert torch.where(~keep)[0].tolist() == [1]
    assert attrs[~keep].tolist() == [[0, -1]]
    assert keep[query].all()
    assert keep[attrs[:, 1] > 0].all()


def test_rejects_label_layout_that_does_not_match_tasks():
    rowptr = torch.tensor([0, 0, 0])
    col = torch.tensor([], dtype=torch.long)
    edges = torch.tensor([[0], [99]])
    attrs = torch.tensor([[0, -1]])
    try:
        overlap_aware_support_keep_mask(
            rowptr, col, torch.tensor([0]), torch.tensor([0]), torch.tensor([[1]]), edges, attrs
        )
    except ValueError as error:
        assert "label nodes" in str(error)
    else:
        raise AssertionError("invalid metagraph layout was accepted")


def test_handles_independent_label_offsets_for_multiple_tasks():
    # Task 0 support node 4 is also adjacent to its rival anchor 1. Task 1
    # support node 8 is also adjacent to its rival anchor 3.
    neighbors = [[4], [4], [8], [8], [0, 1], [], [], [], [2, 3]]
    col = torch.tensor([value for row in neighbors for value in row])
    rowptr = torch.tensor([0, 1, 2, 3, 4, 6, 6, 6, 6, 8])
    centers = torch.tensor([4, 5, 8, 9])
    task_ids = torch.tensor([0, 0, 1, 1])
    anchors = torch.tensor([[0, 1], [2, 3]])
    source = torch.arange(4).repeat_interleave(2)
    destination = torch.tensor([4, 5, 4, 5, 6, 7, 6, 7])
    edges = torch.stack([source, destination])
    attrs = torch.tensor([
        [0, 1], [0, -1], [1, 0], [1, 0],
        [0, 1], [0, -1], [1, 0], [1, 0],
    ])

    keep = overlap_aware_support_keep_mask(
        rowptr, col, centers, task_ids, anchors, edges, attrs
    )

    assert torch.where(~keep)[0].tolist() == [1, 5]
    assert keep[attrs[:, 0].bool()].all()
    assert keep[attrs[:, 1] > 0].all()
