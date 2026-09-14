from __future__ import annotations

import torch
import torch.nn.functional as F


def random_walk_negative_sampling_loss(
    embeddings: torch.Tensor,
    positive_pairs: torch.Tensor,
    negative_pairs: torch.Tensor,
) -> torch.Tensor:
    """Original-GraphSAGE-style logistic loss over positive and negative pairs."""
    positive_score = (embeddings[positive_pairs[0]] * embeddings[positive_pairs[1]]).sum(-1)
    negative_score = (embeddings[negative_pairs[0]] * embeddings[negative_pairs[1]]).sum(-1)
    return F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()

