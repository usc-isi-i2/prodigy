import torch

from scripts.experiments.setup.vision_native_cross_ssl.evaluate import (
    build_fixed_episodes,
    episode_seed,
)


def test_fixed_pseudo_episodes_repeat_and_are_target_specific():
    generator = torch.Generator().manual_seed(4)
    features = torch.randn(400, 16, generator=generator)
    features = torch.nn.functional.normalize(features, dim=1)
    seed = episode_seed("twibot20", 0)
    left, left_hash = build_fixed_episodes(features, episodes=3, seed=seed)
    right, right_hash = build_fixed_episodes(features, episodes=3, seed=seed)
    assert left_hash == right_hash
    assert all(torch.equal(a, b) and torch.equal(c, d) for (a, c), (b, d) in zip(left, right))
    _, other_hash = build_fixed_episodes(
        features, episodes=3, seed=episode_seed("election2020", 0)
    )
    assert other_hash != left_hash
