from data.social_llm_dataset import _episode_seed


def test_episode_seed_default_preserves_historical_split_seed():
    assert _episode_seed("test") == sum(ord(char) for char in "test")


def test_episode_seed_offset_changes_only_requested_experiment():
    assert _episode_seed("test", 4) == _episode_seed("test") + 4
