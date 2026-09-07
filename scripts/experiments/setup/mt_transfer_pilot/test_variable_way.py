#!/usr/bin/env python3
from collections import Counter

from variable_way import VARIABLE_NM_WAYS, episode_n_way


def test_variable_way_distribution():
    counts = Counter(VARIABLE_NM_WAYS)
    assert len(VARIABLE_NM_WAYS) == 20
    assert counts == {2: 6, 3: 5, 5: 4, 10: 3, 30: 2}


def test_only_nm_varies():
    assert episode_n_way("neighbor_matching", True) == VARIABLE_NM_WAYS
    assert episode_n_way("neighbor_matching", False) == 30
    assert episode_n_way("classification", True) == 2
