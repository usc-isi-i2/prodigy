from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).with_name("compute_graph_divergence.py")
SPEC = importlib.util.spec_from_file_location("compute_graph_divergence", MODULE_PATH)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


def test_directed_newman_corrects_for_imbalanced_endpoint_marginals() -> None:
    # Directed mixing counts: [[70, 10], [10, 10]].  Raw homophily is .80,
    # while source and destination endpoint marginals imply .68 by chance.
    yu = np.asarray([0] * 80 + [1] * 20)
    yv = np.asarray([0] * 70 + [1] * 10 + [0] * 10 + [1] * 10)

    result = MOD.directed_label_mixing(yu, yv)

    assert result["label_homophily"] == pytest.approx(0.8)
    assert result["label_homophily_expected"] == pytest.approx(0.68)
    assert result["label_assortativity_newman"] == pytest.approx(0.375)
    assert result["labeled_edge_count"] == 100
    assert result["label_mixing"]["counts"] == [[70, 10], [10, 10]]
    assert result["label_mixing"]["same_label_rate_by_source_class"] == {
        "0": pytest.approx(0.875),
        "1": pytest.approx(0.5),
    }


def test_directed_newman_uses_distinct_source_and_destination_marginals() -> None:
    # Counts [[4, 4], [1, 1]] have identical rows, hence no association even
    # though 50% of edges are same-label and endpoint marginals differ.
    yu = np.asarray([0] * 8 + [1] * 2)
    yv = np.asarray([0] * 4 + [1] * 4 + [0] + [1])

    result = MOD.directed_label_mixing(yu, yv)

    assert result["label_homophily"] == pytest.approx(0.5)
    assert result["label_homophily_expected"] == pytest.approx(0.5)
    assert result["label_assortativity_newman"] == pytest.approx(0.0)


def test_directed_newman_handles_single_class_without_dividing_by_zero() -> None:
    result = MOD.directed_label_mixing(np.zeros(12), np.zeros(12))

    assert result["label_homophily"] == 1.0
    assert result["label_homophily_expected"] == 1.0
    assert result["label_assortativity_newman"] is None
