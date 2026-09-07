"""Frozen, episode-paired decisions for the one public PRODIGY KG test.

Inputs contain one accuracy fraction per cached episode, not query-level
correctness rows. The caller must verify episode identities and model/input
replay before using these statistical decisions. This module neither runs
models nor authorizes execution of role interventions after a failed quality
gate; the caller must enforce that ordered execution contract.
"""

from __future__ import annotations

import numpy as np


EXPECTED_EPISODES = 500
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260907
CONFIDENCE_LEVEL = 0.95
_CHUNK_SIZE = 1_000

INTERPRETATION = (
    "Paired episode-bootstrap intervals concern the sampled task distribution "
    "conditional on this fixed graph and checkpoint. Shared entities do not "
    "supply independent graphs or training seeds. Intervals including zero "
    "are inconclusive, not positive evidence. Input order must identify the "
    "same cached episodes in every arm."
)


def _accuracy_vector(values, name: str) -> np.ndarray:
    """Require a nonempty, one-dimensional vector of real accuracy fractions."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric episode-accuracy vector") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"{name} must be a nonempty one-dimensional episode-accuracy vector; "
            "do not pass or flatten query-level correctness rows"
        )
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric accuracy fractions")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite accuracy fractions")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} accuracies must lie in [0, 1]")
    return array


def paired_difference(left_acc, right_acc) -> dict:
    """Bootstrap the mean of paired episode differences, accepting any length.

    This lower-level helper allows small deterministic fixtures. Production
    callers use ``decide_quality`` or ``decide_roles`` to enforce 500 episodes.
    The same resampled episode index selects both arms. Resetting this local
    generator also shares resampling indices across same-length comparisons;
    NumPy's process-wide random state is neither read nor modified.
    """
    left = _accuracy_vector(left_acc, "left_acc")
    right = _accuracy_vector(right_acc, "right_acc")
    if left.shape != right.shape:
        raise ValueError("paired arms must have identical episode-vector shapes")
    differences = left - right
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_DRAWS, _CHUNK_SIZE):
        stop = min(start + _CHUNK_SIZE, BOOTSTRAP_DRAWS)
        indices = rng.integers(0, differences.size, size=(stop - start, differences.size))
        means[start:stop] = differences[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.percentile(means, [2.5, 97.5], method="linear")
    return {
        "episodes": int(differences.size),
        "mean_difference": float(differences.mean(dtype=np.float64)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval_method": "two-sided percentile, linear quantiles",
        "resampling_unit": "paired episode accuracy difference",
    }


def _validated_arms(arms: dict, expected_episodes: int) -> dict:
    if (
        isinstance(expected_episodes, (bool, np.bool_))
        or not isinstance(expected_episodes, (int, np.integer))
        or expected_episodes < 1
    ):
        raise ValueError("expected_episodes must be a positive integer")
    vectors = {name: _accuracy_vector(values, name) for name, values in arms.items()}
    for name, vector in vectors.items():
        if vector.shape != (expected_episodes,):
            raise ValueError(
                f"{name} must have shape ({expected_episodes},), one accuracy per "
                f"cached episode; found {vector.shape}"
            )
    return vectors


def _comparison(left: np.ndarray, right: np.ndarray, direction: str) -> dict:
    result = paired_difference(left, right)
    if direction == "positive":
        result["criterion"] = "ci_lower > 0"
        result["strict_passed"] = result["ci_lower"] > 0.0
    elif direction == "negative":
        result["criterion"] = "ci_upper < 0"
        result["strict_passed"] = result["ci_upper"] < 0.0
    else:
        raise ValueError(f"unsupported comparison direction: {direction}")
    return result


def _decision(gate: str, arms: dict, comparisons: dict) -> dict:
    return {
        "gate": gate,
        "gate_passed": all(item["strict_passed"] for item in comparisons.values()),
        "episodes": int(next(iter(arms.values())).size),
        "means": {name: float(values.mean(dtype=np.float64)) for name, values in arms.items()},
        "comparisons": comparisons,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
            "interval_method": "two-sided percentile, linear quantiles",
            "resampling_unit": "paired episode accuracy difference",
        },
        "interpretation": INTERPRETATION,
        "requires_verified_episode_pairing_and_replay": True,
    }


def decide_quality(
    native_acc, raw_acc, untrained_acc, *, expected_episodes: int = EXPECTED_EPISODES
) -> dict:
    """Pass only when native-minus-each-prototype lower bounds exceed zero.

    ``expected_episodes`` is overrideable for unit fixtures only; the declared
    scientific protocol requires its default of 500. No rounding or tolerance
    is applied to the strict decision inequalities.
    """
    arms = _validated_arms(
        {"native": native_acc, "raw": raw_acc, "untrained": untrained_acc},
        expected_episodes,
    )
    comparisons = {
        "native_minus_raw": _comparison(arms["native"], arms["raw"], "positive"),
        "native_minus_untrained": _comparison(
            arms["native"], arms["untrained"], "positive"
        ),
    }
    return _decision("quality", arms, comparisons)


def decide_roles(
    native_acc, support_acc, query_acc, *, expected_episodes: int = EXPECTED_EPISODES
) -> dict:
    """Pass only when support helps and query suppression hurts by their CIs.

    This pure metric helper does not execute interventions. The runtime must
    refuse role evaluation unless its prior verified quality gate has passed.
    ``expected_episodes`` is overrideable for unit fixtures only.
    """
    arms = _validated_arms(
        {"native": native_acc, "support": support_acc, "query": query_acc},
        expected_episodes,
    )
    comparisons = {
        "support_minus_native": _comparison(arms["support"], arms["native"], "positive"),
        "query_minus_native": _comparison(arms["query"], arms["native"], "negative"),
    }
    return _decision("roles", arms, comparisons)
