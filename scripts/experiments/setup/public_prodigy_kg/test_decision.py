"""Synthetic episode-accuracy decisions; no model/data loading or execution."""

import json
import unittest

import numpy as np

from . import decision


class PairedDifferenceTests(unittest.TestCase):
    def test_constant_difference_has_degenerate_exact_interval(self):
        result = decision.paired_difference([0.75] * 4, [0.5] * 4)
        self.assertEqual(result["mean_difference"], 0.25)
        self.assertEqual(result["ci_lower"], 0.25)
        self.assertEqual(result["ci_upper"], 0.25)
        self.assertEqual(result["episodes"], 4)

    def test_identical_variable_arms_are_paired_not_independently_resampled(self):
        values = [0.0, 1.0, 0.25, 0.75]
        result = decision.paired_difference(values, values)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual(result["ci_lower"], 0.0)
        self.assertEqual(result["ci_upper"], 0.0)

    def test_matches_explicit_episode_index_bootstrap(self):
        left = np.array([0.25, 0.5, 0.75, 0.25])
        right = np.array([0.5, 0.25, 0.25, 0.5])
        rng = np.random.default_rng(20260907)
        indices = rng.integers(0, 4, size=(10_000, 4))
        means = (left[indices] - right[indices]).mean(axis=1)
        expected = np.percentile(means, [2.5, 97.5], method="linear")
        actual = decision.paired_difference(left, right)
        np.testing.assert_array_equal([actual["ci_lower"], actual["ci_upper"]], expected)
        self.assertEqual(actual["mean_difference"], float((left - right).mean()))

    def test_reproducible_without_mutating_global_rng_or_inputs(self):
        left = np.array([0.25, 0.75, 0.5])
        right = np.array([0.5, 0.5, 0.25])
        before_left, before_right = left.copy(), right.copy()
        state = np.random.get_state()
        first = decision.paired_difference(left, right)
        second = decision.paired_difference(left, right)
        after = np.random.get_state()
        self.assertEqual(first, second)
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        np.testing.assert_array_equal(left, before_left)
        np.testing.assert_array_equal(right, before_right)

    def test_invalid_vectors_are_rejected_without_silently_flattening(self):
        invalid = [[], 0.5, [[0.5]], [np.nan], [np.inf], [-0.01], [1.01],
                   [0.5 + 0j], ["0.5"], [{"accuracy": 0.5}]]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                decision.paired_difference(values, [0.5])
        with self.assertRaises(ValueError):
            decision.paired_difference([0.5], [0.5, 0.5])


class QualityDecisionTests(unittest.TestCase):
    def test_full_500_episode_protocol_pass_and_metadata(self):
        result = decision.decide_quality(
            np.full(500, 0.75), np.full(500, 0.5), np.full(500, 0.625)
        )
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["means"], {"native": 0.75, "raw": 0.5, "untrained": 0.625})
        self.assertEqual(result["episodes"], 500)
        self.assertEqual(result["bootstrap"]["seed"], 20260907)
        self.assertEqual(result["bootstrap"]["draws"], 10_000)
        self.assertEqual(result["bootstrap"]["confidence_level"], 0.95)
        self.assertIn("fixed graph and checkpoint", result["interpretation"])
        self.assertIn("independent graphs or training seeds", result["interpretation"])
        self.assertTrue(result["requires_verified_episode_pairing_and_replay"])
        json.dumps(result, allow_nan=False)

    def test_tie_with_either_baseline_fails_strict_conjunction(self):
        for raw, untrained in [([0.75] * 3, [0.5] * 3), ([0.5] * 3, [0.75] * 3)]:
            result = decision.decide_quality([0.75] * 3, raw, untrained, expected_episodes=3)
            self.assertFalse(result["gate_passed"])

    def test_beating_one_readout_does_not_pass_if_other_is_better(self):
        result = decision.decide_quality([0.5] * 3, [0.25] * 3, [0.75] * 3,
                                         expected_episodes=3)
        self.assertTrue(result["comparisons"]["native_minus_raw"]["strict_passed"])
        self.assertFalse(result["comparisons"]["native_minus_untrained"]["strict_passed"])
        self.assertFalse(result["gate_passed"])

    def test_positive_point_estimate_with_interval_crossing_zero_is_inconclusive(self):
        result = decision.decide_quality([0.0, 0.0, 1.0], [0.25] * 3, [0.0] * 3,
                                         expected_episodes=3)
        comparison = result["comparisons"]["native_minus_raw"]
        self.assertGreater(comparison["mean_difference"], 0.0)
        self.assertLess(comparison["ci_lower"], 0.0)
        self.assertFalse(result["gate_passed"])

    def test_no_four_decimal_rounding_in_decision(self):
        tiny = 2.0 ** -40
        result = decision.decide_quality([0.5 + tiny] * 3, [0.5] * 3, [0.5] * 3,
                                         expected_episodes=3)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["comparisons"]["native_minus_raw"]["ci_lower"], tiny)

    def test_public_gate_rejects_query_rows_and_wrong_episode_counts(self):
        for values in [np.ones((500, 80)), np.ones(500 * 80), np.ones(499), np.ones(501)]:
            with self.subTest(shape=values.shape), self.assertRaises(ValueError):
                decision.decide_quality(values, np.ones(500), np.ones(500))
        with self.assertRaises(ValueError):
            decision.decide_quality(np.ones(500), np.ones(499), np.ones(500))

    def test_invalid_test_override_rejected(self):
        for size in [True, False, 0, -1, 3.0, "3"]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                decision.decide_quality([0.5] * 3, [0.25] * 3, [0.0] * 3,
                                         expected_episodes=size)


class RoleDecisionTests(unittest.TestCase):
    def test_support_helps_and_query_hurts_passes(self):
        result = decision.decide_roles([0.5] * 3, [0.75] * 3, [0.25] * 3,
                                       expected_episodes=3)
        self.assertTrue(result["gate_passed"])
        self.assertEqual(result["means"], {"native": 0.5, "support": 0.75, "query": 0.25})
        self.assertEqual(result["comparisons"]["query_minus_native"]["ci_upper"], -0.25)
        json.dumps(result, allow_nan=False)

    def test_less_harmful_support_is_not_a_repair(self):
        result = decision.decide_roles([0.75] * 3, [0.5] * 3, [0.25] * 3,
                                       expected_episodes=3)
        self.assertFalse(result["gate_passed"])
        self.assertFalse(result["comparisons"]["support_minus_native"]["strict_passed"])
        self.assertTrue(result["comparisons"]["query_minus_native"]["strict_passed"])

    def test_query_improvement_or_zero_endpoint_fails(self):
        for support, query in [(0.75, 0.75), (0.75, 0.5), (0.5, 0.25)]:
            with self.subTest(support=support, query=query):
                result = decision.decide_roles([0.5] * 3, [support] * 3, [query] * 3,
                                               expected_episodes=3)
                self.assertFalse(result["gate_passed"])

    def test_negative_query_point_estimate_with_zero_upper_bound_fails(self):
        result = decision.decide_roles([0.5] * 3, [0.75] * 3, [0.5, 0.5, 0.25],
                                       expected_episodes=3)
        comparison = result["comparisons"]["query_minus_native"]
        self.assertLess(comparison["mean_difference"], 0.0)
        self.assertEqual(comparison["ci_upper"], 0.0)
        self.assertFalse(result["gate_passed"])

    def test_roles_reject_missing_episode_and_nonfinite_arm(self):
        with self.assertRaises(ValueError):
            decision.decide_roles([0.5] * 500, [0.75] * 499, [0.25] * 500)
        with self.assertRaises(ValueError):
            decision.decide_roles([0.5] * 500, [0.75] * 500, [np.nan] * 500)


if __name__ == "__main__":
    unittest.main()
