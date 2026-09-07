import unittest

import numpy as np

from scripts.experiments.setup.target_performance_mechanisms.support_boundary_calibration import (
    apply_calibrator, fit_calibrator,
)


class SupportBoundaryCalibrationTests(unittest.TestCase):
    def test_offset_discrimination(self):
        margins = np.array([10., 11., 12., 14., 15., 16.])
        labels = np.array([0, 0, 0, 1, 1, 1])
        fit = fit_calibrator(margins, labels)
        np.testing.assert_array_equal(apply_calibrator(margins, fit) >= 0, labels)
        self.assertLess(fit["gradient_max_abs"], 1e-10)
        self.assertEqual(fit["regularization"], 1.0)
        self.assertEqual(fit["mean"], margins.mean())
        self.assertEqual(fit["scale"], margins.std())

    def test_constant_margins_finite(self):
        fit = fit_calibrator([5.] * 4, [0, 1, 1, 1])
        p = apply_calibrator([-100., 5., 100.], fit)
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(fit["scale"], 1.0)
        self.assertEqual(fit["slope"], 0.0)
        np.testing.assert_array_equal(p, np.repeat(p[0], 3))
        self.assertGreater(p[0], 0)
        self.assertLess(p[0], np.log(3))  # Regularized intercept, not empirical prevalence.

    def test_invalid_labels(self):
        for labels in ([0, 0], [1, 1], [-1, 1], [0, 2], [0, .5], [0, np.nan]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                fit_calibrator([1, 2], labels)
        for margins, labels in (([], []), ([1, 2], [0, 1, 0]),
                                ([[1, 2]], [0, 1]), ([1, np.inf], [0, 1])):
            with self.assertRaises(ValueError):
                fit_calibrator(margins, labels)

    def test_permutation_invariance(self):
        rng = np.random.default_rng(123)
        margins = rng.normal(size=40)
        labels = rng.integers(0, 2, size=40)
        permutation = rng.permutation(40)
        self.assertEqual(fit_calibrator(margins, labels),
                         fit_calibrator(margins[permutation], labels[permutation]))

    def test_objective_and_stationarity(self):
        margins, labels = np.array([1., 2., 8., 9., 12.]), np.array([0, 1, 0, 1, 1])
        fit = fit_calibrator(margins, labels)
        z = (margins - fit["mean"]) / fit["scale"]
        score = z * fit["slope"] + fit["intercept"]
        objective = np.mean(np.logaddexp(0., score) - labels * score)
        objective += .5 * (fit["slope"] ** 2 + fit["intercept"] ** 2)
        self.assertAlmostEqual(fit["objective"], objective)
        residual = np.exp(-np.logaddexp(0., -apply_calibrator(margins, fit))) - labels
        np.testing.assert_allclose([np.mean(residual * z) + fit["slope"],
                                    np.mean(residual) + fit["intercept"]], 0, atol=1e-10)

    def test_numerical_range_and_small_supports(self):
        rng = np.random.default_rng(19)
        for n in (2, 3, 6, 20, 100):
            for _ in range(100):
                margins = rng.normal(size=n) * 10 ** rng.uniform(-8, 8) + rng.normal() * 100
                labels = rng.integers(0, 2, n)
                labels[:2] = [0, 1]
                fit = fit_calibrator(margins, labels)
                self.assertLessEqual(fit["gradient_max_abs"], 1e-10)


if __name__ == "__main__":
    unittest.main()
