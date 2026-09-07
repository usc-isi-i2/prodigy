"""Fixed, support-only scalar logistic calibration (no query-label access).

For support-standardized margin z and theta=(slope, intercept), minimize
mean(logaddexp(0, slope*z + intercept) - y*(slope*z + intercept))
+ 0.5 * (slope**2 + intercept**2). Both parameters are regularized with
lambda=1, including the intercept. Standardization uses the support population
standard deviation; constant support margins use scale=1. No hyperparameters
are selected using query outcomes. Returned log-odds concern label 1 versus 0.
"""

import numpy as np


def _vector(values, name):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite one-dimensional array")
    return result


def _sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -values))


def fit_calibrator(margins, labels):
    """Fit only on held-out support predictions and their binary labels.

    Returns a JSON-serializable dictionary containing support mean/scale and
    the unique regularized logistic optimum. The slope is not constrained to
    be positive. Callers must ensure calibration examples were genuinely held
    out when their margins were produced; this function cannot audit that split.
    """
    margins = _vector(margins, "margins")
    labels = _vector(labels, "labels")
    if margins.shape != labels.shape:
        raise ValueError("margins and labels must have the same shape")
    if not np.array_equal(np.unique(labels), [0.0, 1.0]):
        raise ValueError("labels must contain both classes, encoded exactly 0 and 1")
    # Sorting makes reductions independent of support-example input ordering.
    order = np.lexsort((labels, margins))
    margins, labels = margins[order], labels[order]
    mean = float(np.mean(margins))
    scale = float(np.std(margins, ddof=0))
    if not np.isfinite(mean) or not np.isfinite(scale):
        raise ValueError("support standardization overflowed")
    if scale == 0.0:
        scale = 1.0
    z = (margins - mean) / scale
    design = np.column_stack((z, np.ones_like(z)))
    theta = np.zeros(2, dtype=np.float64)

    def objective(candidate):
        score = design @ candidate
        # Signed form avoids cancellation for confidently correct examples.
        return float(np.mean(np.logaddexp(0.0, (1 - 2 * labels) * score))
                     + 0.5 * np.dot(candidate, candidate))

    for iteration in range(100):
        probability = _sigmoid(design @ theta)
        gradient = design.T @ (probability - labels) / len(labels) + theta
        if np.max(np.abs(gradient)) <= 1e-10:
            break
        weights = probability * (1 - probability)
        hessian = (design.T * weights) @ design / len(labels) + np.eye(2)
        direction = np.linalg.solve(hessian, gradient)
        previous = objective(theta)
        step = 1.0
        for _ in range(60):
            candidate = theta - step * direction
            roundoff = 8 * np.finfo(np.float64).eps * max(1.0, abs(previous))
            if objective(candidate) <= previous - 1e-4 * step * np.dot(gradient, direction) + roundoff:
                theta = candidate
                break
            step *= 0.5
        else:
            raise RuntimeError("calibrator line search failed")
    else:
        raise RuntimeError("calibrator did not converge")
    return {"mean": mean, "scale": scale, "slope": float(theta[0]),
            "intercept": float(theta[1]), "regularization": 1.0,
            "objective": objective(theta), "iterations": iteration,
            "gradient_max_abs": float(np.max(np.abs(gradient)))}


def apply_calibrator(margins, fit):
    """Apply the frozen transform; return calibrated label-1-versus-0 log-odds."""
    margins = _vector(margins, "margins")
    parameters = np.asarray([fit[key] for key in ("mean", "scale", "slope", "intercept")],
                            dtype=np.float64)
    if not np.isfinite(parameters).all() or parameters[1] <= 0:
        raise ValueError("fit must contain finite parameters and a positive scale")
    mean, scale, slope, intercept = parameters
    return slope * ((margins - mean) / scale) + intercept
