"""Descriptive, overlap-standardized ranking-loss risk; no statistical inference."""
import numpy as np


def _quantile(x, w, q):
    """Weighted inverse empirical CDF; exact threshold ties belong to low."""
    order = np.argsort(x, kind="stable")
    cumulative = np.cumsum(w[order])
    index = np.searchsorted(cumulative, np.asarray(q) * cumulative[-1], side="left")
    return x[order[np.minimum(index, len(x) - 1)]]


def risk_summary(margin, interaction, loss, degree_group, weights):
    """Caller supplies only initially native-correct pairs and original pair weights.

    Quantiles use predictors/weights only. Decile-edge ties enter the lower bin;
    interaction-median ties enter low. Risks are standardized to identical stratum
    masses over overlap strata, not separately pooled high/low group masses.
    Coverage is conditional on supplied eligible pairs; input_mass preserves their
    original unconditional mass. Empty inputs return zero coverage and null risks.
    """
    margin, interaction, loss, weights = [np.asarray(x, dtype=float) for x in
                                         (margin, interaction, loss, weights)]
    degree_group = np.asarray(degree_group)
    arrays = (margin, interaction, loss, weights, degree_group)
    if any(x.ndim != 1 or x.shape != margin.shape for x in arrays):
        raise ValueError("Inputs must be aligned one-dimensional arrays")
    if any(not np.isfinite(x).all() for x in arrays[:4]):
        raise ValueError("Numeric inputs must be finite")
    if np.any((loss < 0) | (loss > 1)) or np.any(weights <= 0):
        raise ValueError("Loss must be in [0, 1], weights strictly positive")
    if not np.isin(degree_group, ["equal", "cue_correct", "cue_wrong", "cue_tied"]).all():
        raise ValueError("Unknown degree group")
    total = float(weights.sum())
    if not np.isfinite(total):
        raise ValueError("Total weight must be finite")
    cuts = _quantile(margin, weights, np.arange(1, 10) / 10) if len(margin) else []
    bins = np.searchsorted(cuts, margin, side="left")
    assignments = {"crude": ["all"] * len(margin),
                   "margin": [str(b) for b in bins],
                   "margin_degree": [f"{b}:{g}" for b, g in zip(bins, degree_group)]}
    result = {}
    for name, labels in assignments.items():
        labels = np.asarray(labels)
        strata, covered, high_sum, low_sum = [], 0., 0., 0.
        for label in sorted(set(labels)):
            selected = labels == label
            x, y, w = interaction[selected], loss[selected], weights[selected]
            median = float(_quantile(x, w, .5))
            high = x > median
            row = {"stratum": str(label), "n": int(selected.sum()),
                   "mass": float(w.sum()), "median": median}
            for arm, mask in (("high", high), ("low", ~high)):
                mass = float(w[mask].sum())
                row.update({f"{arm}_n": int(mask.sum()), f"{arm}_mass": mass,
                            f"{arm}_risk": float(w[mask] @ y[mask] / mass) if mass else None})
            valid = row["high_mass"] > 0 and row["low_mass"] > 0
            row["risk_difference"] = row["high_risk"] - row["low_risk"] if valid else None
            if valid:
                covered += row["mass"]
                high_sum += row["mass"] * row["high_risk"]
                low_sum += row["mass"] * row["low_risk"]
            strata.append(row)
        result[name] = {"risk_difference": (high_sum - low_sum) / covered if covered else None,
                        "high_risk": high_sum / covered if covered else None,
                        "low_risk": low_sum / covered if covered else None,
                        "coverage": covered / total if total else 0.,
                        "input_mass": total, "covered_mass": covered, "strata": strata}
    result["margin"]["quantile_edges"] = np.asarray(cuts).tolist()
    result["margin_degree"]["quantile_edges"] = np.asarray(cuts).tolist()
    return result
