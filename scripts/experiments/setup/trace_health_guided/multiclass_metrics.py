"""Score paired episodes without pooling unrelated local class columns."""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def episode_metrics(labels, logits):
    labels = np.asarray(labels)
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Expected query-by-class logits")
    classes = np.arange(logits.shape[1])
    if labels.shape != (len(logits),) or not np.isin(labels, classes).all():
        raise ValueError("Labels must index the episode's local class columns")
    if not np.isfinite(logits).all() or len(np.unique(labels)) != len(classes):
        raise ValueError("Each class needs queries and all logits must be finite")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    prediction = logits.argmax(axis=1)
    # Explicit one-vs-rest avoids sklearn's special binary dispatch and permits
    # the same definition for two-way and twenty-way episodes.
    auc = np.mean([
        roc_auc_score(labels == c, probabilities[:, c]) for c in classes
    ])
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, labels=classes,
                                   average="macro", zero_division=0)),
        "ovr_auc": float(auc),
        "nll": float(log_loss(labels, probabilities, labels=classes)),
    }


def paired_summary(native, changed, *, seed=20260907, draws=10000):
    """Episode bootstrap, conditional on the evaluated task and checkpoint.

    Inputs map immutable episode identities to (local labels, logits). The caller
    must verify equal class-column identities and query ordering from cache
    receipts; matching label vectors alone cannot establish input parity.
    """
    if not native or native.keys() != changed.keys():
        raise ValueError("Expected identical nonempty episode inventories")
    differences = []
    for key in sorted(native):
        labels, baseline = native[key]
        other_labels, altered = changed[key]
        if not np.array_equal(labels, other_labels) or np.shape(baseline) != np.shape(altered):
            raise ValueError("Paired query labels or class dimensions differ")
        a, b = episode_metrics(labels, baseline), episode_metrics(other_labels, altered)
        differences.append({metric: b[metric] - a[metric] for metric in a})
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(draws, len(differences)))
    result = {}
    for metric in differences[0]:
        values = np.array([row[metric] for row in differences])
        interval = np.quantile(values[indices].mean(axis=1), [.025, .975])
        result[metric] = {"mean_delta": float(values.mean()),
                          "interval": interval.tolist(),
                          "positive_episodes": int((values > 0).sum()),
                          "negative_episodes": int((values < 0).sum()),
                          "episodes": len(values)}
    return result
