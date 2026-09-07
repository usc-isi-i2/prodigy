"""Symmetrize observed query contexts; never treat this as a realized-input bound."""
from collections import defaultdict
import math

import numpy as np


def identity_groups(ids, tasks):
    ids, tasks = np.asarray(ids), np.asarray(tasks)
    if ids.ndim != 1 or ids.shape != tasks.shape or (ids < 0).any():
        raise ValueError("invalid query identities/episodes")
    grouped = defaultdict(list)
    for i, key in enumerate(zip(tasks.tolist(), ids.tolist())):
        grouped[key].append(i)
    return [np.asarray(v, dtype=np.int64) for v in grouped.values()]


def cyclic_permutation(ids, tasks, shift=1):
    """Destination row receives source row p[i]; plan does not read labels."""
    p = np.arange(len(ids))
    for g in identity_groups(ids, tasks):
        p[g] = np.roll(g, shift)
    return p


def symmetrized_groups(logits, truth, ids, tasks):
    logits = np.asarray(logits, dtype=np.float64)
    truth = np.asarray(truth)
    if logits.ndim != 2 or truth.shape != (len(logits),) or not np.isfinite(logits).all():
        raise ValueError("invalid predictions/truth")
    if not np.issubdtype(truth.dtype, np.integer) or (truth < 0).any() or (truth >= logits.shape[1]).any():
        raise ValueError("invalid class indices")
    if len(ids) != len(logits) or len(tasks) != len(logits):
        raise ValueError("query metadata length mismatch")
    z = logits - logits.max(axis=1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    pred = logits.argmax(axis=1)
    rows = []
    for g in identity_groups(ids, tasks):
        k, labels = len(g), truth[g]
        if len(np.unique(labels)) != k:
            raise ValueError("requires distinct pseudo-classes for repeated query identities")
        # Every context's score vector is evaluated against every group's truth.
        cross = lp[g][:, labels]
        m = cross.max(axis=1)
        logmass = m + np.log(np.exp(cross-m[:, None]).sum(axis=1))
        original = float(-lp[g, labels].sum())
        sym = float(-cross.sum()/k)
        floor = k*math.log(k)
        mass_deficit = float(-logmass.sum())
        imbalance = float((-cross.mean(axis=1)+logmass-math.log(k)).sum())
        if not np.isclose(sym, floor+mass_deficit+imbalance, atol=1e-10, rtol=1e-12):
            raise ValueError("loss decomposition failed")
        if min(mass_deficit, imbalance) < -1e-10:
            raise ValueError("negative loss decomposition component")
        correct = float((pred[g, None] == labels[None, :]).sum()/k)
        if correct > 1+1e-12 or sym < floor-1e-10:
            raise ValueError("symmetrized bound failed")
        rows.append({"episode": int(tasks[g[0]]), "center_id": int(ids[g[0]]),
            "size": k, "cohort": "repeated" if k > 1 else "unique",
            "original_correct": int((pred[g] == labels).sum()),
            "symmetrized_correct": correct, "correct_ceiling": 1,
            "original_nll_sum": original, "symmetrized_nll_sum": sym,
            "nll_floor_sum": floor, "mass_deficit_sum": mass_deficit,
            "within_group_imbalance_sum": imbalance})
    return rows


def group_totals(rows):
    n = sum(r["size"] for r in rows)
    if not n:
        return {"queries": 0, "identity_groups": 0}
    mapping = {"original_accuracy": "original_correct", "symmetrized_accuracy": "symmetrized_correct",
        "accuracy_ceiling": "correct_ceiling", "original_nll": "original_nll_sum",
        "symmetrized_nll": "symmetrized_nll_sum", "nll_floor": "nll_floor_sum",
        "mass_deficit": "mass_deficit_sum", "within_group_imbalance": "within_group_imbalance_sum"}
    return {"queries": n, "identity_groups": len(rows),
        **{k: sum(r[v] for r in rows)/n for k, v in mapping.items()}}
