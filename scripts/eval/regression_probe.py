"""Few-shot regression probe — the protocol, checkpoint-free so it can self-test.

Why this exists
---------------
The episodic ``task_name=regression`` eval cannot measure regression. Three facts,
each verifiable in the tree:

1. ``models/general_gnn.py:30`` builds ``regression_head`` (Linear->ReLU->Linear)
   whenever ``task_name == "regression"``, and ``:158`` predicts with it, bypassing
   ``decode()`` entirely -- so the support set's label prototypes never enter.
2. The head is absent from every NM/CL/FP checkpoint (38 keys, none of them
   ``regression_head``), and the load is ``strict=False``, so it stays at its random
   initialisation.
3. ``--eval_only True`` makes ``trainer.py:1486`` run one ``do_eval`` under
   ``no_grad`` and return at ``:1503``. No optimizer step exists on that path.

So the reported number is a fixed random projection of the frozen embedding. Because
``run_single_experiment.py:34`` seeds before the model is built and every job passed
``--seed 0``, that projection is at least identical across arms -- comparisons are
controlled, but the metric has almost no power. It sits at or below the raw-feature
floor on exactly the targets features predict best.

What this replaces it with
--------------------------
The protocol already used for ``features_only_floor.csv``, lifted from
``setup/topology_feature_ssl/leakage_baseline.py:74`` rather than reinvented, so a
probe on frozen embeddings is directly comparable to that published floor: per
episode fit ``StandardScaler`` + ``Ridge`` on the support rows, predict the query
rows, accumulate predictions across episodes, and score Spearman ONCE over the pool.

The episode set is built once per (dataset, target) and SHARED by every arm, so all
arms see identical support and query nodes -- the property the old eval lacked.

Self-test (no cluster, no GPU, no checkpoint)::

    python scripts/eval/regression_probe.py --self-test
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

import numpy as np


def apply_transform(y: np.ndarray, transform: str) -> np.ndarray:
    """Mirror ``data/midterm.py:_apply_target_transform`` exactly.

    The loader applies log1p to the RAW tensor and only then counts finite entries,
    so a negative value becomes NaN and drops out. Clipping first (or masking first)
    would silently keep rows the benchmark discards, and the two numbers would stop
    being comparable.
    """
    if transform == "none":
        return y
    if transform == "log1p":
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log1p(y)
    raise ValueError(f"Unknown transform={transform!r}. Use 'none' or 'log1p'.")


@dataclass
class EpisodeSet:
    """Support/query node indices for every episode, shared across arms.

    ``support[i]`` and ``query[i]`` index into the *labeled-node* array, not into
    the graph's node ids; :func:`build_episodes` returns the node ids separately so
    a caller embeds each node exactly once.
    """
    support: np.ndarray          # (episodes, shots) int
    query: np.ndarray            # (episodes, n_query) int
    nodes: np.ndarray            # (n_used,) graph node ids referenced by the above
    target: np.ndarray           # (n_used,) target value aligned to `nodes`


def build_episodes(node_ids: np.ndarray, target: np.ndarray, shots: int,
                   n_query: int, episodes: int, seed: int = 0) -> Optional[EpisodeSet]:
    """Sample ``episodes`` disjoint support/query splits from the labeled nodes.

    Returns None when there are not enough labeled nodes for a single episode.
    Indices are remapped onto the compacted ``nodes`` array so the caller embeds
    only the nodes actually used (~episodes x (shots + n_query)), not the whole
    graph -- on covid19 that is ~11k nodes instead of 23M.
    """
    n = len(node_ids)
    if n < shots + n_query or episodes < 1:
        return None
    rng = np.random.default_rng(seed)
    sup_raw, qry_raw = [], []
    for _ in range(episodes):
        pick = rng.choice(n, size=shots + n_query, replace=False)
        sup_raw.append(pick[:shots])
        qry_raw.append(pick[shots:])
    sup_raw = np.asarray(sup_raw)
    qry_raw = np.asarray(qry_raw)

    used = np.unique(np.concatenate([sup_raw.ravel(), qry_raw.ravel()]))
    remap = np.full(n, -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return EpisodeSet(support=remap[sup_raw], query=remap[qry_raw],
                      nodes=np.asarray(node_ids)[used],
                      target=np.asarray(target)[used])


def probe_spearman(features: np.ndarray, episodes: EpisodeSet, alpha: float = 1.0,
                   standardize: bool = True) -> dict:
    """Fit a ridge probe per episode; score the accumulated query predictions once.

    ``features`` is (n_used, d), row-aligned to ``episodes.nodes``. Accumulating
    across episodes before scoring mirrors the floor's own accumulation -- scoring
    each episode separately and averaging is a different statistic and would not be
    comparable to the published numbers.
    """
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    y = episodes.target
    preds: list[float] = []
    trues: list[float] = []
    for sup, qry in zip(episodes.support, episodes.query):
        xs, ys = features[sup], y[sup]
        if standardize:
            scaler = StandardScaler().fit(xs)
            xs_t, xq_t = scaler.transform(xs), scaler.transform(features[qry])
        else:
            xs_t, xq_t = xs, features[qry]
        model = Ridge(alpha=alpha).fit(xs_t, ys)
        preds.extend(model.predict(xq_t).tolist())
        trues.extend(y[qry].tolist())

    preds_a, trues_a = np.asarray(preds), np.asarray(trues)
    rho = spearmanr(preds_a, trues_a).statistic
    ss_res = float(np.sum((trues_a - preds_a) ** 2))
    ss_tot = float(np.sum((trues_a - trues_a.mean()) ** 2))
    return {
        "spearman": float(rho) if np.isfinite(rho) else float("nan"),
        "rmse": float(np.sqrt(ss_res / len(trues_a))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_pred": int(len(trues_a)),
        "alpha": float(alpha),
    }


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #

def _self_test() -> int:
    checks, failures = 0, []

    def check(cond, label):
        nonlocal checks
        checks += 1
        print(f"  {'OK  ' if cond else 'FAIL'} {label}")
        if not cond:
            failures.append(label)

    rng = np.random.default_rng(0)
    n, d = 4000, 32
    node_ids = np.arange(n) * 3            # non-contiguous ids, as in a real graph
    x = rng.normal(size=(n, d))
    w = rng.normal(size=d)
    y_signal = x @ w + 0.1 * rng.normal(size=n)

    ep = build_episodes(node_ids, y_signal, shots=10, n_query=12, episodes=200, seed=0)
    check(ep is not None, "episodes build")
    check(ep.support.shape == (200, 10) and ep.query.shape == (200, 12),
          "episode shapes")
    check(len(ep.nodes) == len(ep.target), "nodes and target aligned")
    check(set(np.asarray(ep.nodes).tolist()).issubset(set(node_ids.tolist())),
          "episode nodes are real node ids")

    # support and query must never overlap within an episode -- that would leak.
    overlap = any(set(s.tolist()) & set(q.tolist())
                  for s, q in zip(ep.support, ep.query))
    check(not overlap, "no support/query overlap within an episode")

    feats = x[np.searchsorted(node_ids, ep.nodes)]
    res = probe_spearman(feats, ep, alpha=1.0)
    check(res["n_pred"] == 200 * 12, "prediction count")

    # Estimator correctness is asserted where the problem is well-posed: with 100
    # supports for 32 dims a planted LINEAR signal must come back almost exactly.
    ep_big = build_episodes(node_ids, y_signal, shots=100, n_query=12, episodes=200, seed=0)
    feats_big = x[np.searchsorted(node_ids, ep_big.nodes)]
    res_big = probe_spearman(feats_big, ep_big, alpha=1.0)
    check(res_big["spearman"] > 0.9,
          f"recovers a planted linear signal at 100 shots (rho={res_big['spearman']:.3f})")

    # ...and the 10-shot number is pinned as WEAK on that same perfect signal. This
    # is a property of the protocol, not a defect: 10 supports cannot determine 32
    # coefficients, so Ridge shrinkage does most of the work. The benchmark's
    # embeddings are 256-d and its floor is 768-d, both far worse conditioned than
    # this, which bounds how much any 10-shot regression number can ever say.
    check(0.2 < res["spearman"] < 0.8,
          f"10 shots on the SAME perfect signal is much weaker "
          f"(rho={res['spearman']:.3f} vs {res_big['spearman']:.3f} at 100)")

    # permutation control: shuffling the target must destroy the correlation. This
    # is the check that catches a feature/target row misalignment, which would
    # otherwise manufacture signal from nothing.
    ep_perm = EpisodeSet(ep.support, ep.query, ep.nodes,
                         rng.permutation(ep.target))
    res_perm = probe_spearman(feats, ep_perm, alpha=1.0)
    check(abs(res_perm["spearman"]) < 0.1,
          f"permuted target gives ~0 (rho={res_perm['spearman']:+.3f})")

    # pure noise features must also give ~0
    res_noise = probe_spearman(rng.normal(size=feats.shape), ep, alpha=1.0)
    check(abs(res_noise["spearman"]) < 0.1,
          f"noise features give ~0 (rho={res_noise['spearman']:+.3f})")

    # a shared episode set must be deterministic for a given seed
    ep2 = build_episodes(node_ids, y_signal, shots=10, n_query=12, episodes=200, seed=0)
    check(np.array_equal(ep.support, ep2.support) and np.array_equal(ep.nodes, ep2.nodes),
          "episode set is deterministic for a fixed seed")

    # too few labeled nodes -> None, not a crash
    check(build_episodes(node_ids[:5], y_signal[:5], 10, 12, 10) is None,
          "insufficient labeled nodes returns None")

    print(f"\n{checks - len(failures)}/{checks} checks passed")
    return 0 if not failures else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return _self_test()
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
