# Raw multitask-SSL snapshots

The classification and regression exports are kept separately by experiment:

- `pairs_node_classification.csv` and `pairs_node_regression.csv`
- `rotation_node_classification.csv` and `rotation_node_regression.csv`

The two `*_static_link_prediction_void_pre_20260723.csv` files preserve the
corresponding historical static-link-prediction outputs without mixing or
overwriting them.  They were produced by the invalid episodic static-LP
evaluator before the 2026-07-23 repair and must not be cited as benchmark
results.  Use the repaired pair-link outputs under `../pair_lp/` instead.
