# Extreme-tail objective, validation-only frozen experiment

Run `run.py --dry-run` for contract. Six cells: BCE/tail × seeds0/1/2, each2000
updates, evaluation every50, earliest maximum standalone2018 Hits@50. Input graph
and2017 fresh-negative panel match candidate_fresh_v1. Objective alone changes:
BCE plus coefficient1 mean softplus(1+n-p), all2048 sampled positives against50
highest-scoring training negatives, refreshed every10. No boundary-positive
oversampling: that would confound this objective comparison. Top2048 mining uses
identical random slot streams; model-dependent negative identities can differ.
Same-year probe contains training positives; it is diagnostic only and has been
explored historically. No test input, test scoring, final refit, architecture or
hyperparameter search. Previous2019 test exploration remains disclosed.
Advance only if all3 selected paired gains are positive and their mean>=0.005;
otherwise stop. Success permits proposed earlier-year replication, not test access.
Runtime h2_tail_v1, dedicated worktree prodigy-collab-h2-tail, GPU1 only. Offline
W&B plus full curves, earliest selected checkpoint, final checkpoint, hashes,
matched sample-stream fingerprints and exact selected replay are required.
