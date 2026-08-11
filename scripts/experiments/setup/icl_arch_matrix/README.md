# PRODIGY × VISION × GILT architecture matrix

This is a lightweight, descriptive comparison—not a final paper-number run.

- Seed: `0` only.
- Pretraining: the final-core 31 physical source sets, all using the same source-confined
  30-way/3-shot/4-query neighbor-matching stream.
- Budget: 500 optimizer updates, batch size 4 (2,000 episodes per model), with checkpoints
  at 20, 60, 180, and 500 updates.
- Architectures: repository PRODIGY and thin adapters over pinned official VISION/GILT
  implementations (see `upstream_pins.json`).
- GILT keeps its official model/optimizer defaults but uses one frozen seed-0 orthogonal
  768-to-128 feature alignment. This prevents fitting a separate PCA on each downstream
  target (and prevents source-set leakage through a PCA fit over excluded source graphs).
- Downstream task: tuning-free binary node classification for all three architectures.
- Evaluation: 10-shot, 128 fixed seed-0 episodes on `covid_political`, `election2020`,
  `ukr_rus_suspended`, and `twibot20`; query counts follow the graph catalog (12, 1, 1,
  and 12 per class). ROC-AUC is primary; accuracy and F1 are retained as diagnostics.
- Every result records a hash of the actual support/query nodes. Aggregation refuses the
  grid unless the hashes agree across all architectures. The episode RNG is reset after
  architecture-specific model initialization, and the audit includes global centers,
  sampled neighborhood node IDs, and sampled edges.

Use `MODEL_IDS=ss_ukr_rus` with the Tucker launchers for a one-cell pilot. A full launch
uses all 31 source sets. Only Tucker GPUs 0 and 1 are accepted by the training launcher.
