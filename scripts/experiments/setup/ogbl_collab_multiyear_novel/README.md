# ogbl-collab multi-year novel expert

This bounded treatment tests whether chronological diversity improves the compact
joint model's forward novel-link ranking. It reuses the audited fresh-negative BCE
seed controls and changes only the training schedule: treatment steps cycle equally
through 2015, 2016, and 2017 episodes. Every episode has its own strictly historical
adjacency, features, positives, and deterministic fresh-negative panel. The model
weights are shared; no later graph is used to encode an earlier episode.

The three treatment seeds train for 2,000 total updates, use the existing optimizer,
sampling, hard-mining, architecture, standardization, and 2018 earliest-maximum
standalone Hits@50 selection. Advancement requires positive seed-matched gains over
the archived 2017-only controls in all three seeds, mean gain at least 0.5 point,
and positive novel net hits in every seed. No 2019 access or refit is authorized.

Run `run.py dry-run` before `prepare`. Then run `train --seed 0/1/2` on separate
owned GPUs, followed by `aggregate`.

