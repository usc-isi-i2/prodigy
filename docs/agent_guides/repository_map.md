# Repository Map

Use this map to find the current source of truth before adding files or reviving old
work.

## Code and experiments

- `experiments/`: training engine, parameters, layers, and sampler. This is model
  code, not the per-experiment setup tree.
- `models/`: encoder and metagraph model implementations.
- `data/`: datasets, episode construction, and dataloading.
- `scripts/experiments/setup/<name>/`: configs and run/eval launchers.
- `scripts/experiments/analysis/<area>/.../<name>/`: analysis code, findings, data,
  and figures. `scripts/experiments/analysis/README.md` is the canonical index.
- `scripts/experiments/setup/covid_ukr/`: COVID/Ukraine merged experiments.
- `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`: general checkpoint evaluator.
- `scripts/eval/pair_link_eval.py`: valid static-link-prediction evaluator.
- `scripts/harness/`: shared analysis/export and experiment harnesses.
- `docs/graph_catalog.json`: canonical graph registry.

Cross-experiment conclusions live under
`scripts/experiments/analysis/synthesis/cross_experiment/`.

## Archived analyses

Two archive sets are distinct:

- `scripts/experiments/analysis/archive/` contains superseded work retained in the
  working tree. Its `README.md` explains the split.
- Twenty-three analyses removed on 2026-07-26 exist only on branch
  `archive/retired-analyses-2026-07` and tag
  `archive/retired-analyses-2026-07-26`.

Tag `pre-cleanup-2026-07-26` captures the pre-consolidation repository state.

## Materials outside the repository

Paper planning lives at `/Users/philipp/projects/gfm/paper`, outside Git. It includes
`state_doc.md`, `directions_jul26.md`, `LoG_extended_abstract/`, `related_work/`, and
superseded drafts under `archive/`. Put prose planning there, not in
`scripts/experiments/setup/` or `docs/`.

There is no current `slides/` tree. Presentation decks and their build scripts were
removed on 2026-07-26 in revision `6dd1635`; recover them from Git only when needed.

## Tucker

- Repository: `/dataMeR1/phil/gfm/prodigy`
- Data root: `/dataMeR1/phil/data`
- State and logs: worktree-local `state/` and `log/` directories
