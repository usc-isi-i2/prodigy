# Execution log

## Repository synchronization

- Local worktree: `/Users/philipp/projects/gfm/prodigy`.
- Local branch: `codex/final-core-three-seed-sync`.
- Fetched `origin` before auditing evidence.
- Pushed the pre-existing three local commits through `8707ca5`.
- Audited 217 pre-existing unstaged artifacts, validated the qualifying analysis
  products, and committed them in scoped groups. The loose `output/` export tree
  remains intentionally untracked.
- Later audit/protocol commits remain local because the remote-write approval
  gate rejected the push after the user had stepped away. Source files were not
  hand-copied to Tucker as a workaround.

## Evidence audit

- Read `AGENTS.md` and the canonical analysis index before inspecting runs.
- Enumerated local result tables, analysis figures, and registered provenance.
- Enumerated Tucker worktrees, `state/`, `log/`, checkpoint manifests, active
  sessions, and GPU occupancy read-only.
- Verified the complete final-core evidence locally:
  `FINAL_EXPERIMENT_EVIDENCE_OK observed=1944/1944`.
- Verified SAMGPT downstream CLS coverage as 279 aggregate rows: 31 physical
  models × 9 targets, each aggregating 3 training seeds.
- Verified the native-source export as 390 rows total: 130 each for PRODIGY,
  VISION, and GILT. The valid VISION portion is five sources × five targets ×
  five checkpoints plus five random-init controls.
- Read the native trainers, not just filenames. This established that VISION is
  label-free feature-similarity SSL while GILT uses target-label episodic
  classification and therefore cannot count as native SSL.
- Audited upstream GILT read-only at
  `/dataMeR1/phil/gfm/upstream/inductnode`, commit
  `ba46cf4ebd1931712854708c221eaba646641785`. Its native
  `src/engine_graphcl.py` implements augmented-view GraphCL with NT-Xent, but
  the clean checkout has no `.pt`, `.pth`, or `.ckpt` artifact. Thus the native
  objective is now registered while all GILT result cells remain missing.
- Verified the GraphSAGE pilot-v1 checkpoint and its 2,000-update native
  link-prediction provenance. Its existing downstream evidence is a narrow
  full-train TwiBot probe.

## New execution

- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-vision-all9`.
- Tucker branch/commit: `codex/vision-all9-finalcore` at
  `16112313acb27652bec70f232fdf1fa80303669f`.
- Launched VISION seed 1 in tmux session `vision-all9-s1` on physical GPU 2.
- Launched VISION seed 2 in tmux session `vision-all9-s2` on physical GPU 3.
- Both runs use nine balanced sources, 2,500 optimizer updates, 10,000 total
  pseudo-episodes, and checkpoints 100/300/900/2,500. Start time was
  `2026-08-25T09:11:41Z`.
- Seed 1 completed at `2026-08-25T09:38:53Z`; seed 2 completed at
  `2026-08-25T09:39:30Z`. Both terminal result files contain exactly five
  downstream CLS rows.
- Replayed checkpoints 100/300/900/2,500 for seeds 0/1/2 in tmux sessions
  `vision-traj-g2` and `vision-traj-g3`. The result has 12 files and 60/60
  logical cells, with one episode fingerprint per target across all cells.
- Reused SAMGPT all-nine GraphCL checkpoints 20/60/180/500 for seeds 39/40/41
  and evaluated all nine registered downstream targets in tmux sessions
  `samgpt-sat-g2` and `samgpt-sat-g3`. GPU 3 completed at
  `2026-08-25T09:43:10Z`; GPU 2 completed at `2026-08-25T09:43:51Z`.
  The result has 12 files and 108/108 logical cells with one episode fingerprint
  per target.
- GPUs 0, 1, and 4–7 were never selected by a launched command.

## Adaptation implementation and checks

- Added the unified frozen-encoder protocol and extractors for three PRODIGY
  seeds, three VISION seeds, three SAMGPT seeds, GraphSAGE pilot-v1, raw logistic
  regression, and a raw-feature MLP.
- Added full-grid analysis for learning curves, label-efficiency AUC, and
  updates-to-95%.
- Unit tests pass (`3 passed`), all Python files compile, shell syntax checks
  pass, and `git diff --check` is clean.
- A Tucker pre-launch artifact smoke check found that `static_train` is absent
  from several classification graphs. The extractors were corrected before any
  result was produced to use the same canonical `graph.edge_index` across all
  topology-using encoders.

The adaptation run location will be added when those outputs exist.
