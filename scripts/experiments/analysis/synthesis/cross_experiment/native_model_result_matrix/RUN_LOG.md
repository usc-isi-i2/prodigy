# Execution log

## Repository synchronization

- Local worktree: `/Users/philipp/projects/gfm/prodigy`.
- Local branch: `codex/final-core-three-seed-sync`.
- Fetched `origin` before auditing evidence.
- Pushed the pre-existing three local commits through `8707ca5`.
- Audited 217 pre-existing unstaged artifacts, validated the qualifying analysis
  products, and committed them in scoped groups. The loose `output/` export tree
  remains intentionally untracked.
- After explicit user approval, the audit/protocol/result commits were pushed
  through `43f013e`. Source moved to Tucker only through git; it was not
  hand-copied between the laptop and cluster.

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
- Created isolated Tucker worktree
  `/dataMeR1/phil/social-gfm/code-pilot-v1` at the pilot's recorded commit
  `c6fd912fba1c12b12b8a6e9b0d112b63b0c563a3`. On GPUs 2 and 3, reconstructed
  fixed-budget GraphSAGE states at 0/20/60/100/300/900/2,000 updates under
  `/dataMeR1/phil/social-gfm/experiments/pilot-v1-trajectory`. The rerun
  2,000-update state is tensor-identical to the registered checkpoint, with
  `max_abs_diff=0` and matching state SHA-256. No other GPU was selected.
- Evaluated all seven GraphSAGE prefix states with the existing official-split
  TwiBot frozen-encoder probe on GPU 2. The probe used 8,278/2,365/1,183
  train/validation/test nodes and script SHA-256
  `c59f9cad4f253a512a5be45c588950e13542b2491a4e03932789ce048d6a1be7`.
  The raw JSON/CSV is registered locally; it is explicitly a narrow full-label
  CLS trajectory, not the matched adaptation protocol.

## New execution

- Pushed the complete queue implementation to
  `codex/final-core-three-seed-sync` at `9577507`, then created isolated Tucker
  worktree `/dataMeR1/phil/gfm/prodigy-native-matrix` on branch
  `codex/native-matrix-overnight-runtime`.
- Launched the priority queue at `2026-08-25T14:33:28Z`. Raw, three-seed VISION,
  three-seed SAMGPT, and three-seed PRODIGY representation caches completed on
  GPUs 2/3. The GraphSAGE extractor initially stopped because importing
  `benchmark_run._node_batch` pulled in an unused `pyarrow` dependency.
- Replaced that import with the exact non-counterfactual pilot-v1 tensor adapter,
  updated both terminal and trajectory extractors, passed the focused tests,
  and pushed `ad122e7`. After the active cross-SSL job finished, the idle Tucker
  worktree fast-forwarded to that commit. The resumed GraphSAGE terminal replay
  completed all four target caches in six seconds; completed caches were not
  rerun.
- Used otherwise-idle GPU 3 during PRODIGY extraction to replay VISION cross-SSL
  part B and then part A behind an exact 50-row gate. The collected result has
  75 + 50 = 125 cells and one episode fingerprint per target. GPUs outside 2/3
  were never selected.
- The main adaptation export completed at 3,744 rows. Historical pilot-v1
  trajectory replay then exposed an older three-argument `LinkPredictor`
  constructor; commit `a2c17f0` restores the exact old signature. The resumed
  replay produced 28/28 caches and 2,184/2,184 matched-head saturation rows.
- The master transition over the already-complete cross-SSL results exposed a
  Bash `set -u` same-declaration initialization bug. Commit `1ba4221` split the
  local declarations; the focused Tucker test passed and the transition then
  registered 125 cells without replaying them.
- Adaptation and GraphSAGE saturation outputs were collected locally, validated
  against exact row/model/target/seed/fingerprint contracts, analyzed, and all
  figures visually checked.
- VISION mixture training ran in the same isolated worktree on physical GPUs 2
  and 3 only. It completed at `2026-08-25T17:51:53Z`: 12/12 new terminal
  checkpoints and 48/48 JSONL files, each with exactly five target rows
  (240 new cells). Adding the reused all-nine seed-0 model yields 260/260
  physical cells and 300/300 order-expanded ladder cells. The master tmux
  session exited cleanly after writing its completion marker. Both mixture
  figures were generated and visually checked.
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
- Added the native VISION three-order mixture plan and launcher. It schedules 12
  missing fixed-compute models on GPUs 2/3, reuses the existing all-nine model,
  and preserves all four downstream CLS checkpoints. Execution completed with
  every registered cell present and fingerprint-consistent.
- Added a distinct VISION native cross-SSL evaluator for the existing five
  specialists and five saved checkpoints. It fixes 128 label-free pseudo-task
  episodes per target and rejects fingerprint drift across 125 cells; it does
  not reuse or relabel downstream CLS output. Execution completed at 125/125
  cells without using downstream labels.
- Unit tests pass (`3 passed`), all Python files compile, shell syntax checks
  pass, and `git diff --check` is clean.
- A Tucker pre-launch artifact smoke check found that `static_train` is absent
  from several classification graphs. The extractors were corrected before any
  result was produced to use the same canonical `graph.edge_index` across all
  topology-using encoders.

The matched adaptation export is retained under
`scripts/experiments/analysis/evaluation/adaptation_efficiency/data/`; the
VISION mixture export is retained under this analysis folder's
`data/vision_native_mixture_raw/`.
