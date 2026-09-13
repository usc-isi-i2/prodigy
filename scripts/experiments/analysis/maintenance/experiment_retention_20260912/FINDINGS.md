# Experiment retention and findings index — 2026-09-12

Scope: 29 local mixture-scaling worktrees and 42 registered Tucker worktrees. This includes the conversation’s older MLP diagnostics and the recent mini/decoder/bias/gap experiments. Other PRODIGY repository edits are outside this cleanup. The live gallery worktree has been reused by another task and is retained.

## Evidence preservation

- Existing experiment narratives are archived verbatim under `data/narratives/<worktree>/`; their validity caveats remain authoritative.
- Five ignored raw-decoder baseline files are rescued under `data/decoder_raw_baseline/`. The ignore cause was `results/*/raw/`.
- Dirty local snapshots include staged/unstaged binary patches and complete untracked files under `data/local_changes/`.
- Twelve dirty Tucker worktrees are preserved in `data/tucker_dirty_snapshots.json.gz`: original HEAD, status, separate staged/unstaged binary patches, and untracked file bytes with SHA-256. This is a recovery archive, not an adjudicated merge or newly validated scientific result.
- Legacy standalone activation, ablation, untrained-model and AUC/BCE reports are extracted under `data/recovered_reports/`.
- Checkpoints, graph caches, W&B offline runs and Tucker run directories are retained. No claim that deterministic retraining reproduces a checkpoint bit-for-bit.

## Experiment findings map

| Experiment family | Findings / interpretation | Preservation |
|---|---|---|
| Original MLP ladder, throughput, activation geometry, feature histograms, coordinate KS, Suspended CSV repair | `data/narratives/mlp-session-findings/results/mlp_ladder_diagnostics_20260911/FINDINGS.md` | Historical corrupted-Suspended caveat applies; do not pool with corrected mini results. |
| Input-dimension ablation and untrained baseline | `LEGACY_DIAGNOSTICS.md` and `data/recovered_reports/` | Historical checkpoint; descriptive ablation, not mutual-information estimates. |
| GraphSAGE/GraphMAE and context-MLP pilots | Per-worktree `results/RESULTS.md` and `results/STRICT_STUDY.md` snapshots | Dirty GraphMAE state preserved separately; not automatically merged. |
| Nonzero mini FR and original LP | `data/narratives/mixture-bias-lp/results/nonzero_mini_transfer/FINDINGS.md` | Original neighbor results superseded due context overlap. |
| Disjoint neighbor retraining | Same directory `DISJOINT_CONTEXT.md` | Degree-matched evaluation is historical. |
| Uniform evaluation | Same directory `UNIFORM_EVAL.md` | Frozen models; different negative distribution and ratio from older evaluation. |
| Midterm decoder ablation | `data/narratives/mixture-decoder-test/results/decoder_bias_midterm/FINDINGS.md` | Raw-control artifacts rescued in this archive. |
| Bias LP matrices | `data/narratives/mixture-bias-lp/results/nonzero_mini_transfer/BIAS_LP.md` | 18 models, 162 cells; selected checkpoints retained on Tucker. |
| Fixed-panel gap audit | `data/narratives/mixture-gap-audit/results/lp_gap_audit/FINDINGS.md` | Positive-edge generalization gap; no model retraining. |
| Newer sequential/interleaved/async studies | Per-worktree narratives in `data/narratives/mlp-pair-error-code/`; remaining source worktrees retained | Outside the six removals; do not equate code presence with completed findings. |

## Local worktree disposition

| Worktree | Branch / pinned revision | Disposition | Archived narratives |
|---|---|---|---|
| `/Users/philipp/projects/gfm/mixture-scaling` | `codex/social-source-completion` / `7f8ec812d0e8` | Retain: unrelated dirty work preserved | 6 |
| `/private/tmp/coordinate-ks` | `codex/coordinate-ks` / `e14e39cef561` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/feature-dimension-ks` | `codex/feature-dimension-ks` / `8ad08fbbdb99` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mixture-bias-lp` | `codex/bias-lp-matrices` / `d944d7ba61c0` | Remove redundant checkout after archive is committed | 10 |
| `/private/tmp/mixture-decoder-test` | `codex/decoder-bias-test` / `c82f8c89ea3d` | Remove redundant checkout after archive is committed | 9 |
| `/private/tmp/mixture-disjoint` | `codex/disjoint-neighbor-lp` / `0a473096efef` | Remove redundant checkout after archive is committed | 7 |
| `/private/tmp/mixture-gap-audit` | `codex/lp-gap-audit` / `89ca0a001290` | Remove redundant checkout after archive is committed | 11 |
| `/private/tmp/mixture-mini-lp` | `codex/mlp-mixed-batch-pilot` / `74419e883a46` | Retain: current/other task or not approved in this batch | 10 |
| `/private/tmp/mixture-mini-transfer` | `codex/mini-transfer-fast` / `9b54a0ab01d5` | Remove redundant checkout after archive is committed | 5 |
| `/private/tmp/mixture-uniform-eval` | `codex/uniform-mini-eval` / `0fa26a6e15e9` | Remove redundant checkout after archive is committed | 8 |
| `/private/tmp/mlp-activation-inspect` | `codex/mlp-activation-inspect` / `86ffcfd88b92` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-auc-bce` | `codex/mlp-auc-bce` / `0b6ffeff9e1e` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-dimension-ablation` | `codex/mlp-dimension-ablation` / `170ae5ea167f` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-feature-hist` | `codex/mlp-feature-hist` / `936ab5d56078` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-input-study` | `codex/mlp-input-study` / `337a30308a4c` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-input-study-report` | `codex/mlp-input-study-report` / `1dc3241cb673` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-integration` | `codex/mlp-integration` / `3e9ca70c7785` | Retain: current/other task or not approved in this batch | 5 |
| `/private/tmp/mlp-leaky-test` | `codex/mlp-leaky-test` / `79a84dc3d8d0` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/mlp-metrics-checkpoints` | `codex/mlp-metrics-checkpoints` / `7f8ec812d0e8` | Retain: current/other task or not approved in this batch | 5 |
| `/private/tmp/mlp-pair-error-code` | `codex/mlp-input-shift-audit` / `2effe52373bc` | Retain: current/other task or not approved in this batch | 17 |
| `/private/tmp/mlp-session-findings` | `codex/mlp-session-findings` / `d91984bbc824` | Retain: current/other task or not approved in this batch | 5 |
| `/private/tmp/mlp-untrained-baseline` | `codex/mlp-untrained-baseline` / `03f25ce5a0ce` | Retain: current/other task or not approved in this batch | 3 |
| `/private/tmp/suspended-repair` | `codex/suspended-csv-repair` / `b29c3fb26ed8` | Retain: current/other task or not approved in this batch | 3 |
| `/Users/philipp/projects/gfm/mixture-scaling-balanced-pilot` | `codex/graphsage-balanced-pilot` / `d14d79b9c6e1` | Retain: current/other task or not approved in this batch | 2 |
| `/Users/philipp/projects/gfm/mixture-scaling-graphmae` | `codex/sage-graphmae-source-lattice` / `70184d47aea1` | Retain: unrelated dirty work preserved | 2 |
| `/Users/philipp/projects/gfm/mixture-scaling-mlp-fast` | `codex/mlp-fast-scheduler` / `d4195ed3e847` | Retain: current/other task or not approved in this batch | 3 |
| `/Users/philipp/projects/gfm/mixture-scaling-mlp-overlap` | `codex/mlp-overlap-eval` / `6b6336940f9a` | Retain: current/other task or not approved in this batch | 3 |
| `/Users/philipp/projects/gfm/mixture-scaling-node-only` | `codex/node-only-transfer` / `9116bb4345c7` | Retain: current/other task or not approved in this batch | 2 |
| `/Users/philipp/projects/gfm/prodigy/.worktrees/graphsage-all9-lp` | `codex/graphsage-all9-lp` / `f3be39582b9d` | Retain: current/other task or not approved in this batch | 2 |

## Recreate a removed checkout

Keep all branches. From the mixture-scaling main repository:

```bash
git worktree add /tmp/<name> <branch>
```

Pinned commit IDs and current remote containment are recorded in `data/local_worktrees.json`. Recreate at the pinned commit with `git worktree add --detach /tmp/<name> <sha>` if the branch has advanced. Current gallery stays at `/tmp/mixture-mini-lp/results/nonzero_mini_transfer/gallery.html`. Removed local file links require restoring that checkout or opening the archived narrative.

## Remaining state is intentional

Original dirty worktrees remain dirty: their full changes are recorded, not silently committed into someone else’s branch or discarded. Thus this is an inventory with preserved recovery data, not a claim that every repository is clean. Unreviewed experiment outcomes are retained rather than fabricated. No adaptation-efficiency experiment was run in this conversation; it remains a discussed proposal.

Tucker inventory records runtime sizes and dirty state. No tmux sessions or compute processes were present at inspection. No Tucker worktree, graph artifact, checkpoint or offline run is scheduled for deletion in this batch.
