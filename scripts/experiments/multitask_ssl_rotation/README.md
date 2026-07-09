# Multi-task SSL rotation — one encoder, all SSL tasks

**Status: proposed — nothing run yet.** Standalone experiment (sibling to, not part
of, `topology_feature_ssl`'s six-arm reading chain).

## Question

If we pretrain **one** encoder by **rotating over all our SSL tasks — one task per
episode** — does its frozen representation transfer **better across all graphs and
all downstream tasks** (node classification, node regression, static link
prediction) than an encoder trained on any **single** SSL objective?

## Tasks in the rotation

Three self-supervised objectives, all defined on the retweet graphs and all sharing
the same metric-episode structure (so they mix cleanly on a shared encoder):

| task | what it learns | sampler | augmentation | loss |
|---|---|---|---|---|
| **NM** (`neighbor_matching`) | instance/neighborhood discrimination (feature shortcut + local topology) | `NeighborTask` | none | metric (multiway) |
| **CL** (`contrastive`) | invariance to feature corruption (two-view instance discrimination) | `ContrastiveTask` | `NZ0.2` (2 views) | metric (multiway) |
| **FP** (`masked_feature_prediction`) | reconstruct masked node features (generative) | `ContrastiveTask` | `NZ0.3` (mask) | reconstruction (aux MSE head) |

NM and CL share the metric loss; FP swaps in a reconstruction head over the masked
nodes. All three run through the identical forward — only the augmentation and the
loss differ per episode.

## Arms — four

| arm | task_name | prefix | budget |
|---|---|---|---|
| **NM** | `neighbor_matching` | `mtr_NM` | 40k episodes |
| **CL** | `contrastive` | `mtr_CL` | 40k episodes |
| **FP** | `masked_feature_prediction` | `mtr_FP` | 40k episodes |
| **MIX** | `nm_fp_cl` (rotation) | `mtr_MIX` | 40k episodes total (~13.3k/task, 1:1:1) |

NM/CL/FP are the single-objective **controls**; MIX is the **treatment**. Budget is
**matched total compute** (all arms 40k) — MIX sees ~⅓ the per-task exposure of each
control, which is the intended "same compute, mixed vs pure" comparison. (Caveat: a
MIX win could be understated because each of its objectives trained on ⅓ the data; a
matched-per-task 120k MIX is the natural follow-up if MIX is promising.)

## Fixed setup (identical across all four arms)

- **Pretrain corpus:** the 3-way merged retweet graph
  (`/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt`), one
  seed. Same corpus as `topology_feature_ssl`, so the three in-domain eval sources
  (ukr_rus, covid, midterm) are *in* the mix — the clean **transfer** read is on the
  held-out datasets (twibot20, election2020); treat in-domain as fit.
- **Encoder:** bio-only 768-d GTE embeddings, mean-agg SAGE, 1 layer / 1 hop,
  emb_dim 256, undirected message passing. No structural inputs (unlike E1). All four
  arms share this exact encoder — they differ **only** in the SSL objective.
- **Episode sampling — GLOBAL** (no within-source confinement). `ContrastiveTask`
  (CL/FP) has no strata support, so global sampling is the only regime all three
  objectives can share; using it for NM too keeps the arms single-variable (objective
  only). This differs from `topology_feature_ssl`'s within-source-balanced NM, so
  `mtr_NM` is **not** the same encoder as B0 — the controls here are internal.
- **`batch_size: 1`** — one task per gradient step, which is what makes per-episode
  rotation and per-episode loss dispatch exact (enforced for MIX).
- **Eval — the frozen-encoder benchmark**, one sweep per arm: node classification
  (10-shot), node regression (10-shot, log1p, 3 targets), static link prediction
  (0-shot). Metrics: cls → acc/AUC, reg → Spearman, static-LP → ROC-AUC.

## How the rotation works (mechanism)

`MIX` uses the original PRODIGY `MultiTaskSplitBatch([NeighborTask, ContrastiveTask,
ContrastiveTask], ["nm","cl","fp"], counts)` — each episode is assigned one task,
round-robin over a count-weighted (shuffled) schedule. Per episode:

1. **Sampler** switches (nm→NeighborTask, cl/fp→ContrastiveTask) — handled by
   `MultiTaskSplitBatch`, which tags every episode's labels with its task name.
2. **Augmentation** switches — the `Collator` reads the task tag and applies
   `aug_by_task[name]` (nm→identity, cl→`NZ0.2`, fp→`NZ0.3`).
3. **Loss** switches — the `Collator` sets `graph.mix_is_fp`; the trainer dispatches
   the reconstruction loss on fp episodes and the metric loss on nm/cl episodes.

Val/test fall back to **pure NM** as a coherent pretrain monitor (mixing
reconstruction and metric scores in one accumulated eval is meaningless); checkpoint
selection uses NM val accuracy. The real comparison is the downstream sweep.

## Eval — compare on all graphs / all tasks

`run_eval_sweep.sh` freezes each arm and runs the joint benchmark over the focused-5
datasets, keyed by `model = arm`. The headline is a table subtraction:

**T1 — transfer benchmark.** Rows = arms; columns = tasks × (in-domain vs held-out):

| arm | cls (acc/AUC) | reg (Spearman) | static-LP (ROC-AUC) |
|---|---|---|---|
| NM | | | |
| CL | | | |
| FP | | | |
| **MIX** | | | |

## What we can conclude — the reading

The point is the **joint** criterion, not any single task. Report the per-task vector
and score arms by `min(feature_score, topological_score)` (feature = cls/reg,
topological = static-LP), never a single mean.

- **MIX − max(NM, CL, FP)** on the joint `min(...)` bar is the headline. MIX clears it
  ⇒ **rotating over heterogeneous SSL objectives buys an encoder that is good at
  everything, which no single objective delivered** — the multi-task union is worth
  the (matched) compute.
- **Per-task attribution of the single-objective controls:** NM/CL are expected
  strong on feature tasks (cls) and weak on regression (instance discrimination
  collapses continuous variation); FP (generative) is expected the reverse — stronger
  on regression, weaker on discrimination. If **MIX ≈ the per-task max of the three
  controls on every task**, the rotation is successfully inheriting each objective's
  strength. If **MIX < the best control on some task**, mixing diluted that objective
  (the matched-total-compute caveat) — motivating the 120k matched-per-task follow-up.

## Possible headline stories

- **"Rotation is a free lunch"** — MIX matches/beats the best single objective on
  every task at equal compute. One encoder, all tasks; no need to pick an objective.
- **"Rotation trades off"** — MIX is a Pareto middle: better `min(...)` than any pure
  objective but below each objective's own specialty. Still the best *general*
  encoder, but specialists win their own task.
- **"Dilution dominates"** — MIX underperforms the controls broadly; at this budget,
  ⅓ per-task exposure hurts more than cross-task transfer helps. Bounded negative
  result → run the 120k matched-per-task MIX before concluding.

## Relationship to topology_feature_ssl

That experiment isolates **one lever per arm** (a clean reading chain B0→…→E4) and its
E4 is a *fixed* 3-objective package (MFR⊕LP⊕structural, summed). This experiment is
the opposite shape: the **maximal-mixture** question (does rotating the whole SSL menu
beat any single objective) with its **own internal controls** and a simpler encoder.
Kept standalone so it does not perturb that reading chain.

## Appendix — deferred

- **120k matched-per-task MIX** (each objective sees 40k) — separates the compute
  effect from the per-task-exposure effect; the natural follow-up if MIX is promising.
- **Unequal rotation weights** (`mix_task_counts`) — e.g. up-weight FP if regression
  is the bottleneck.
- **Add static-LP as a fourth rotated objective** — needs a per-episode link head
  (heterogeneous with the metric/recon heads); larger trainer change, deferred.
- **Diagnostics reuse** — the `topology_feature_ssl` 2×2 ablation + capability probes
  run on these frozen encoders unchanged; fold in if the benchmark read is ambiguous.
