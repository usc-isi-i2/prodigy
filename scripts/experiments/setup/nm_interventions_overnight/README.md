# Source-held-out NM intervention campaign — 2026-09-04

Authorized goal starts 2026-09-04 09:19:21 UTC; eight-hour reporting deadline is
2026-09-04 17:19:21 UTC. GPUs 0–3 only. User authorizes execution, monitoring,
commits/pushes/pulls, and output collection. No CLS/LP or further training seeds yet.

## Frozen protocol

TwiBot-20 is the unseen graph because the graph catalog explicitly supports NM,
classification and static LP. Its features/topology may be resident in the immutable
all-nine block-concatenated artifact, but no training/selection samples may touch its
component. Every training batch checks real-node source IDs. The source order is the
existing order A with TwiBot-20 removed: ukraine, covid, midterm, covid-political,
election2020-political, ukraine-suspended, hongkong, facebook-page-reference.
Freeze this choice before outcomes. Graph IDs are resolved from artifact metadata.

Artifact: `/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt`.
Context and training positives: static_train. Stopping/selection: static_validation
on active training sources only. Final evaluation: static_test on all nine sources.
Use the existing immutable split, never generate splits per intervention.

Independent initialization at every rung; seed 0. Baseline uses 256-dimensional
S,U,M GraphSAGE, 30-way/3-shot/4-query episodes, batch 1, AdamW lr 0.001,
weight decay 0.001, balanced confined episodes, 2-hop 9/9 fanouts, 101-node limit,
one-hop NM positive walks. Batch 1 follows the actually measured shared-trainer
protocol and avoids assuming unmeasured batch-4 speed. All methods use the same
baseline; historical batch-4 results are not treated as paired controls.

Maximum 10,000 episodes, validation every 2,000, 16 fixed episodes per active
source, macro-source ROC-AUC. Stop after two checks without improvement exceeding
0.001; retain the actual highest validation checkpoint (earlier exact ties win).
Final cap always triggers validation. No test evaluations in the training loop.
Validation batches and RNG streams are frozen, stored in memory, cloned on replay,
and fingerprinted. Common two-hop evaluation applies to all training arms.
Checkpoints at validation steps preserve matched-budget comparisons. Plateau is
operational, not proof of asymptotic convergence. The budget arm caps at 1250*rung.

## Arms and attribution

`plan.py:ARMS` defines each concrete intervention. 17 methods (baseline plus 16
individual arms), eight rungs = 136 physical runs before combination. The eight-source
endpoint comes first. The 64-episode blocked schedule is cyclic; stopping occurs only
after multiple full source cycles. Its exact realized exposure is logged and compared.

- exposure: proportional instead of balanced source probability.
- schedule: 64 consecutive episodes per source, round-robin cycles.
- composition: cross-source classes with balanced source marginals, constrained pools.
- centers: uniform sampling over eligible log2 degree bands, then centers.
- positives: uniform unique neighbors instead of sorted random-walk discoveries.
- negatives: one size-weighted degree band per episode; all competing centers match it.
- context: 1-hop/100 fanout training, common 101-node cap; fixed 2-hop evaluation.
- optimization: unit global episode-gradient norm before AdamW. This tests magnitude
  balancing, not PCGrad/conflict removal.
- alignment: per-node feature standardization. Tests feature-scale alignment, not a
  learned cross-domain semantic map.
- sharing: learned per-source feature affine transforms, identity for absent sources.
- capacity: 512-dimensional encoder, report parameter/runtime changes.
- objective: NM plus weighted masked-node-feature reconstruction. This necessarily
  combines corruption and auxiliary prediction; do not attribute to loss alone.
- region_adaptive: loss-adaptive degree-region sampling (no source labels in weights),
  with 70% uniform-band base and bounded 30% adaptive component. Compare with centers
  to isolate adaptivity; compare with baseline for total effect.
  Updates assign episode loss to degree bands of support/query subgraph centers;
  these are a proxy for the prototype-center bands used for sampling, rather than
  a per-prototype loss estimate.
- coverage: random-start cyclic eligible-center traversal; record locality/order caveat.
- budget: 1250 episodes/source cap, up to 10k at rung eight; endpoint is a replication
  of baseline distribution, not a new endpoint intervention.

Low-degree eligibility is a separate arm: retain centers with at least two distinct
neighbors; if fewer than seven, partition neighbors into disjoint support/query pools
and repeat only within each pool. The same node never appears in both roles for one
class. Repeated queries reduce effective unique examples; record this limitation and
keep evaluation strict and unchanged. Degree>=7 centers retain baseline positives.


Every model contains the same auxiliary head for initialization parity; only the
objective arm trains it. Source affine parameters initialize at identity including
unseen graph rows, which receive no data gradient. The graph wrapper is shared read-only;
all feature modifications act on collated sampled graphs.

## Execution and reproduction

Local branch `codex/nm-interventions-overnight`, local worktree
`/Users/philipp/projects/gfm/prodigy/.worktrees/nm-interventions-overnight`.
Tucker worktree `/dataMeR1/phil/gfm/prodigy-nmi-overnight`.
Base revision `9ea87fe` from `codex/ladder-sampling-profile`.

Generate configs with `python scripts/experiments/setup/nm_interventions_overnight/plan.py`.
Run tests before a full-graph smoke; production cannot start on smoke checkpoints.
Use `experiments/run_shared_graph.py` with explicit config list, GPUs 0 1 2 3,
4 models/GPU initially, total 64 CPU workers, and a fresh absolute run directory.
Set PATH/conda/prodigy/LD_LIBRARY_PATH and WANDB_MODE=offline inside tmux command.
Actual production uses GPUs 1–3, four models/GPU and 48 total loader workers: GPU 0
rejected even a standalone CUDA allocation at preflight. GPUs 4–7 are untouched.
A failed batch is inspected before replay. Completed configurations/checkpoints are
retained; any retried partial run has a fresh ID. Never pull a running worktree.

First all individual NM evaluations; select compatible positives using training-source
validation only. Freeze combination flags before opening unseen-graph final metrics.
Require average paired validation delta >0.001, positive signs on at least half of
rungs, and no average per-source regression below -0.01. For region_adaptive, also
compare against centers. Prefer the larger validation gain for mutually exclusive
center-policy alternatives. Do not combine a budget diagnostic with a sampler recipe.
Resolve conflicts greedily by descending validation gain (arm name breaks exact ties).
Blocked scheduling conflicts with proportional exposure and cross-graph composition;
cross-graph composition conflicts with episode-wide degree-matched negatives.
Low-degree eligibility and uniform positives are alternative positive-construction
policies. Record every compatibility exclusion in the frozen selection artifact.
If no eligible improvements exist, document baseline as retained and skip redundant
combination training. Combined success must beat baseline and best single arm.

Final NM: 512 fixed 30-way/3-shot/4-query episodes per target, fixed source-independent
random seeds, shared target-major caches, common metric implementation and fingerprints.
Before testing each selected checkpoint, the evaluation runner replays its training-source
validation panel and requires identical episode fingerprints, accuracy/loss within
1e-6, and ROC-AUC within 1e-5 (100 times below the practical-effect threshold;
nearly tied ranks can swap under CUDA scatter rounding). `run_evaluation.sh
--validation-only --replay-repeats 3` quantifies repeat variation without test results.
Report all cells plus included-source, not-yet-included and TwiBot-20-only summaries.
Do not compare these numbers to historical figures without metric/panel parity.
Primary endpoint conclusion uses paired source-macro delta >0.001; negative < -0.001;
otherwise inconclusive. Missing required cells means incomplete. Single-seed results
are exploratory; do not manufacture confidence intervals from historical seed flags.

Analysis/evidence belongs under
`scripts/experiments/analysis/transfer/ablations/prodigy_nm/nm_interventions_overnight/`.
Store effective configs, exact revisions, manifests, selected checkpoints, curves,
source exposure, per-cell timings, fingerprints and result tables. Record manual curve
inspections and operational interventions in an append-only analysis RUN_LOG.
