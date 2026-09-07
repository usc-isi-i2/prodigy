# Fresh-chat handoff — 7 September 2026

## Status and authority

The user stopped both overlapping chats and will continue in a fresh chat.
Do not resume either old chat, its goal, or its automation. This thread's goal
was marked blocked solely to stop automatic continuation while ownership was
discussed; the publication goal was NOT achieved or abandoned.

The new chat should become the single coordinator. Explicitly delegate bounded
tasks if useful, but do not let two independent chats own the same experiment.
User wants a credible high-impact ICLR contribution, not repeated simulated
scores, endless controls, or a conveniently narrowed definition of success.

Final read-only Tucker check for this handoff: no tmux server; no processes
matching `centered_ridge_training`, `run_temporal_roles`, or `public_prodigy_kg`.
The comparator evaluation status is complete (5 targets × 2 streams × 2 schedules).
This is a scoped process check, not a statement about every user's cluster job.
No jobs were started/stopped, no manuscript expanded, and nothing pushed for
this handoff. Do not restart completed training.

## Read these first, in order

This directory is abbreviated **LEAF** below:
`/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology/scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms`.

1. `ARGUMENT_CURRENT_DECISION.md` — latest concise figure-led scientific argument,
   including the completed direct-training result. Start here, not older pitches.
2. `CONTRIBUTION_REQUIREMENTS.md` — advisor decisions and latest numeric comparisons.
   It is chronological: read through the final completed-outcome section.
3. `ARGUMENT_REPLICATION_DECISION.md` — public two-seed crossover, failed role bridge,
   failed GILT scope test, and paired empirical sensitivity ranges.
4. Producing comparator findings:
   `/private/tmp/prodigy-centered-ridge-training/scripts/experiments/analysis/graphs/transfer_prediction/centered_ridge_training/FINDINGS.md`.
   HEAD at handoff: `873561ec`, branch `codex/centered-ridge-training`.
5. Public evidence:
   `/private/tmp/prodigy-publickg-paired-state/scripts/experiments/analysis/graphs/transfer_prediction/public_prodigy_kg/FINDINGS.md`.
   HEAD `4dc54247`, branch `codex/publickg-paired-state`.
   **Its final paragraph still says the direct comparator is pending. This is
   stale; the completed comparator findings and latest LEAF notes supersede it.**

## Current scientific conclusions — do not reset to earlier hypotheses

### Public component result replicates, but is scoped

Original-style Wiki → FB15K-237 PRODIGY, native batch-statistics inference,
128 exactly paired episodes, two training initializations. Early/late checkpoint
names 2000/8000 correspond to 2,001/8,001 completed updates.

| Encoder / inference | Seed 0 accuracy | Seed 1 accuracy |
|---|---:|---:|
| Early / early | 77.4023 | 73.7988 |
| Early / late | 78.0957 | 75.3516 |
| Late / early | 73.1055 | 71.7090 |
| Late / late | 75.0391 | 73.4082 |

Later inference helps either encoder; later encoder output hurts either inference
module. A nearly flat total can hide opposing changes. This is fixed-component
attribution, not evidence of compensating training gradients or unchanged
information. Intermediate ridge remains strong. Seed-one 500-episode accuracy:
trained centered ridge 78.3425%, native 72.3825%, strongest tested initialization
readout 72.6675%, strongest text readout 71.6100%. Native NLL is better.
Two seeds on the same episodes are not two targets or 1,000 independent episodes.

Running-statistics normalization retains a readout advantage; the crossed
component directions were NOT tested under that normalization. Paired empirical
episode-reweighting ranges and leave-one-episode-out checks preserve all eight
component signs. These are not entity/domain/training-seed confidence intervals.

### Three attractive explanations failed

- **Public support-specific temporal bridge:** mixed support/query encoder ages
  give roughly 40–51% accuracy versus 73–78% for matched ages. Late supports hurt
  early queries but help late queries, both seeds. This rejects the nominated
  role-independent damage explanation. Off-distribution mixed inputs do not
  establish coordinate drift as the cause of natural deterioration.
- **Architecture-general compensation:** nominated GILT COVID Political → TwiBot20
  test failed; native conditions were near chance and metric directions disagreed.
  This was a scoped native-component adapter, not a full controlled official GILT
  benchmark comparison. Do not replace its target to seek a favorable replication.
- **Special training benefit from learned inference:** the strengthened direct
  objective closes/reverses the old advantage. Fixed-four macro-F1, direct minus
  native-trained encoder under the SAME centered readout: +1.466/+1.141 points
  original blocked/interleaved, +0.846/+1.038 fresh. Versus joint training:
  +2.561/+1.266 original, +1.365/+0.676 fresh. One training seed. Centering and
  learned source-training scale changed together; their separate effects are
  unknown. No necessity claim and no new auxiliary-head principle.

### Latest practical accounting

Changing the readout on a native-trained encoder supplies 1.78–4.13 macro-F1
points on the fixed four-target panel. Changing training then adds 0.85–1.47;
total 3.02–5.17 across the four stream/schedule cells. Thus 55–80% of this
measured gain requires no retraining. This is an exact readout-first score
decomposition, NOT an order-independent causal mediation fraction.
Election gains essentially nothing. Direct retraining hurts fresh blocked
Facebook by 0.39 points and fresh interleaved suspension by 3.14, compared with
native training under the common readout. All five targets remain visible.

### Social localization remains valid but not a universal public mechanism

Nine-source × five-target × two-stream support-edge map exists: Hong Kong is
not the only affected source. Political foreign-source gains occur for 5/8,
Facebook for 3/8; bot suppression harms all nine. This is not nine-source K/V
evidence. Social support-value interventions can change performance with attention
and final queries fixed: values help Hong Kong political transfer but hurt Ukraine
bot transfer. The public two-layer inference architecture need not share that
same fixed-query pathway. Keep raw-text and untrained controls prominent.

## Artifact map and verification scope

All relative paths in this list are under LEAF unless explicitly absolute.

- Current figures: `figures/readout_training_accounting.png`,
  `figures/centered_training_decision.png`,
  `figures/crossover_replication_sensitivity.png`,
  `figures/crossover_and_role_boundary.png`. All visually inspected locally.
- `compare_centered_training.py`: compares saved metrics, checks unique condition
  inventories and raw-readout parity, recomputes panels and all additive identities.
  `data/centered_ridge_eval_20260907/summary.json` is a **compact export**, not the
  byte-identical full summary. Full source hash:
  `ec6339f8fa96e0a0df0b1e537894a48993cdb75207aa44bcdfd69abdc557284c`.
  Independent receipt audit: 640 output + 320 unique input files, no mismatches.
  This audit did not independently rescore every prediction tensor.
- Full comparator summary/protocol/status are in the producing comparator analysis
  above, and Tucker `/dataMeR1/phil/gfm/prodigy-centered-ridge-eval/log/centered_ridge_eval_20260907`.
- Training root: `/dataMeR1/phil/gfm/prodigy-centered-ridge-training/log/centered_ridge_training_full_20260907`.
  Both 2,500-update arms passed exact initial-model and consumed IDs/roles/topology
  matching, 625 episodes/source. Not an every-feature-byte audit.
- Public summaries/protocols/statuses: `public_prodigy_kg/data/` in the public
  producing worktree; full tensors under its Tucker `log/` equivalent.
  `verify_temporal_evidence.py --seed 0` / `--seed 1` verifies pinned summaries
  and row means, not full inference. Cross-seed 500-input audit passed exactly.
- Temporal roles code: `/private/tmp/prodigy-publickg-temporal-roles`, branch
  `codex/publickg-temporal-roles`, HEAD `d10aab89`; Tucker same basename.
  Results under Tucker `log/roles_seed0_20260907` and `roles_seed1_20260907`.
  Local compact results: `data/temporal_roles_20260907.json`.
- Actual-example evidence: `FINDINGS_PUBLIC_CROSSOVER_BRIDGE.md`,
  `analyze_public_crossover_examples.py`, `inspect_public_crossover_examples.py`,
  `data/public_crossover_examples_20260907.json`,
  `data/public_crossover_semantic_examples_20260907.json`.
  Includes correct/wrong pattern accounting and concrete relation examples;
  examples were selected by fixed ordering, not claimed representative prevalence.
- Social decisions: `DECISION_KV_GENERALITY.md`, `FINDINGS_CLASS_REFERENCE_KV.md`,
  `FINDINGS_MESSAGE_SCALE_DECISION.md`, `FINDINGS_LABEL_INTERFACE_DECISION.md`.
- Prior-art boundary: `FINDINGS_TEMPORAL_NOVELTY_BOUNDARY.md`; generic probing,
  stitching, compatibility, and representation/readout decomposition are prior art.
- Retrospective source-choice consequence: `FINDINGS_SOURCE_CHOICE_CONSEQUENCE.md`.
  TwiBot source-choice advantage persists fresh, suspension reverses; not prospective
  held-out selection and not a general successful selector.
- Broader experiment map: `scripts/experiments/analysis/README.md` in the repo.
  Preserve singleton/pair/LOO/mixture evidence; do not rebuild from scratch.

## Closed avenues and manuscript state

Do not restart support-CV selector variants, norm/direction control accumulation,
alignment rescue, replacement GILT targets, or centering/scale factorization just
to rescue rejected stories. Failed source-geometry predictors remain failed.
Long-training social results have label-interface/collapse qualifications; do not
cite them as clean replications without their findings files.

Paper directory:
`/Users/philipp/projects/gfm/paper/transfer-prediction/mechanism-draft-2026-09-06`.
It contains multiple manuscripts and PDFs. **Subsequent explicit user request
to update the PDF was completed on 7 September:**
`output/pdf/class_reference_transfer_draft.pdf` is now titled *Separating Encoded
Signal from Learned Inference in Graph Transfer*. It incorporates the two-seed
crossover, failed temporal-role/GILT predictions, strengthened direct training,
readout-first accounting and scoped social pathways. It has six main-text pages
and twelve pages total; all were rendered and inspected, with zero overfull boxes
or unresolved citations. Editable inputs and builder were updated alongside it.
The former PDF/sources are preserved in `archive/pre_encoded_signal_update_20260907/`.
Final PDF SHA-256: `2b7e3ecd173fa909d12c78384c53ec7bd42cf5f233ded9f77be7d47a1d6a6dfd`.
Older alternative manuscripts remain stale. The complete historical control
appendix is still not integrated, and this update does not establish submission
readiness or attainment of the publication goal. No experiments or pushes resumed.
Reviewer artifact access remains TBD; do not release private artifacts or remove
required factual disclosures by assuming permissions/author approval.

## Preservation and operational cautions

LEAF worktree: `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`,
branch `codex/role-topology-interactions`, HEAD at handoff `65946f82`.
Latest notes, scripts, compact data and figures are largely **untracked/private**.
A fresh clone or git pull will NOT recover them. Keep this worktree intact.
Do not clean, reset, delete, stage broadly, or relocate without preserving them.
Root checkout is a different dirty branch (`codex/strong-mlp-baseline`); leave it
alone. Several `/private/tmp` producing worktrees contain important evidence;
preserve those too. No backup outside the existing worktrees was made here.

Read applicable AGENTS.md. Use Tucker for heavy work, dedicated worktrees/tmux,
and git for source transfer. Own GPUs 2/3 per this experiment worktree; avoid others.
Check processes before changing a checkout. Stage explicit paths, pull before
push, never force-push shared branches. Prior permission here was code/tests-only
push, not private results/prose/figures; do not infer release permission from a
different worker's commit history or assume everything is private because local
files are private. Verify repository visibility separately if relevant.

## First action for the new sole coordinator

Read this handoff and the latest argument, reconcile the other chat's final
handoff if supplied, and present ONE advisor decision: which defensible conclusion
could carry the paper and what single missing piece genuinely changes that decision.
Do not launch another experiment just because previous ones are finished.
If pursuing the direct-training practical claim, the recorded next gate is an
exact second-initialization direct/native replication, not objective tuning;
it has not been launched here. The publication objective remains open.
