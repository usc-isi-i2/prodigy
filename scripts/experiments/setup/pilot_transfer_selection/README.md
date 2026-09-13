# Pilot-transfer source selection

Status: protocol frozen before producing any early-checkpoint cross-target
evaluation. This setup is a staged decision experiment, not evidence that the
proposed selector works.

## Why this experiment exists

The final-core study found that fixed-budget PRODIGY mixtures often track a
strong included specialist. That observation is actionable only if a cheap
specialist pilot preserves the donor ordering seen after full training. This
experiment tests that prerequisite before training another source-mixture grid.

The intended contribution is narrow and falsifiable: target-aware,
self-supervised pilot transfer may be sufficient to choose a source subset more
efficiently than indiscriminately pooling every available graph. The protocol
does not assume that a specialist ensemble equals a jointly trained mixture, and
it does not call an existing terminal-result analysis prospective.

## Existing immutable inputs

- final-core training family: 31 physical source sets, seeds 0/1/2;
- specialist checkpoints: nine sources x three seeds at completed updates
  100/300/900/2500;
- archived Tucker state root:
  `/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core`;
- terminal fixed-test evidence: 512 `static_test` episodes for every
  specialist-source/target/seed cell, already admitted by the final-core audit;
- immutable graph split and model configuration from `../final_core/`.

The early checkpoints exist for all 27 specialists. They have not been evaluated
on a complete nine-target fixed panel. Historical final-core validation evaluated
models only on their active source distributions and cannot answer this question.

## Stage A: prerequisite diagnostic

### Intervention and evaluation

Evaluate every specialist checkpoint at updates 100, 300, and 900 separately on
each of the nine targets using `static_validation`, with message passing restricted
to `static_train`. Use 512 fixed episodes per target, one immutable episode plan
per target, and identical episode fingerprints across sources, checkpoints, and
training seeds. Do not inspect `static_test` while generating or validating the
early matrix.

This produces 9 sources x 3 training seeds x 3 early checkpoints x 9 targets =
729 physical validation cells. Existing update-2500 fixed-test specialist cells
remain the terminal reference; they are not rerun or reselected.

Primary metric: 30-way neighbor-matching accuracy. Macro one-vs-rest ROC-AUC is
secondary because it is close to ceiling in this protocol. Training seed is the
replication unit; episodes are paired measurement occasions.

### Frozen analyses

For each early checkpoint:

1. Compute within-(target, seed) Spearman rank correlation across all nine donors
   against update-2500 test accuracy.
2. Repeat after excluding the target's own specialist.
3. Select the best donor from validation only and report terminal-test top-1
   regret, top-2 coverage, and exact donor-hit rate.
4. Average validation scores across training seeds, then enumerate every source
   subset at k=3 and k=5. Score a subset by the target-balanced maximum included
   specialist validation accuracy. Compare the selected subset's terminal
   specialist-envelope score with the terminal oracle envelope and with source
   strength, node-count, frozen-random, Order-A-prefix, and all-nine references.
5. Report all targets and all three seed-level values. Do not tune thresholds,
   change the metric, drop difficult targets, or substitute ROC-AUC after seeing
   accuracy results.

This is a retrospective prerequisite test because terminal specialist outcomes
already exist. It becomes prospective only if it freezes previously untrained
source sets for Stage B before those models are trained.

### Stage-A gate

Advance using the 300-update pilot only if all conditions hold:

- median all-donor Spearman correlation across the 27 target-seed strata is at
  least 0.70;
- median foreign-only Spearman correlation is at least 0.60;
- mean terminal top-1 regret of the validation-selected donor is no more than
  2.0 accuracy percentage points;
- for both k=3 and k=5, the validation-selected subset is within 1.0 accuracy
  percentage point of the terminal specialist-envelope oracle; and
- none of the four criteria fails in two or more training seeds when recomputed
  seedwise.

Failure stops the pilot-selection branch. A favorable checkpoint other than 300
does not rescue a failed 300-update gate; updates 100 and 900 describe the
cost-quality curve.

## Stage B: prospective subset test, conditional on Stage A

Before training, freeze one k=3 and one k=5 subset using only the update-300
validation matrix. If the maximizing subset has already been trained in
final-core, choose the highest-scoring subset not present among the 31 final-core
source sets; resolve exact ties lexicographically by canonical dataset key. Record
the complete ranked candidate table and its hash.

For each k, train the pilot-selected subset and four controls:

- globally strongest sources under the same update-300 validation matrix;
- largest sources by catalog node count;
- a random previously untrained subset generated with frozen seed 20260912;
- the Order A prefix.

Use three training seeds, the unchanged final-core model/configuration, balanced
source sampling, batch size four, and exactly 2,500 optimizer updates. This is
5 arms x 2 source counts x 3 seeds = 30 planned models, minus only exact physical
duplicates established before launch. Reuse an existing final-core checkpoint
only when its resolved configuration and semantic source set are exact matches;
otherwise train the cell.

Evaluate all frozen models on the same nine-target `static_test` panel only after
the source-set choices, manifest, and checkpoint rule are immutable. The primary
estimand is target-balanced test accuracy of pilot selection minus each control,
formed within training seed. Report the all-nine final-core model as a separate
fixed-compute reference, not as a matched-k arm.

The method succeeds only if pilot selection beats every matched-k control in mean
accuracy at both k values, has positive paired differences in at least two of
three seeds at each k, and does not reduce any target mean by more than 1.0 point
relative to the best control for that target. Otherwise report the observed
coverage tradeoff or Pareto frontier rather than a general source-selection win.

## Stage C: allocation test, conditional on Stage B

Do not optimize source weights until Stage B establishes that pilot-based source
membership is useful. Freeze an exposure rule from validation-only learning curves,
then compare it against uniform sampling within the selected k=5 source set under
the same total episode budget and three training seeds. The allocation objective,
constraints, tie handling, and minimum effect must be recorded in a protocol
amendment before any allocation run begins. Test outcomes from Stages A/B may
motivate that amendment but may not select weights directly.

## Evidence admission and stopping rules

- Validate checkpoint identity, completed update, resolved config, source set,
  split, and episode fingerprints for every cell.
- Missing required cells make a stage incomplete; they are never zero-filled.
- A smoke result is never admitted as scientific evidence.
- Selection uses validation only. Test results never choose a source set,
  checkpoint, allocation, threshold, or metric.
- Keep raw and admitted evidence in the matching analysis leaf under `data/`, and
  retain a machine-readable completeness and provenance receipt.
- The user launches any large Tucker run. Preparation must include a dry run with
  exact cell counts, commands, output roots, and owned-GPU allocation.

The machine-readable constants and gates are in `protocol.yaml`.

## Prepared execution

`build_manifest.py` emits the 729-cell semantic manifest and three 27-checkpoint
job manifests. `run_stage_a_tucker.sh` assigns two persistent workers to each
owned GPU 0--3, runs one validation-only smoke at update 300, then evaluates
updates 100, 300, and 900 sequentially. The production launcher is intentionally
not run automatically; the user launches it after reviewing the dry run.

Local command-only dry run:

```bash
DRY_RUN=1 \
EVAL_LOG_ROOT=/private/tmp/pilot-transfer-selection-dry-run \
EVAL_STATE_ROOT=/private/tmp/pilot-transfer-selection-state \
PYTHON=/opt/homebrew/bin/python3.11 \
bash scripts/experiments/setup/pilot_transfer_selection/run_stage_a_tucker.sh
```

Tucker production command, after the branch is available in a dedicated worktree:

```bash
tmux new-session -d -s pilot-transfer-stage-a \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/pilot_transfer_selection/run_stage_a_tucker.sh'
```

The evaluator reuses final-core's materialized CPU replay, strict checkpoint
loading, source-boundary checks, and raw/observed episode fingerprints. Its new
`--eval-split`, `--checkpoint-step`, and `--protocol` arguments are opt-in; the
original final-core defaults remain test, update 2500, and the original protocol.
`aggregate_stage_a.py` rejects missing/extra cells, test-split results, wrong
checkpoint paths, source mismatches, non-finite metrics, or fingerprint drift.
