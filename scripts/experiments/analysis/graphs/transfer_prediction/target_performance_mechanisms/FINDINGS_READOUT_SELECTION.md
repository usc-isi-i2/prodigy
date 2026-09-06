# Twenty supports can choose a better cross-target prediction rule

6 September 2026. The single declared utility test is complete: nine source
specialists, five targets, two episode streams, PRODIGY and frozen-base SAMGPT.
The decision uses the same twenty labeled supports, no query labels, no extra
labeled accounts and no new training. This is an application of the mechanism
analysis, not a novel cross-validation algorithm or a universal transfer fix.

## Primary result

On the declared covid-political / facebook-page-reference / twibot20 panel,
support-only selection beats every fixed candidate's panel-mean AUC in each
family and each stream. Every source-target cell has equal weight; each panel
has 24 foreign-source cells. All comparisons below use the same support-only
score scaling.

| Model family | Stream | Selected | Best fixed across panel | Gain | Hindsight best fixed per target |
|---|---|---:|---:|---:|---:|
| PRODIGY | Original | .80417 | .77744 | +.02673 | .82561 |
| PRODIGY | Second | .81001 | .77633 | +.03368 | .83098 |
| SAMGPT frozen base | Original | .78035 | .76462 | +.01573 | .79178 |
| SAMGPT frozen base | Second | .78482 | .77428 | +.01054 | .80707 |

The best fixed panel rule is the encoder prototype for PRODIGY and the raw
prototype for SAMGPT. Against the original native prediction scores, selected
mean gains are .09425/.09855 for PRODIGY and .06715/.07056 for SAMGPT; those
larger gains combine changing the rule with support-only rescaling and should
not replace the controlled primary comparison.

The all-five-target summary also exceeds every fixed rule on average:
PRODIGY selected .77848/.78122 versus best fixed .76375/.76440; SAMGPT
.73460/.74251 versus .72800/.73425. This is secondary evidence, not an additional
untouched-target confirmation.

## What succeeds, and what does not

The selector responds to task differences. On facebook-page-reference it picks
raw features in 88.4%/92.1% of PRODIGY episodes and 85.8%/93.6% of SAMGPT episodes.
On election2020-political, PRODIGY picks its encoder prototype in 72.7%/71.0%;
SAMGPT instead picks raw features in 85.8%/86.7%. On twibot20 it picks encoder
prototypes in 46.7%/47.0% and 64.3%/60.7%, respectively.

But it is not a reliable per-target improvement over the best fixed readout.
On twibot20, PRODIGY selection scores .6306/.6120 versus the fixed encoder's
.6765/.6577. It exceeds each fixed comparator in only 0/24 and 1/24 individual
primary source-target cells for PRODIGY, and 1/24 and 0/24 for SAMGPT. The main
utility is adapting across tasks when a single fixed rule would otherwise be
used. The hindsight target-specific rule remains stronger in all four panels.
Do not write that twenty supports recover an oracle or reliably select the best
readout for every target.

## Fixed procedure

For each source model and each episode, ten balanced leave-one-pair-out folds
predict every support once from nine supports per class. Candidate ROC-AUC is
computed over its twenty held-out support margins. Exact ties prefer raw,
then encoder prototype, then learned inference. PRODIGY offers all three;
SAMGPT offers the first two. The winning rule predicts the original queries.

Each candidate's query margin is divided by the RMS of its twenty held-out
support margins, with zero RMS replaced by one. Every fixed comparator uses
the identical procedure. Positive scaling leaves class decisions and within-
episode ranking unchanged but changes pooled cross-episode ranking; this is
not a calibrated-probability claim. Original scores and unscaled selection
remain in the exported table, including cases where scaling reduces AUC.

The selector receives neither query labels nor query representation rows.
Its cached representations inherit each encoder's unlabeled graph context;
SAMGPT uses transductive full-graph representations. This is not inductive
isolation of supports from all query nodes. Selection is over readouts of a
fixed source checkpoint, not over source checkpoints. An untrained encoder was
not among the candidates: the strong existing untrained twibot20 prototype
control prevents any claim that this is the best available inference pipeline.

## Scope and publication decision

Both families use existing single training seeds and different pretraining
objectives, views and budgets. The targets and candidate baselines had already
been examined before this selector was specified. Only the decision rule and
primary comparison were fixed before computing its query outcomes. The two
episode streams are not additional training seeds or independent target domains.

Keep the result as a modest practical consequence of the representation/readout
distinction. No additional selector sweep is needed for the current paper draft.
The completed same-account neighborhood-resampling control is separately reported
in `FINDINGS_FIXED_SUPPORT_CONTEXT.md`: it detects greater hongkong sensitivity
but does not materially repair its political performance.

## Evidence and reproduction

- `data/support_readout_selection/`: complete 1,260 metric cells, 180
  family/source/target/stream cells, 57,600 candidate/episode decisions,
  protocol, audits, derived means and contrasts.
- `analyze_support_readout_selection.py` checks complete counts, unique keys,
  finite metrics and every AUC/tie-break decision, then reproduces all tables.
- `plot_support_readout_selection.py` renders all five targets and both streams.
- Runtime revision `992354e9`; Tucker output
  `/dataMeR1/phil/gfm/prodigy-mechanisms-crossmatch/log/readout_selection_full_20260906`.
  The two-family smoke and full run completed. Existing matched query predictions
  were reused; model-state hashes were unchanged and each PRODIGY source/cache's
  first batch reproduced its full inference output exactly.
- Local branch `codex/target-performance-mechanisms`, worktree
  `/Users/philipp/projects/gfm/prodigy-mechanisms`.
