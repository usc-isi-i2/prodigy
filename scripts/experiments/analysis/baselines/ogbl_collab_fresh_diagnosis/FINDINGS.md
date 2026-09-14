# Three hypotheses for a sub-million-parameter HyperFusion challenge

Investigation: September 14, 2026. No new training, fusion sweep, or 2019 evaluation.
New evidence comes from archived 2018 predictions, training histories, and replay
of six saved checkpoints on two 2017 negative panels with identical positives and
graph inputs. Prior repeated 2019 exploration remains disclosed. No result here
establishes a leaderboard-accepted win.

## Main conclusion

The strongest new finding is negative-panel overfitting, not a lack of model
capacity. “Fresh negatives” meant one fresh 100,000-pair panel per training year;
the optimizer then repeatedly mined and trained against that same panel. Training
longer learns a separation that largely disappears when only negative identities
change. This is distinct from the previous official-negative leakage problem.

Three plausible mechanisms deserve tests, in this order: continually renew the
negative population; optimize the extreme ranking tail; add a transferable signal
for new collaboration formation. These are falsifiable hypotheses, not three
high-confidence promises of a win.

## What the investigation establishes

### 1. Fixed-year checkpoint replay isolates negative-generalization failure

The comparison holds all 119,622 training positives, the graph through 2016,
node features, and model checkpoint fixed. Only the negative set changes.
Both negative sets contain 100,000 pairs; their overlap is zero and the probe
negatives contain none of those target-year positives.

| Seed | Selected step | Selected: training negatives | Selected: separate negatives | Step 2,000: training negatives | Step 2,000: separate negatives |
|---|---:|---:|---:|---:|---:|
| 0 | 150 | 69.80% | 69.16% | 99.92% | 64.82% |
| 1 | 200 | 88.01% | 74.59% | 99.91% | 68.52% |
| 2 | 250 | 86.35% | 75.05% | 99.91% | 67.42% |

For seed 0 at step 2,000, the 50th-negative logit is −2.89 on training negatives
and +18.74 on separate negatives. This is a within-checkpoint comparison; logits
from different checkpoints should not be treated as a common calibrated scale.
All three models improve on seen negatives while worsening on separate negatives.
A year shift is therefore not necessary for deterioration. The experiment does
not establish that resampling alone will improve future-year performance, nor does
it isolate negative memorization from its interaction with positive overfitting.
The positive examples were deliberately training examples, so these numbers are
not generalization or leaderboard scores. The separate panel had been used in
older research, but was absent from these fresh models' training-negative sets.

### 2. The training dynamics and checkpoint criterion matter

Fresh experts' best standalone 2018 scores occur at updates 150, 100 and 100.
The fusion-based selector instead chose 150, 200 and 250. For seeds 1 and 2,
standalone peaks were 69.0250% and 68.9320%, versus 67.4073% and 67.2126% at
the selected checkpoints. These earlier checkpoints were not refitted or tested
in this investigation. The finding narrows what the failed ensemble ruled out.

By update 2,000, validation scores are 54.6152%, 53.9794%, 58.3899%, while
sampled BCE is about 0.005–0.007. Mining makes sampled loss nonstationary, but the
fixed-year probe independently confirms a severe generalization gap.

### 3. Averaging fails at the tail despite slightly better overall ordering

For seeds 0+1, negative-score Spearman correlation is 0.913. They share 35 of
50 top negatives; all three seeds share 33. Their average improves AUC from
0.977046 for seed 0 to 0.977470, yet Hits@50 falls from 68.7155% to 68.5607%.
It rescues 612 seed-0 misses and loses 705 seed-0 hits. The two models' label-aware
positive union contains 42,078 hits; averaging loses 913 of those and adds only
29 outside it. The union is a non-realizable oracle, not a proposed predictor.

Seed 0's Hits@50 is 68.7155%; Hits@60 is 69.9071% and Hits@75 is 71.2086%.
This shows how much positive mass sits close to the negative boundary. Changing K
or deleting known negatives would be invalid as a method; these are sensitivity
diagnostics only. We must learn a better ordering against unseen negatives.

### 4. Success and failure concentrate in different structural regimes

On the 2018 panel, seed 0 hits 99.9679% of 28,072 repeated collaborations, but
only 41.3095% of 32,012 novel collaborations. Of 17,794 positives missed by all
three seeds, 17,785 are novel; only nine are repeats. There is virtually no room
for a repeat-only improvement to close the gap on this panel.

The model recovers 20.68% of zero-AA positives versus AA-DC's 9.16%, but hits
94.72% of nonzero-AA positives versus AA-DC's 98.86%. It rescues 2,793 AA-DC
misses while losing 1,976 AA-DC hits. Across all seeds, consistent rescues number
2,225 and consistent losses 1,897. Rescues tend to be high-cosine, active pairs;
losses tend to have lower cosine and less recent activity. Promoted negatives
share many rescue features. These profiles are descriptive, not learned rules.

Historical task composition also changes substantially: 2017 training positives
are 74.08% zero-AA and 23.67% mutually recently active; 2018 positives are 35.12%
zero-AA and 59.90% mutually recently active. Only 47,546 of 119,622 2017 positives
have both endpoints already present in the input graph. Treating years as
interchangeable draws is unjustified. Earlier warm-only training also failed, so
simply filtering cold pairs is not an established fix.

## Hypothesis 1 — Renew negatives throughout training

**Claim.** Much of the lost performance is caused by fitting a finite negative
panel; continually refreshed independent negatives will delay or prevent tail
overfit and yield a stronger standalone expert under the same parameter budget.

**Concrete change.** Retain the existing joint architecture, BCE, optimizer,
positive samples, 2,000-update budget, and 1,024 uniform plus 1,024 mined negative
draws per update. Replace the fixed 100,000-pair pool with an independently
regenerated pool every 50 updates; refresh the top-2,048 hard set every 10 updates
as before. Seed every renewal in advance. Do not mine validation or test pairs.
Keep a separate fixed same-year negative probe unused by training or selection
of individual negative identities. Audit overlaps without test-driven resampling.
Count all fitted inference state as before, approximately 496,448 scalars.

**Cheapest discriminating test.** Matched fixed-versus-renewed runs over all three
optimization seeds, first for the existing 2017-to-2018 transition. Select each
standalone checkpoint by standalone validation Hits@50, identically in both arms.
Then verify the gain on an earlier forward-year transition before final refitting.
Track seen/probe tail cutoffs, probe Hits, and future-year Hits through training.

**Prediction.** Renewed negatives produce a smaller seen-versus-probe gap and a
less sharply deteriorating future-year curve. The selection window should broaden.

**Reject as the route forward** if it merely raises training loss, leaves the
probe-tail gap unchanged, or does not improve selected forward-year Hits across
seeds. A flatter but worse curve is not success. This is the strongest mechanism
hypothesis; a >71.29% result is still unproven.

## Hypothesis 2 — Optimize the relevant tail, not average classification

**Claim.** The current objective rewards improvements over easy negatives that do
not help Hits@50. A ranking objective focused on the upper 0.05% of independent
negative scores will improve useful positive margins without sacrificing the
existing strong hits.

**Concrete change.** Keep the joint architecture and use the same negative streams
as the control. Add a partial-ranking loss of the form mean softplus(margin +
s(negative) − s(positive)) against the top 50 negatives of a 100,000-pair training
pool. Use broad positive sampling so existing hits remain represented, with a
predeclared component emphasizing positives just below the training-tail cutoff.
Do not use official validation/test negatives in the loss. Test the tail loss
against BCE with the sampler held fixed; only combine it with renewal after each
factor's effect is understood. No new neural inference parameters are required.

**Evidence.** High overall AUC coexists with poor tail ranking, and averaging
improves AUC while worsening Hits@50. The PLNLP paper's controlled loss comparison
reports 68.72% versus 64.97% on collab with architecture and negative sampling held
fixed and random-walk augmentation disabled. That supports testing ranking loss;
it neither validates this particular partial-tail objective nor predicts our gain.
Source: [PLNLP paper](https://arxiv.org/pdf/2112.02936), Section 5.3 and Table 3.

**Prediction.** On untouched-by-training negative probes, fewer hard negatives
cross the positive boundary and net recovered positives increase. AUC need not
improve. Inspect zero-AA and nonzero-AA strata to detect destructive tradeoffs.

**Reject** if the gain exists only on mined training negatives, if it improves
AUC without Hits@50, or if lost existing hits cancel rescues on forward-year folds.
This is not another hard-negative BCE sweep: the objective and its target
quantile change, while the sampler is controlled.

## Hypothesis 3 — Add temporal path information for novel collaborations

**Claim.** Existing inputs confuse plausible new collaborators with static
lookalikes because the encoder discards edge dates and multiplicities. The age of
a connection path may distinguish collaboration formation from stale proximity.

**Concrete change.** Add exactly two deterministic pair inputs: median formation
age of length-three paths, and a missing-path indicator. For u-a-b-v, formation
year is the latest first-observed year among its three edges; use only events
before the prediction year. This is the exact definition with a surviving lead
in the earlier matched analysis. Standardize from training data only, then feed
the features to the existing decoder. Two new decoder inputs add 512 weights;
allow four extra mean/std scalars, for a conservative total around 496,964.
Do not substitute raw path counts, a learned shallow gate, or test-fitted thresholds.

**Evidence and uncertainty.** In the older strict matched audit, path counts lost
their apparent advantage after matching all 14 existing inputs. Younger path
formation remained directionally associated with positives: 67.3% group-weighted
pairwise preference in the broader cohort, comprising only 33 negative groups with
paths on both sides. This has not been established on fresh-expert errors or across
years. It is the weakest of the three hypotheses and should not yet trigger training.

**Cheapest discriminating test.** Repeat the same all-input-matched comparison on
the fresh experts' errors and earlier historical years, with the definition and
calipers frozen. Preserve unmatched and missing-path cases. Proceed to a two-feature
ablation only if the direction is consistent across years and not driven by a few
negative groups; evaluate the full panel, not just the matched subset.

**Reject** if the association reverses or disappears in earlier years, or if the
feature ablation fails to improve full-panel net hits. There is no basis for a
larger architecture sweep from the present evidence.

## Decision and evidence boundary

Run hypothesis 1 first. Correct standalone checkpoint selection in both arms as
a common control, not a claimed independent treatment. Hypothesis 2 is second;
hypothesis 3 first needs observational replication. Use chronological folds and
report their changing cohorts, seed-level results, and per-stratum recoveries/losses.
Never treat three optimization seeds as independent datasets.

The historical 70.1569% best test seed is not a stable expected baseline. Approximately
525 additional net hits on the 46,329-positive test panel would bridge its gap to
71.29%, but no diagnostic here demonstrates that gain. A final candidate must be
frozen before any further test access. Because 2019 has already influenced prior
research, report that history and resolve acceptance with benchmark maintainers;
a new freeze does not retroactively create an untouched test set.

## Reproducibility

Branch: codex/collab-fresh-ensemble; isolated local clone under work/ensemble-repo.
Tucker worktree: /dataMeR1/phil/gfm/prodigy-fresh-ensemble.
Validation diagnosis revision: 789718f6; fixed-year probe revision: 4207df99.
Runtime roots: fresh_diagnosis_v1 and fresh_sameyear_probe_v1 under
/dataMeR1/phil/gfm/ogbl_collab_compact_joint/.
W&B offline runs: 95gmaouk and 7mpnornr. No training or 2019 file access.
Probe used GPU 0 for short checkpoint replay; measured replay loop 0.875 seconds.

Validation panel/score hashes verified and official OGB metric parity checked.
All six checkpoint identities verified. The separate negative panel was verified
against original provenance and checked for zero overlap with these models'
training negatives and target positives. An independent laptop sort-based audit
exactly reproduced all 12 seen/unseen aggregate metrics from saved scores. Raw
scores remain private; compact JSON receipts are preserved with the analysis.
