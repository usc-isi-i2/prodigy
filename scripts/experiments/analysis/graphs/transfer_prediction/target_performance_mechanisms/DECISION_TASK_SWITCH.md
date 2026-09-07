# Next decisive question: task-selective transfer at fixed graph inputs

7 September 2026. Private advisor/researcher decision, not a manuscript claim.
Local worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, current code `1f261789`.

## Why this test, after the failed orientation bridge

The completed Hong Kong orientation-risk test fails its two-stream discovery
gate. No more sources, bins, or predictors will be searched under that plan.
The positive evidence left is a cue-conditioned ranking tradeoff: training
loses degree-cue-correct rankings while gaining many cue-wrong rankings, and
pooled-feature readouts improve. This does not establish useful replacement
of one task capability by another; attenuation toward chance also explains it.

The next question is therefore whether the *same graph examples* exhibit
opposite training trajectories when the supports communicate different rules.
This varies the target label function at fixed paired inputs. It is not
another intervention on graph construction or a claim that all distribution
shift is in p(y|x).

## Bounded design selected before model outcomes

- Discovery source: corrected Hong Kong, saved steps100 and2500, both existing
  original/fresh TwiBot20 input streams. No new training or source sweep.
- Structural rule: whether the sampled center receives any real-node
  background edge (incoming degree >0). Call this a receive-edge rule, not
  arbitrary structural reasoning or the original bot ground truth.
- Content-feature rule: above/below a fixed leading principal-component
  projection of normalized center biography embeddings. Fit the component and
  threshold on the unlabeled original cached bank only, before model outcomes;
  reuse them on fresh. Orient the component by making its largest absolute
  loading positive; threshold at the original-bank median, exact ties low.
  Reject a degenerate leading eigenspace rather than choosing another axis.
  This is a content-feature rule, not validated semantic classification.
- Deduplicate center IDs deterministically by first cached occurrence within
  each stream before fitting/selecting. Preserve that occurrence's exact
  neighborhood. Record its original cache hash and subgraph index. No graph
  resampling or features constructed using original target labels.
- For each of128 episodes, sample5 supports and6 queries from each of the four
  joint rule cells. Thus both tasks have10 supports and12 queries per class,
  with exactly the same44 points, role masks and graph tensors. Centers must
  be unique within an episode. Repeat across episodes is allowed and disclosed.
  If any cell lacks11 distinct centers, stop for feasibility; do not search
  alternative rules after observing model outputs.
- Run both rule assignments and their class-ID complements. Keep the same
  label-token vectors in every condition, same point order and metagraph
  connectivity. Change only support-label edge attributes and scoring labels.
  Verify query-label changes alone cannot alter logits. Counterbalance labels
  in analysis, not by selecting a favorable polarity.
- Compare native inference, the existing S0-pool ridge, and support-fitted
  scalar-rule baselines. Keep frozen eval-mode normalization. Report absolute
  accuracy and mean episode AUC, each averaged over both label polarities.
  No fitted query labels, neural adaptation or intervention selection.

The directional hypothesis requires native receive-edge-task performance to
decline AND native content-feature-task performance to improve in both streams,
with useful late pooled-feature structural decodability. A content task near
chance or merely deteriorating less is not evidence of replacement. Exact
material-effect/decodability criteria will be written before neural execution,
after input-only feasibility and baseline construction; no model outcomes
will be used to choose them. No performance outcome has been inspected for
these task assignments.

### Frozen practical discovery criteria before neural execution

Independent advisor review selects mean episode AUC, averaged across the two
class-ID polarities, as primary. Require ALL of the following separately in
both streams: native receive-edge change <=−.03; native content-feature change
>=+.03; early native receive-edge AUC >=.60; late native content-feature AUC
>=.60; late S0-pool ridge receive-edge AUC >=.70 and its change >=−.02.
Report accuracy and individual polarities without an additional post-outcome
gate. These are practical effect-size criteria, not significance thresholds.
Failure does not authorize changing PCA axes, thresholds, cells or sources.

The rule-definition bank is transductive and unlabeled. Joint-cell balancing
defines a diagnostic distribution, not the original bot benchmark. Fresh may
share account IDs with original; neither pairs nor streams are independent
account replications. Structural labels describe the retained sampled
occurrence, not an invariant account property.

Input-only execution amendments, before any task/model outcomes: the first
audit stopped because original episodes permute the same two token vectors;
the diagnostic fixes the first recorded table and accepts only exact order
permutations in the old bank. The second audit reached rule fitting and
stopped because some center feature vectors are exactly zero. Normalized
content direction is undefined there. Exclude these centers from BOTH paired
tasks, record their IDs privately and report their counts; do not assign an
invented direction or silently turn missingness into the content rule.
Nonfinite features remain fatal. These input-eligibility amendments do not
change the rule, thresholds, model gate or chosen source. Failed logs remain.

## Feasibility evidence, not a task result

Read-only inspection on Tucker of original batch000 confirms176 extractable
cached subgraphs (four44-point episodes),768-dimensional center features,
global center IDs, and identical two-label embedding tables across its four
episodes.87/176 sampled centers have zero incoming degree;89 have positive
degree. `to_data_list()` retains each subgraph's graph tensors, global IDs,
center identity and pooling edges. Thus existing inputs can be reused rather
than rebuilding the full graph. Joint content/structure cell coverage remains
unchecked; one batch's marginal balance does not prove full feasibility.
The first inspection used the wrong PyG keys accessor and terminated after
printing shapes; the corrected read-only inspection completed. No remote
files, models, checkpoints, sessions or GPUs were changed.

### Completed full-bank input feasibility

The input-only checker subsequently completed at code `e33f4d39` in Tucker
`/dataMeR1/phil/gfm/prodigy-task-switch`, output
`log/task_switch_bank_20260907`. Its tmux session and PID229688 are terminal.
All64 input batch hashes were verified; no weights or prediction files were
opened. Two PCA helper tests pass. Prior failed logs `bank.log` and
`bank_v2.log` remain; completed log is `bank_v3.log`.

| Bank | Eligible unique centers | Four joint-cell counts (00,01,10,11) | Zero-feature exclusions |
|---|---:|---|---:|
| Original |1860|366,484,564,446|278|
| Fresh |1866|345,488,586,447|284|

Both banks exceed the required11 distinct centers in every cell. The frozen
original PCA top eigenvalues are .01986582 and .01800730, so its leading
direction passes the nondegeneracy requirement. Neither bank has a score
exactly tied at the frozen threshold. Thus the nominated task pair is feasible
without changing either rule or resampling neighborhoods. The eligibility
restriction excludes roughly13% of unique cached accounts; conclusions will
explicitly concern accounts with nonzero biography embeddings.

Aggregate local receipt: `data/task_switch_feasibility_20260907.json`.
Account identities, excluded IDs, exact occurrence references and frozen PCA
parameters remain in private Tucker artifacts. Only code/tests were pushed.
Next implementation is the matched episode constructor and label-only replay,
with the frozen gate above; no neural task-switch outcome exists yet.

## Contribution boundary and subsequent causal decision

If the paired rule switch succeeds, the contribution candidate is that graph
ICL transfer has a task-conditioned training tradeoff: the graph and input
features alone do not determine whether further pretraining helps. The
support-reference localization stays separate until experimentally connected.
This diagnostic by itself does not show that NM caused a particular invariance,
nor provide deployment utility or establish an8+ contribution.

Code inspection confirms NM groups random-walk endpoints by their shared
anchor (`data/dataloader.py`, `_sample_center_members`) and its sampler uses
symmetrized adjacency. This does NOT prove actual classes mix opposite directed
roles or that the objective rewards erasing receive-edge information. Only if
the task-switch test succeeds should actual training-task collisions be audited
to nominate a single matched task-construction intervention. No role-aware
retraining is authorized by code inspection alone.

Prior work already studies feature preferences under competing demonstrations:
[Si et al., ACL2023](https://arxiv.org/abs/2305.13299) and task recognition versus
new input-label mapping in [Pan et al., Findings ACL2023](https://arxiv.org/abs/2305.09731).
Do not claim invention of task-prior diagnostics. The intended new evidence is
the fixed-graph training crossover and its computational cause, if established.

## Completed discovery: both tasks improve, rejecting structural-capability loss

All48 cells and four checkpoint/stream receipts completed at code `7485bf84`,
Tucker `prodigy-task-switch/log/task_switch_discovery_20260907`. The session
and PID231385 are terminal. There is no new training. Native old-batch logits
match exactly for each model/stream; weights and normalization buffers remain
unchanged. All384 cross-condition final-query comparisons are bit-exact.
Sixteen direct query-label leakage checks (first batch of each model/stream,
four task/polarity conditions) are exact; do not claim all-batch leakage checks.
Independent code review found no fatal paired-design or metrics issue.
Seven constructor/PCA tests pass, including native sequence parity.

Mean episode AUC, arithmetic average across both label polarities:

| Stream | Native receive-edge,100→2500 | Native content-feature,100→2500 | Pool-ridge receive-edge,100→2500 |
|---|---|---|---|
| Original |.65944→.90934|.50109→.66970|.85786→.90896|
| Fresh |.63502→.91678|.48985→.65457|.85764→.92361|

Every individual polarity improves for both native tasks in both streams.
Native receive-edge accuracy improves .66488→.80729 and .65837→.81169;
content accuracy improves .49593→.61735 and .49251→.60482. Scalar-rule AUC
is1.0 in both tasks/streams; receive-edge accuracy is1.0 and content accuracy
.96322/.97201. These known-rule baselines show the labels are learnable from
the nominated scalar; they are not fair model competitors with identical
information access.

The frozen full gate **fails** because structural performance improves by
24.99/28.18 points instead of declining by at least3. All other criteria pass.
This is not borderline or an insufficient-power excuse. Reject the nominated
tradeoff between these two capabilities, and do not start the conditional
role-aware NM retraining plan or search alternative PCA rules to rescue it.

The useful positive conclusion is stronger than 'the encoder retains degree':
the late native in-context model itself successfully uses a receive-edge
distinction when supports specify that rule. Thus loss of the original
bot-ranking advantage does not establish a general inability to use this
structural cue. Receive-edge presence is coarser than degree magnitude/ranking;
the synthetic balanced bank is not the original bot episode distribution.
This result constrains the explanation but does not identify support ambiguity,
noisy cue-label correspondence, or NM-induced invariance as its cause.

Private evidence: `data/task_switch_discovery_20260907.json`,
`data/task_switch_receipts_20260907.json`, and
`data/task_switch_summary_20260907.json`. The figure
`figures/task_switch_argument.png` shows both complete endpoint trajectories.
Individual predictions and selections remain private on Tucker. No manuscript
expansion or new experimental block follows automatically from this result.

## Original support correspondence: input-only audit

At code `b0a5e728`, all256 original bot episodes were audited without model
forwards. For each episode choose the polarity of the binary receive-edge cue
using support agreement only; ties produce a constant query score. Count
exceptions to that polarity. These are exceptions to a simple cue, not bad or
corrupted bot labels. AUC uses only the support-chosen polarity, never a query
oracle. Three synthetic tests verify reversal, ties and query-label blindness.

| Stream | Mean support exceptions | Median | Interquartile range | Episodes with >=20% exceptions | Cue query AUC |
|---|---:|---:|---|---:|---:|
| Original |32.70%|35%|25–40%|114/128|.67025|
| Fresh |33.09%|35%|25–40%|117/128|.65592|

Only1 original episode and0 fresh episodes have perfect correspondence;
6 episodes per stream tie at50%. The clean synthetic task is therefore unlike
almost every original support set in this respect. This establishes relevance
of imperfect correspondence, not that training becomes less tolerant of it.
The original audit includes zero-feature accounts, whereas the synthetic bank
excludes them; do not silently equate the populations.

Tucker output: `prodigy-task-switch/log/support_cue_audit_20260907`; completed
session and PID235792 verified terminal. Local receipt:
`data/support_cue_audit_20260907.json`. All64 cached batch hashes checked.

Advisor's conditional next falsifier is a balanced support-exception test on
the already fixed clean-task graphs, with clean query labels and no training.
Nested masks must swap1 or2 supports in EACH of the four joint cells (20/40%),
so corruption does not create a content-feature class association. If pursued,
40% is the primary nominated level because it is closest to the observed35%
median;20% is descriptive. Require late native AUC at40% to be at least3points
below early in BOTH streams. A larger drop from the stronger late clean
baseline is not enough. No model outcomes at these exception levels exist.

This is a bounded test of learned tolerance to synthetic support exceptions,
not a new paper identity.40% approaches uninformative supports and random
exceptions differ from systematic real cue-label relationships. Passing would
not establish mediation of the original bot decline; failing would close
this intolerance explanation without noise-level or source search.

## Completed support-exception test: late capability survives

All72 cells/four receipts completed at `9db58f85`, Tucker
`prodigy-task-switch/log/support_exceptions_20260907`. Session and PID237504
are terminal. Clean inputs and all three decoder logits reproduce exactly
across every batch, both polarities and all checkpoint/stream combinations.
Weights/buffers are unchanged,640 final-query comparisons are exact, and24
first-batch query-label-blindness forward checks pass. Two mask tests and five
constructor tests pass; independent code audit found no fatal issue.

Native mean episode AUC (polarity averaged):

| Stream | Clean,100→2500 |20% exceptions,100→2500|40% exceptions,100→2500|
|---|---|---|---|
|Original|.65944→.90934|.69340→.87977|.60699→.67866|
|Fresh|.63502→.91678|.66455→.86814|.59448→.69010|

The primary40% training changes are **+7.1669/+9.5622 AUC points**, not the
nominated <=−3point changes. Accuracy also improves .55632→.63818 and
.54427→.64144. Thus the synthetic support-intolerance explanation **fails**.
The later model loses more AUC relative to its stronger clean baseline, but
remains materially better in absolute performance: the prespecified comparison
prevents mistaking greater clean-to-exception loss for negative transfer.

Pool-ridge AUC at40% improves .64312→.67980/.66113→.71294. The scalar-rule
baseline is perfect at all levels because balanced exceptions preserve the
positive correspondence in every episode and it receives the exact defining
scalar. This is intentional and means the test is not generic unconstrained
label-noise robustness. The early20% AUC exceeds clean by3.40/2.95points;
report the nonmonotonicity, not a universally monotone degradation claim.
It does not rescue the failed late-versus-early criterion.

**Stop this branch:** no additional noise levels, new random masks, source
sweep or training intervention is nominated. Together the tests demonstrate
increasing native capability on this receive-edge task with both clean and
balanced imperfect supports. They contradict general structural-capability
loss and the nominated loss of tolerance. Systematic cue-label structure and
degree magnitude remain outside scope, not established explanations.

Local private evidence: `data/support_exceptions_20260907.json`,
`data/support_exceptions_receipts_20260907.json`,
`data/support_exceptions_summary_20260907.json`;
figure `figures/support_exceptions_argument.png` rendered and visually checked.
The frozen-gate aggregator passes synthetic pass and incomplete-grid checks.
Only code/tests were pushed on `codex/role-topology-interactions` from local
`.worktrees/role-topology`; results, figures and prose remain private.

## Population accounting: missing query biographies do not contain the decline

Saved original Hong Kong bot predictions, no forwards, code `6c88c0bc`:
partition all positive/negative query pairs into both-feature-nonzero versus
at-least-one-zero. Preserve original support sets and equal-episode pair
weights; group conditional values divide their contributions by group mass.

| Stream | Both-nonzero pair mass | Native conditional AUC,100→2500 | Pool-ridge conditional AUC,100→2500 | Both-nonzero contribution to native change |
|---|---:|---|---|---:|
|Original|76.26%|.65709→.60928|.62073→.67573|−3.6458points|
|Fresh|74.50%|.64255→.60851|.58375→.66400|−2.5363points|

Pairs with a zero-feature endpoint contribute another−1.2424/−.3282points;
both groups sum to the original full-model decline−4.8882/−2.8646points.
Thus decline persists among nonzero-feature queries and is not confined to
the excluded query population. This is not a matched rerun of the synthetic
task: original supports still include zero features (mean13.71%/12.93%), and
only3/9 original/fresh episodes contain no zero-feature supports. Do not claim
the support-population difference has been ruled out or that the original
and synthetic populations are now identical.

Input/prediction batch hashes verified throughout; pair partitions reconcile
exactly per episode. A synthetic complete-ranking reversal check passes.
Tucker `prodigy-task-switch/log/zero_bio_accounting_20260907/results.json`;
local `data/zero_bio_accounting_20260907.json`. Process completed successfully.

**Advisor synthesis after closing this branch:** return the paper center to
class-reference construction, its fixed-query causal localization, and its
heterogeneous transfer utility. Synthetic task success is an important
boundary against capability-erasure stories, not a new explanation of natural
bot failures. The missing acceptance-critical result remains a condition on
natural support/reference computation predicting context-effect signs outside
discovery. No new explanation is established merely because it is untested.
