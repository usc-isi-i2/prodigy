# Paper synthesis: distinguish reference repair from fragile ranking gains

7 September 2026. Advisor synthesis after independent contribution review.
Updated after the completed matched-page prototype test. No additional
inference control nominated. Private worktree `.worktrees/role-topology`,
branch `codex/role-topology-interactions`, code `8ccd3686`.

## Current argument: information and its learned use can prefer different contexts

The strongest new matched evidence is not the late offset diagnosis. On the
same page episodes, fixed prototypes prefer suppressed supports at both
step100 and step2500, across all three seeds and two streams. Native inference
mostly prefers intact supports at step100 but prefers suppression in all six
comparisons at step2500. The encoder/readout distinction is established in
few-shot learning; our specific evidence concerns interpreting the sign of
graph-context utility under a learned inference rule.

| Step | Native intact / suppressed AUC | Prototype intact / suppressed AUC |
|---|---:|---:|
| 100 | .538 / .504 | .570 / .649 |
| 2500 | .692 / .701 | .764 / .782 |

These are within-episode AUC means. Both paths improve in absolute ranking.
The native crossover cannot simply be read as context information becoming
harmful during training: the fixed readout's preference did not reverse.
At step100, suppressed representations carry usable information that native
inference fails to exploit. This does not isolate the class updater from
native query-space processing, and does not prove the cause of that failure.

### Revised evidence order for the next paper figure

1. Matched page trajectory plus fixed prototype: the interpretation problem.
2. Nine-source role map and fixed-query support-to-reference intervention:
   its graph-specific scope and computational localization.
3. Early political stability versus late initialization-sensitive ranking:
   why apparently similar intervention gains must not be conflated.
4. Useful public model benefits from both roles: the boundary, not a hidden
   replication. Offset subtraction belongs in supporting evidence, not the
   headline figure; it did not repair normal inference.

### Consequential test already in progress, not a new experiment request

The independently maintained encoder-solver isolation study trains native,
joint native+ridge, gradient-isolated and ridge-only arms on matched blocked
and interleaved source schedules. At this read-only check its launcher
PID34607 was live, with ridge-only blocked training child PID47736; no final
transfer result was inspected because none was present in the recorded
findings. Do not infer completion from earlier arms' log filenames.

Use its existing gate, without adapting it to the outcomes: isolation must
improve over native and joint across the fixed panel, and its full solver
must add value over its own ridge readout beyond a single target. Improved
encoder features with an unnecessary solver would not support a new graph-ICL
method claim. This study is not a direct causal test of Hong Kong's historical
single-source mechanism, since its sources and schedule differ. Keep the two
claims distinct even if the training result is favorable.

Until that completed evidence is available, do not add more inference controls
or portray the current diagnostic account as an established high-impact method.
The original objective remains open; the inference-localization branch has
yielded its current conclusion and should not grow by default.

### Partial training audit, before target outcomes

Read-only audit on 7 September of the four completed blocked-schedule arms in
Tucker `prodigy-encoder-solver-isolation/log/isolation_full_20260907`, recorded
training revision `0b9d96414e8cbf9c55c838178cc0bace5c5f0eb1`: all four saved
initial model states are tensor-exact, all terminal training-state records
report 2500 completed updates and the intended objective, and all consumption
audits contain ordered steps 1–2500. Per-source payload digests match across
native, joint, isolated and ridge-only, with 625 consumed episodes per source.
This is a partial audit, not the full eight-arm verifier or a transfer result.
The interleaved training remains in progress under launcher PID34607.

The evaluation-plan code was inspected locally: baseline replay includes
full-model predictions and support-fitted ridge/prototype readouts at saved
stages, plus raw-feature readouts, using the fixed five-target/two-stream panel.
The training launcher does not automatically execute that evaluation plan.
Cross-thread coordination was denied by the app privacy check; no private
findings were transmitted, and permission remains pending. Do not duplicate
or mutate the other thread's running experiment to work around this boundary.

### Complete training audit (supersedes partial status above)

All eight terminal checkpoints and `DONE.json` (arms=8, steps=2500,
smoke=false) are now present. Launcher PID34607 exited and its tmux session
ended. The existing read-only verifier passed exact initialization equality
across all eight models, intended objective/completed-step metadata, complete
ordered consumption records, and matching per-source payload digests across
both schedules and all objectives. No downstream evaluation process or eval
directory was found in the experiment log root at this check.

Encoder trajectory equality did NOT hold: isolated versus ridge-only maximum
tensor errors were 11.673935 (blocked) and 17.576612 (interleaved), both in BN
running variance. This is not only a BN-buffer difference: `lin_x.weight`
relative L2 differences were 1.0434 and .9701; its maximum entry differences
were .5793 and .5518. Both BN batch counters equal 2500. The earlier first-step
CUDA null comparison does not establish that this full-trajectory divergence
is entirely numerical. Do not claim identical learned encoders or attribute
the divergence to a proven cause. The deployment comparison remains useful,
but any causal interpretation must retain this qualification. No retraining,
new control, or target-outcome selection was launched during this audit.

Downstream status update: the other worker subsequently launched the planned
evaluation at `log/isolation_eval_20260907`, with live sessions
`isolation-eval-original` and `isolation-eval-fresh` (first replay PIDs67913
and68212). Commands use the fixed eight-model inventory, baseline plus stage
readouts, 32 batches and episode offsets0/100003. This thread did not launch
duplicates, modify the checkout, or transmit the denied coordination payload.
The status above saying no evaluation was running is historical, not current.

### Completed panel: isolation fails the AUC gate

Both evaluation sessions ended; all ten target/stream files contain136 rows
(eight models ×17 decoders). Fixed-four-target AUC means, averaged over both
streams, are:

| Schedule | Native full | Joint full | Isolated full | Isolated U1 ridge |
|---|---:|---:|---:|---:|
| Blocked | .830809 | .810206 | .800192 | .806125 |
| Interleaved | .820070 | .828078 | .800737 | .812780 |

The nominated isolation method fails the required AUC comparison against
native, joint and its own ridge readout for both schedules. This is sufficient
for no-go under the conjunctive prespecified gate; do not launch the three-seed
expansion or market isolation as an established repair. It does not disprove
all possible coadaptation explanations or establish one for the historical HK
effect. Native/joint U1 readouts are also stronger in these aggregate AUCs,
so these results do not support a simple claim that native gradients uniformly
damage encoder transfer.

Verification qualification: the full existing analysis failed its saved-metric
parity check. A read-only scan of all240 nominated decoder cells found decision
metric mismatches in some ridge readouts; all AUC/NLL differences remained below
the existing1e-6 threshold and no full-model mismatch was reported. Code
inspection identifies a tie-convention difference: the production global path
maps probabilities to semantic classes before argmax, whereas the audit maps
the result of local argmax. Those need not agree for ties with reversed class
order. CPU/GPU softmax rounding can additionally affect near-ties. These are
candidate causes requiring exact reproduction, not a tolerance waiver. The
macro-F1 outcome audit remains incomplete. Preserve this failure and the
original outputs; no training or evaluation rerun is warranted merely to fix
the offline decision computation.

### Decision discrepancy resolved; distinguish training utility from inference utility

A read-only production-rule reconstruction over all240 nominated decoder
cells maps probabilities into semantic class order before argmax (except FB,
which retains episode-local class order). All reconstructed accuracy, binary
F1 and AUC values match saved metrics within the unchanged1e-6 threshold.
There are7793 probability-tie occurrences across these repeated model/decoder
cells; this is not7793 independent accounts. The old local-argmax mapping
explains the decision mismatch. No model rerun or threshold relaxation was
needed. The existing analysis script itself has not been modified here;
the previously failed full audit remains recorded as such.

Fixed-four-target macro-F1 means across both streams:

| Schedule | Native full | Joint full | Isolated full | Isolated U1 ridge |
|---|---:|---:|---:|---:|
| Blocked | .780148 | .756619 | .767480 | .760754 |
| Interleaved | .762907 | .775848 | .754956 | .769456 |

Isolation loses macro-F1 to native under both schedules. Its blocked full
solver modestly beats its own ridge on decisions but not AUC; that mixed
pattern does not satisfy the gate. Preserve it rather than claim every
readout metric favors ridge.

The more useful positive comparison concerns training versus inference:

| Schedule | Native-trained U1 ridge AUC | Joint-trained U1 ridge AUC | Ridge-only U1 ridge AUC |
|---|---:|---:|---:|
| Blocked | .846775 | .848371 | .809311 |
| Interleaved | .841909 | .854561 | .811193 |

The same fixed readout favors native/joint-trained encoders over ridge-only
under both schedules. Yet native/joint models' U1 ridge readouts outperform
their full inference on aggregate AUC. Thus a solver can be useful to the
training system without being the best inference rule. This is a descriptive
matched result, not proof that a specific native-gradient property causes
the difference: one seed, optimizer trajectories differ, and the historical
single-source context intervention is a separate setting. Do not relabel the
failed isolation method as successful. The next synthesis should distinguish
these two utilities, not revive more inference controls or broaden the
manuscript before a coherent contribution is established.

### Evidence recovery: recent notes are not the full research record

A return to training-task construction found that sampler coverage and role
assignment were already tested in the24-model, three-seed factorial documented
in `FINDINGS_REPLAY.md`. The predicted stable Hong Kong benefit failed despite
the intended exposure changes. Do not reopen this as a new untested explanation.

Two earlier completed studies are relevant to contribution assessment and were
not included in the recent independent review's requested reading:

1. Initial-readout-frozen training:18 models, three sources, three seeds, exact
   paired training inputs/initialization and frozen tensors. The Facebook probe
   improves in every matched pair on both streams, but the full-model primary
   fails. `data/readout_training_validation.json` records3060 rows and confirms
   those training gates. This is not the same intervention or probe as U1
   gradient isolation; do not pool them as replications of one mechanism.
2. Corrected production sampler: nine source runs, four saved steps,6120 rows.
   The recorded TwiBot trajectory has increasing pooled ridge AUC and decreasing
   full-model AUC across all nine sources and both streams. The current
   `data/corrected_sampler_validation.json` confirms the inventory and matching
   evaluation inputs, while explicitly denying matched old/new training inputs
   or untouched targets. It supplies breadth for the trajectory observation,
   not nine independent mechanisms or a successful role-suppression remedy.

An independent reassessment has been requested using those sections rather
than only the recent page/synthesis notes. This is recovery of existing
evidence, not newly completed experiments. No new model run or manuscript
expansion was launched.

### Direct corrected-trajectory recheck and stage qualification

Read-only recomputation from Tucker
`prodigy-mechanisms-corrected/log/target_mechanisms/corrected_sampler_replay_20260906/{original,fresh}/twibot20/metrics.jsonl`
confirms every one of the nine source-specific100→2500 pooled-ridge AUC changes
is positive and every full-model change negative, on both streams. The recorded
DONE inventory is36 checkpoints / nine runs /6120 rows. This is an endpoint
contrast, not a claim of monotonic change at every intermediate checkpoint.

Crucially, "pooled" means `S0_pool/ridge`, observed at the input of
`reset_mlp_m`, not `U1_pre_meta/ridge`. U1 ridge itself declines for7/9 sources
on original episodes and8/9 on fresh episodes. Do not say the entire encoder
improves, or locate this broad trajectory solely in the metagraph solver.
The pooled and U1 probes are different feature spaces; their score differences
are not an additive causal decomposition. In particular U1 can still exceed
the pooled probe in absolute AUC while declining over training.

The independent reviewer retracts the broad "empirical branch exhausted"
judgment after reading the omitted studies. Revised contribution candidate:
continued training can improve pooled transferable information while degrading
end-to-end use; the discrepancy survives tested training interventions. This
supports a phenomenon paper, not a proven graph-specific predictive mechanism.
The advisor accepts the revised evidence ordering, with the exact stage labels
above: broad trajectory first, training intervention second, role-localization
and page crossover as explanations of what the trajectory does and does not
imply. No new experiment is nominated by this reassessment alone.

### Across-target boundary and within-episode confirmation

The same100→2500 endpoint comparison across all five corrected-sampler
targets gives full-model AUC improvements for all nine sources on political,
election and pages in both streams, versus decreases for all nine on TwiBot20.
Suspension is mixed. Thus the phenomenon is target-dependent, not universal
training degradation, and nine source replications do not equal nine targets.

A new read-only calculation from saved TwiBot prediction tensors and all
hash-checked cached batches confirms the direction within episodes: for each
of nine sources in both streams, mean episode AUC improves at S0_pool/ridge
and declines for the full model. Each cell uses128 episodes; local binary
class1 scores preserve the correct within-episode orientation. Prediction
batch order and cached input hashes were checked. No model forwards ran.
This rules out a purely cross-episode scoring explanation for the sign pattern.
It does not establish significance for each individual source; the political
source's original-stream pooled-probe increase is only0.0434 AUC points.

Unlike globally pooled AUC, within-episode U1 ridge decreases in8/9 original
and7/9 fresh cells; preserve this metric-specific distinction. The global
U1 counts above must not be reused as within-episode counts. The specific
source exceptions are Pages in original, and Pages/Hong Kong in fresh.

### Frozen-training stage identity corrected in the replacement manuscript

Inspection of `readout_training_constraint.py` and the authoritative
`READOUT_KEYS` confirms only four parameters are frozen: weights/biases of
`layer_list.0.reset_mlp_c` and `reset_mlp_m`. The metagraph solver and background
convolution remain trainable. Recomputed original/fresh Facebook JSON metrics
confirm the consistently improving probe is `U1_pre_meta/ridge`, not an
unspecified separate readout. This is the same stage name used elsewhere,
though not a matched cross-study experiment. Earlier wording that suggested
the probe was a different stage from U1 should not be retained.

U1 improves in all18 source-seed-stream contrasts; S0_pool ridge worsens in17
of18 (only COVID seed2 original has a tiny positive .0191-point change).
The U1 source means reproduce the old findings: COVID+4.194/+6.483,
Hong Kong+9.317/+9.039, Ukraine+3.845/+3.535 AUC points. This sharpens the
stage-specific nature of the intervention without changing its unsuccessful
full-model primary. The replacement main text now specifies the frozen tensors
and both intermediate trends; the old compiled manuscript is still preserved.

## Preserved earlier synthesis

## One conclusion readers should retain

**An improvement from a support-context intervention need not mean that class
reference construction has been repaired. Similar AUC gains can arise in
different reference-geometry regimes, with different decision behavior and
dependence on label initialization.**

This is not a claim that AUC and accuracy differ for the first time, or that
attention/value decomposition is new. The contribution candidate is an
intervention-based way to distinguish two interpretations of graph-ICL
transfer, supported by a predicted contrast in previously unmeasured outcomes.

## Evidence hierarchy for the main figure

| Regime / setting | What the data establish | What they do not establish |
|---|---|---|
| Original 2.5k HK-to-political | Suppression improves within-episode AUC by 7.25 points under either initialization and accuracy by 7.7-7.9 points. The tested mismatch does not explain this benefit. | A deployable selector, all-source benefit, or superiority to every strong readout. |
| Nominated 50k HK-to-political | Suppression gains 8.76/10.35 AUC points under current initialization but only .34/1.87 with the training table; accuracy declines in both cases. References are much less separated after suppression. | The same mechanism robustly replicated by longer training, or complete elimination of the primary effect by initialization. |
| New-outcome prediction | The less-collapsed early configuration has lower initialization-induced decision disagreement and a smaller AUC interaction, as predicted before those outcomes. | Independent domain generality, a causal intervention on collapse, or a universal threshold. |
| Original-style public Wiki-to-FB15K-237 | A useful checkpoint needs context in both roles; suppression sharply harms accuracy. | Replication of harmful support processing or validation of the collapse explanation. |

The nine-source map is the breadth/motivation panel. Fixed-query and
fixed-attention K/V results establish a computational path. They should support
this narrative, not replace it with an inventory of controls.

## Proposed abstract core (not yet inserted into the manuscript)

Graph in-context learners use support neighborhoods to construct the class
references against which queries are scored. We show that superficially similar
improvements from suppressing support context can have different meanings.
In a controlled political-classification setting, suppression improves ranking
and accuracy, and the improvement survives restoration of the pretraining
label initialization. A nominated longer-trained configuration also improves
ranking, but its references nearly coincide and its predictions collapse toward
one class. A fixed initialization-by-context intervention substantially reduces
this ranking gain and changes its decision behavior. The less-collapsed
configuration correctly predicts lower sensitivity to initialization in new
intervention outcomes. Fixed-query, fixed-attention value interventions localize
the support-to-reference path, while a public-benchmark counterexample shows
that removing context is not a general remedy. Together these results motivate
evaluating class-reference separation, initialization dependence and decisions
jointly when interpreting graph-ICL transfer. The evidence distinguishes a
useful support intervention from a fragile ranking gain; it does not yet identify
why intact graph context impairs transfer in the original setting.

### Encoder-localization update

The full retained-episode verification in `FINDINGS_ZERO_MESSAGE_OFFSET.md`
now identifies how the late suppressed encoder enters the collapsed regime.
With messages zeroed, its pre-BN computation is `W_self x + b_self + MLP(0)`.
The large learned MLP(0) term overwhelms an otherwise diverse feature-dependent
signal. Adding the shared offset reduces late pre-BN directional dispersion
about 235-238-fold; BN is not where the collapse first appears. This was
verified on both late streams, with exact reproduction of saved logits and
numerical tensor-level reconstruction of the affine computation.

This strengthens the explanation of the late intervention's side effect,
not the claim that intact context harms supports for this reason. Do not
replace the paper's original scientific question with diagnosing an artifact
of suppression. The early robust improvement remains unexplained by this
large-offset account. A correction of suppressed embeddings, even if useful,
would not alone establish a mechanism of native transfer failure.

The paired subtraction test is now complete (`FINDINGS_OFFSET_INTERVENTION.md`).
It restores suppressed pre-BN dispersion, but post-ReLU/U1 remain nearly
collapsed. Subtraction worsens intact ranking and accuracy on both streams.
Therefore MLP(0) is not a demonstrated transfer-repair target or a sufficient
explanation of final collapse. Close this intervention branch; retain the
finding as a warning that an ablation's improved ranking can coexist with an
out-of-operating-range support representation. The learned-operating-range
description is a scoped interpretation, not a proven explanation of the
original early benefit.

## Current manuscript audit and replacement decisions

Read the current `main.tex`, `generated/abstract.tex`, and relevant body claims
in `paper/transfer-prediction/mechanism-draft-2026-09-06` (outside this repo).
It currently leads with "Localizing and Repairing Support-Side Failures" and
the three-seed role-intervention/control inventory. It does not contain the
new 50k/public/initialization results. Its early evidence is not invalidated.

- Replace the headline emphasis on replicated repair with the distinction
  between successful early repair and fragile late ranking gain.
- Retain the early three-seed effects with their original scope. Do not extend
  the new one-checkpoint initialization result to all three seeds.
- Replace "message presence versus content" as an explanation with the
  actual scale/value evidence; the presence controls alone do not identify
  a discontinuity or normalization artifact.
- Describe the public test as a counterexample, not hide it among caveats.
- Do not claim label initialization explains the original effect. The new
  early factorial contradicts that account.
- Do not import the separate-stream ridge comparison into the new factorial.
- Keep failed support calibration subordinate: it closes a practical-method
  claim, not a second central contribution.

Main manuscript files remain unchanged, respecting the instruction to make
the one-page argument convincing before expanding the manuscript. Earlier
one-page PDFs are historical snapshots; the next consolidated figure should
show both initialization regimes together, not only the 50k moderation.

## Decision gate, not another control list

The independent reviewer recommends synthesis now. The 8+ ambition is not yet
established: two previously observed configurations of one source-target pair
do not identify why learned construction produces collapse. If this synthesis
is still too narrow, the next substantive question is **what training-induced
property drives entry into the collapsed-reference regime?** Do not substitute
another initialization setting, geometric partition, calibration threshold or
checkpoint search for that question.
