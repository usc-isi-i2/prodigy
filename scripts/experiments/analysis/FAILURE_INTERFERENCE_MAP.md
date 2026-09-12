# Failure and interference map

**Snapshot: 12 September 2026.** This is a question → experiment → finding map for
understanding when an improvement in one place degrades performance somewhere else.
It emphasizes the investigations from 6–11 September, then connects them to the older
experiments needed to interpret them. It is an evidence index, not a new meta-analysis.

The broad program inventory remains
[EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md). The canonical directory map remains
[README.md](README.md). This document is narrower: it organizes evidence about errors,
negative transfer, interference, and tradeoffs.

## Table of contents

1. [Current answer](#1-current-answer)
2. [The analysis lenses](#2-the-analysis-lenses)
3. [Example-level failures](#3-example-level-failures)
4. [Support and episode interference](#4-support-and-episode-interference)
5. [Representation and readout failure](#5-representation-and-readout-failure)
6. [Graph-to-graph interference](#6-graph-to-graph-interference)
7. [Task, objective, and architecture tradeoffs](#7-task-objective-and-architecture-tradeoffs)
8. [Data and evaluation failures that can masquerade as model failures](#8-data-and-evaluation-failures-that-can-masquerade-as-model-failures)
9. [Open questions and the cleanest next experiments](#9-open-questions-and-the-cleanest-next-experiments)
10. [Suggested reading order](#10-suggested-reading-order)

## 1. Current answer

The evidence does **not** support a scalar notion of a “better model.” Improvements are
conditional on target, task, checkpoint, readout, support set, and sometimes the exact
query. The strongest current statements are:

- **Different models preserve different examples.** On Facebook, the weaker Election
  specialist uniquely solves 175/1,024 validation queries that the stronger TwiBot
  specialist misses; TwiBot uniquely solves 258. Their per-query oracle reaches .852
  accuracy. The shared raw target evidence is often already correct, and source-specific
  computation destroys different subsets of it.
- **Changing evidence can rescue one example and break another.** Alternative valid
  supports rescue 38/100 selected Ukraine failures and 27/100 selected Hong Kong
  failures at least once, but random replacement also breaks 53% and 84% of matched
  correct controls at least once. Support sensitivity is real; random replacement is
  not a remedy.
- **A graph can help some targets and hurt others.** Pair additions and ladder steps
  have signed, target-specific effects. Certain source combinations are complementary;
  others are redundant or mildly interfering. Source count alone is not explanatory.
- **The same change can improve one task and degrade another.** Graph-entry effects are
  common for NM and repaired static link prediction, but weak/inconsistent for
  classification. MIX improves the historical regression mean relative to NM while
  losing classification and valid LP. Better pretraining loss can accompany worse
  transfer.
- **Failure often occurs after useful evidence is already present.** Common raw or
  intermediate support-fitted readouts can outperform the final models. The proximal
  failure is often learned transformation/readout compatibility, not absence of target
  signal.
- **Hong Kong NM is a particularly hard, multi-cause case.** Degree, recurring hard
  queries, membership ambiguity, support identity/context, class-reference construction,
  and learned value geometry all matter descriptively. None alone explains most errors,
  and the tested bounded geometry repair failed on original HK NM.

## 2. The analysis lenses

| Scale | Unit whose outcome changes | Intervention or comparison | What “harm” means |
|---|---|---|---|
| Example | One fixed query | Compare checkpoints or readouts on bit-identical input | A formerly correct query flips, even if aggregate performance rises. |
| Support/class | One fixed query–anchor case | Replace supports, resample contexts, suppress roles, change aggregation | True-class evidence helps one case but moves another toward a competitor. |
| Episode | A 30-way episode or anchor class | Condition/permutation over query, anchor, supports, and candidates | Errors cluster beyond persistent query difficulty. |
| Representation stage | Raw → encoder → metagraph → final decision | Fit common support readouts at each stage | A stage destroys previously decodable target evidence. |
| Source graph | Add or remove a pretraining graph | Pair lattice, inclusion ladder, LOO, fixed exposure | Target A improves while B declines, or the mixture trails its best constituent. |
| Schedule/exposure | Same sources, different sampling order/allocation | Interleaved, blocked, balanced, proportional, fixed exposure | A source or target benefits at another's expense. |
| Objective/task | Change SSL loss or downstream metric | NM/CL/FP/MIX, classification/regression/LP | A gain on the optimized/native task does not transfer to another task. |
| Architecture/readout | Change encoder or decision rule | PRODIGY/SAMGPT/VISION/GILT, SAGE/PinSAGE, alternative readouts | Rankings reverse because model families consume the same target cues differently. |
| Data/protocol | Repair artifact or evaluator | Alignment audits, split correction, valid pairwise LP | An apparent tradeoff disappears or reverses because it was not a model effect. |

## 3. Example-level failures

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| Can the globally weaker source model still be better on individual examples? | [Local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Yes. Election uniquely wins 175 Facebook validation queries and TwiBot uniquely wins 258; oracle accuracy is .852 versus .681 for the best fixed expert. | Replicates the decomposition on a second stream; one selected source pair. |
| Are the source-model differences already present in raw target inputs? | [Local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Not in the tested contrast. Raw logits are identical and raw ridge is .832/.904 accuracy/AUC. Rankings emerge after source-specific computation through unequal destruction of useful evidence. | Strong localization, not a complete causal account of training. |
| Can simple input regions predict which source will win? | [Local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | No useful reversal was recovered from outcome-blind input clusters; input-only routers did not beat always choosing TwiBot. | One contrast; rules out a simple local-distance story, not all routers. |
| What do concrete flips look like? | [Individual examples](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_EXAMPLES.md) | Twenty-four hydrated cases show semantically plausible raw decisions being preserved by one source path and broken by another; query-only and support-only interventions separate roles. | Selected examples illustrate mechanisms; they are not prevalence estimates. |
| Is query difficulty persistent? | [Canonical NM difficulty](evaluation/error_audit/FINDINGS_NM_DIFFICULTY.md) | Yes, especially native Ukraine: validation/test per-node correctness correlates, and some repeatedly sampled nodes are always wrong in the observed stream. | Repeated occurrences are not independent, and “always wrong” is not intrinsic unsolvability. |
| Are high-degree queries simply the failures? | [NM degree audit](evaluation/error_audit/FINDINGS_NM_DEGREE.md) | Accuracy generally declines toward hubs, even after distinct-neighbor and node-weighted checks, but not monotonically. Degree co-varies with repetition and ambiguity. | Association, not a causal explanation. |
| Can a support-aware rule selectively avoid example-level degradation? | [Local transfer contrast / TRACE](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Intermediate support-consistency routing improves accuracy on all five tested targets, macro +.027, while AUC can move differently (COVID −.003). | Useful proof of selective correction; not a universal source selector. |

## 4. Support and episode interference

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| Does changing only the true-class support set alter a fixed query's outcome? | [Paired support resampling](evaluation/error_audit/FINDINGS_NM_SUPPORT_RESAMPLING.md) | Yes. Alternative identities rescue selected failures more often than same-ID context resampling, but also destroy many correct controls. | Establishes conditional dependence, not a deployable selector. |
| Does geometric separation of support inputs predict rescue? | [Fixed-query support geometry](evaluation/error_audit/FINDINGS_NM_SUPPORT_GEOMETRY.md) | Aggregate support/query summaries track some flips but miss many; nearly unchanged summaries can accompany different decisions. | Simple pre-encoder geometry is insufficient for selection. |
| Are whole sampled inputs enough to explain failures? | [Complete sampled-input audit](evaluation/error_audit/FINDINGS_NM_COMPLETE_INPUTS.md) | Degree, overlap, repetition, ambiguity, and sampled-input descriptors explain slices, but a large residual remains. Exact overlap supplies independent descriptive evidence. | Comprehensive accounting, not causal decomposition. |
| Do failures cluster at class or episode level beyond query identity? | [Episode-detail audit](evaluation/error_audit/FINDINGS_NM_EPISODE_DETAIL.md) and [difficulty audit](evaluation/error_audit/FINDINGS_NM_DIFFICULTY.md) | All-four-query failures within an anchor class exceed within-query permutations on both targets. Extra whole-episode clustering is clearer on Ukraine than HK. | Localizes dependence to class/episode composition without naming the cause. |
| Can deliberately favorable support extremes repair HK? | [HK support extremes](evaluation/error_audit/FINDINGS_NM_HK_SUPPORT_EXTREMES.md) | Extreme selection can expose useful evidence in some cases but does not yield a stable overall fix; favorable and harmful movements coexist. | Diagnostic intervention with true-anchor knowledge. |
| Where do HK errors enter: membership, prototypes, or score construction? | [HK failure localization](evaluation/error_audit/FINDINGS_NM_HK_FAILURE_LOCALIZATION.md) | Membership ambiguity explains a minority; replacements can make evidence available without final recovery; true-class and competitor score paths both move. | Narrows the mechanism but leaves most errors unexplained. |
| Does the frozen encoder contain rescue information that the metagraph loses? | [HK frozen encoder vs metagraph](evaluation/error_audit/FINDINGS_NM_HK_MECHANISM.md) | Under replacement, useful information can exist before the native readout; learned geometry tracks loss changes better than raw means. Original errors are not simply ignored correct prototypes. | Separates original failures from intervention-induced recoverability. |
| Can the inferred HK mechanism be repaired without labels? | [HK goal and bounded repair](evaluation/error_audit/FINDINGS_NM_HK_GOAL.md) | Controlled key/value tests support a value-direction mechanism, but the frozen geometry residual fails on original HK NM. | Mechanistic evidence; negative repair result. |
| Do support/query roles have target-dependent effects? | [Role-context replay](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_ROLE_CONTEXT.md), [support dose](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_SUPPORT_DOSE.md) | Suppressing support messages improves HK political ranking yet monotonically worsens bot ranking; signs can reverse across checkpoints. | Direct evidence that one intervention helps one target while harming another. |
| Is naturally selecting more favorable supports a general repair? | [Natural support](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_NATURAL_SUPPORT.md) | No. Prespecified political predictions pass, but support-only selection often worsens Ukraine political NLL and trails context averaging. | Larger labeled pool is not standard 10-shot evaluation. |

## 5. Representation and readout failure

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| At what stage do strong and weak source models diverge? | [Canonical source stages](evaluation/error_audit/FINDINGS_NM_SOURCE_STAGES.md) | On identical Ukraine/HK inputs, common intermediate heads recover some final errors and lose some final successes. Source effects are distributed across encoder geometry and final readout. | Complete original inputs for two checkpoints and two targets. |
| Is the target signal absent, or consumed badly? | [Local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Often consumed badly: raw ridge is strongest, then performance falls through pre-metagraph, post-metagraph, and full-model stages. | Facebook contrast; broader health correlations support generality. |
| Does intermediate support agreement predict transfer? | [Transfer-health analysis](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Across 135 source–target–seed cells, centered U1 agreement correlates .710 with accuracy and .698 with AUC. | Fails as a quality signal when all models are near chance on Suspended; competence floor needed. |
| Are source rankings properties of the graph alone? | [Cross-model matching](graphs/transfer_prediction/target_performance_mechanisms/FINDINGS_CROSS_MODEL.md) | No. Political source rankings reverse across trained families; TwiBot rankings can reverse within PRODIGY when the readout changes. | Matched target episodes/readouts; metric semantics remain family-specific. |
| Are training failures due only to final decoder updates? | [Collapse and message-path studies](graphs/transfer_prediction/target_performance_mechanisms/README.md) | The campaign localizes several effects to support/reference message paths, role interactions, and learned readout—not one universal decoder defect. | Read individual findings linked from the campaign README; many are bounded controls. |
| Can a more expressive neighborhood encoder fix transfer? | [PinSAGE screen](graphs/encoders/nm_pinsage_fast_ablation/FINDINGS.md) | No in the bounded screen: PinSAGE loses all 16 matched cells to GraphSAGE. | One seed; sufficient to reject the proposed broad sweep, not PinSAGE universally. |

## 6. Graph-to-graph interference

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| When a graph is added, which targets gain or lose? | [NM pairwise source additions](transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/README.md) | The 36-pair intervention matrix contains 648 directional target deltas and separates incumbent, newly added, and jointly held-out effects. Signs are target- and pair-specific. | Seed 0, 2,500 steps; descriptive fixed-budget interventions. |
| Does a mixture beat every constituent on held-out targets? | [Ladder-order robustness](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS.md) | Sometimes. Order C is positive in 20/21 held-out residuals (mean +.0449); Orders A/B average about −.004. Complementarity depends on composition, not count. | One seed; exposure and optimization interference are not separated. |
| Can adding a strong donor erase apparent mixture synergy? | [Ladder-order robustness](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS.md) | Yes. Order C's COVID residual rises to +.0749, then falls to −.0041 after the strong Ukraine donor enters. | Mixture-versus-best-constituent residual, not additive synergy. |
| Is own-target entry uniformly beneficial? | [Fixed-exposure ladder](transfer/ablations/prodigy_nm/source_exposure/nm_ladder_fixed_exposure_nhop2/FINDINGS.md) | For measurable NM entry events, yes: all improve, mean +.103 AUC, even with expected exposure per active source fixed. | Stronger evidence for coverage; one training seed. |
| Do those entry gains transfer to other downstream tasks? | [Downstream ladders](transfer/ablations/prodigy_nm/downstream/nm_ladder_downstream_nhop2/FINDINGS.md) | Repaired LP entry is positive in 19/21 cases; classification only 9/19. | Same encoder can improve relational retrieval while failing to improve labels. |
| Does balancing sources avoid small-graph harm? | [COVID–Midterm merge](transfer/matrices/prodigy_nm/merged_vs_single/nm_covid_midterm/RESULTS.md) | Balanced within-source sampling rescues Midterm above the naive merge and its specialist, but trades performance on COVID. | Direct source-allocation tradeoff; not a universal sampling rule. |
| Does exposure/order explain all negative transfer? | [LOO schedule signal](../setup/nm_loo_schedule_signal/RESULTS.md) and [TRACE schedule scaling](graphs/transfer_prediction/trace_schedule_scaling/FINDINGS.md) | No universal schedule fix. Proportional beats uniform for held-out Ukraine at every block size; all-nine k=1 slightly lowers macro and strongly harms Suspended. | Three seeds in the LOO schedule study; target-specific conclusion. |
| Can graph descriptors tell us which donor will hurt? | [Similarity vs transfer](graphs/transfer_prediction/similarity_vs_transfer/FINDINGS.md), [path-feature coupling](graphs/structure_features/path_feature_coupling/FINDINGS.md), [local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md) | Feature-cloud distance and support-readout health predict transfer better than degree distance. But no single descriptor explains source effects, and raw input clusters fail at per-query routing. | Useful population-level predictors, not causal donor-selection laws. |

## 7. Task, objective, and architecture tradeoffs

### 7.1 What the MLP program adds

These studies are not PRODIGY runs. They live primarily in the sibling
`mixture-scaling` worktrees and isolate feature-only or fixed-context learning from
PRODIGY's sampled GNN/metagraph computation. They belong here because they show that
source interference survives after message passing, episodic labels, and the metagraph
are removed. The mechanism is therefore broader than a PRODIGY-specific attention or
support-reference defect.

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| Does graph-specific transfer remain when the encoder cannot message-pass? | [Node-only MLP transfer protocol](/Users/philipp/projects/gfm/mixture-scaling-node-only/docs/node_only_transfer.md) and [specialist–ladder comparison](/Users/philipp/projects/gfm/mixture-scaling/results/mlp_transfer_explains_ladder/FINDINGS.md) | Yes. Source-specialist rankings and target-specific transfer remain with independent endpoint encoding. The cumulative ladder is predicted much better by the best available specialist than by the specialist mean; later UKR/COVID/Midterm declines remain unexplained. | Nine sources, six matched LP targets, one seed; checkpoints/budgets differ and later historical rungs include the defective Suspended artifact. |
| Does adding fixed neighborhood information uniformly improve a feature MLP? | Nonzero-mini three-view transfer, Tucker `mixture-scaling-input-shift/results/nonzero_mini_transfer/FINDINGS.md` | No. Node+10-neighbor context raises mean LP AUC from .6110 to .6550 and wins 66/81 cells, but improves raw BCE in only 19/81 and lowers same-graph mean AUC from .7409 to .7275. Facebook falls from .8574 to .6903. | Seed 0; larger input also has more parameters. Current LP variants use learned bias and uniform-negative evaluation, so consult the later correction records before quoting a matrix. |
| Is poor probability behavior an unavoidable representation tradeoff? | Midterm decoder test, Tucker `mixture-scaling-input-shift/results/decoder_bias_midterm/FINDINGS.md` | No. Adding a learned bias lowers held-out BCE from .6177 to .1078 and raises AUC from .9552 to .9822; learned scale adds little. This is a concrete decoder bottleneck that improves both ranking and calibration on one graph. | One graph and seed; no cross-graph claim. Important counterexample to “every change must trade something off.” |
| Does the cumulative MLP ladder learn additive mixture value? | [MLP ladder explanation](/Users/philipp/projects/gfm/mixture-scaling/results/mlp_transfer_explains_ladder/FINDINGS.md) | Mostly no. Best-specialist gain explains 51.6% of squared error versus a flat baseline (64.6% before Suspended), while every rung-9 target remains 1.42–3.16 AUC points below its best specialist. Target inclusion drives much of the apparent ladder gain. | Descriptive oracle predictor; not a causal mixture decomposition. |
| Does fixing sparse/dead ReLU activity improve transfer? | [MLP ladder diagnostics](/Users/philipp/projects/gfm/mixture-scaling/results/mlp_ladder_diagnostics_20260911/FINDINGS.md) | Only locally. LeakyReLU improves COVID Political validation BCE, native AUC, and effective rank, while foreign mean AUC falls .16 points; four targets improve and four decline. | Short seed-0 screen; Suspended target was corrupted. |
| Can loss switching retain one source without sacrificing the stronger source's transfer? | Asynchronous KD, Tucker `mixture-scaling-input-shift/results/async_convergence/FINDINGS.md` | Partly. Replacing converged Facebook BCE with distillation improves all six transfer targets versus matched and extended joint-BCE controls, but still trails the Ukraine singleton on four of six. | Seed-0 pilot; improves the joint-training control, not a preservation guarantee. |
| If Ukraine training continues, can Facebook retention and Ukraine fitting both reach their singleton floors? | KD extension, Tucker `mixture-scaling-input-shift/results/async_extension/FINDINGS.md` | No observed checkpoint reaches both. At equal 50k Ukraine exposure, retaining Facebook KD buys +3.3545 Facebook AUC points at a −.1247 Ukraine-point cost versus Ukraine-only continuation. | Controlled continuation of one pair/seed; direct retention–fitting frontier. |
| Does a weaker KD coefficient remove that frontier across seeds? | Seed replication, Tucker `mixture-scaling-input-shift/results/async_seed_replication/FINDINGS.md` | It reliably beats weight 1 in seeds 1 and 2, but exceeds the stronger singleton in only one. Seed 1 improves both source tests; seed 2 degrades both. Source-validation retention does not guarantee source-test retention. | Two new training seeds; repeatedly inspected transfer targets. |
| What does the failed seed reveal under matched exposure? | Seed-2 Ukraine-only control, Tucker `mixture-scaling-input-shift/results/async_seed2_control/FINDINGS.md` | At matched Ukraine exposure, Facebook KD costs .1767 points of six-target transfer and .0753 Ukraine-test points while retaining 2.9728 more Facebook-test points. Removing KD improves transfer but causes severe Facebook forgetting. | Prespecified matched continuation; isolates continued KD after a shared parent, not the whole original pair deficit. |
| Can endpoint weight interpolation escape the retention frontier? | Weight-merge grid, Tucker `mixture-scaling-input-shift/results/async_weight_merge/FINDINGS.md` | Not for the 11 declared mixtures. Facebook passes its singleton validation floor only at Ukraine weights 0–.1; Ukraine passes only at .3–1.0. No candidate qualifies for downstream evaluation. | Exact two endpoints and coarse alpha grid; no claim about other merges or layer-wise methods. |

The MLP program sharpens the overall conclusion in two ways. First, interference is
already present in learned transformations of raw and fixed-context features; a graph
neural message-passing explanation is insufficient. Second, not every failure is a
fundamental Pareto limit: the Midterm decoder-bias intervention improved both BCE and
AUC. The productive distinction is between **correctable parameterization defects** and
**source-retention frontiers** created by shared learned weights.

### 7.2 Other objective and architecture evidence

| Question | Experiment | Finding | Status / boundary |
|---|---|---|---|
| Does combining SSL objectives improve all tasks? | [Multitask SSL](objectives/multitask/multitask_ssl/FINDINGS.md) and [valid LP rescore](objectives/multitask/multitask_ssl/FINDINGS_rescore.md) | No. NM leads classification (.810) and valid LP (.757); MIX loses both (.795/.680) but retains positive mean regression (.097 versus NM −.001). | One training seed; task target sets differ. |
| Does lower native loss mean better transfer? | [SAMGPT C5 saturation](transfer/ablations/samgpt_graphcl/saturation/samgpt_c5_saturation/FINDINGS.md) | No. TwiBot transfer peaks at the first saved 200-update checkpoint while GraphCL loss continues improving. | Clear checkpoint tradeoff; one seed. |
| Does correcting the training implementation improve transfer? | [SAMGPT correctness ablation](transfer/ablations/samgpt_graphcl/saturation/samgpt_covid_correctness_ablation/FINDINGS.md) | Corrected prompt routing lowers loss but worsens tested transfer; resampling costs more without useful gain. | A technically “better” native objective path can be worse downstream. |
| Do engineered topology/feature objectives produce a balanced representation? | [Topology-feature SSL](objectives/topology_vs_features/topology_feature_ssl/FINDINGS.md) | E4/E4r hurt retained classification/regression metrics; historical LP evidence is invalid pending rescore. | Negative result with an explicit evaluator boundary. |
| Does removing graph propagation reveal a cleaner transferable feature model? | [Node-MLP ladder](transfer/ablations/node_mlp/node_mlp_ladder_20260911/RESULTS.md) | Mean LP AUC rises across the cumulative ladder, but individual targets frequently decline. LeakyReLU improves native COVID Political slightly while lowering foreign mean by .16 points. | Rungs 6–9 used a defective Suspended artifact and unequal convergence; not clean causal ladder evidence. |
| Do architecture rankings stay fixed across targets? | [ICL architecture matrix](transfer/matrices/cross_architecture/icl_arch_matrix/FINDINGS.md) and [source lattice comparison](transfer/matrices/cross_model/source_lattice_comparison/README.md) | No universal winner. Architecture/objective and target interact; pair-impact distributions differ across model families and native metrics. | Do not compare raw native metric scales across families. |

## 8. Data and evaluation failures that can masquerade as model failures

| Question | Investigation | Finding | Consequence |
|---|---|---|---|
| Were old static-LP tradeoffs real? | [Multitask rescore](objectives/multitask/multitask_ssl/FINDINGS_rescore.md) | No. The old episodic LP path was center-blind, used frozen random prototypes, and had degree-confounded negatives. | All pre-23-July static-LP numbers are void unless explicitly rescored; temporal LP remains invalid. |
| Did split leakage/incorrect edge membership drive the first NM audit? | [Canonical split audit](evaluation/error_audit/FINDINGS_NM_CANONICAL_SPLIT.md) | The original diagnostic required replacement. The corrected audit uses verified held-out edges and bit-identical realized inputs. | Use canonical-split findings, not the historical error cohorts. |
| Were Election/COVID Political feature clouds misaligned? | [Alignment audit](graphs/features/election_covid_alignment_audit/FINDINGS.md) | No alignment defect. Full-population IDs, hashes, embeddings, and CSV parsing agree. Geometry reflects population/class mixture plus upstream cleaning. | Do not “repair” canonical alignment; raw Election profiles are the remaining causal comparison blocker. |
| Was Suspended's abnormal geometry a model phenomenon? | [Suspended repair note](../../../docs/ukraine_suspended_repair_20260911.md) and [node-MLP warning](transfer/ablations/node_mlp/node_mlp_ladder_20260911/RESULTS.md) | The canonical artifact was defective and replaced; its nonzero feature KS falls from .2184 to .0732 after repair. | Pre-repair Suspended models/results require provenance warnings or reruns. |
| Do repeated `--seed` evaluations supply episode uncertainty? | Repository evaluation audit summarized in [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md) | No. Episode sampling is determined by split name; `--seed` affects label downsampling, not episode resampling. | Do not report seed sweeps as eval-episode confidence intervals. |
| Are nominal final checkpoints comparable? | Terminal-save audit summarized in [EXPERIMENT_OVERVIEW.md](EXPERIMENT_OVERVIEW.md) | Pre-26-July runs stop one checkpoint interval short; newer runs emit a true terminal checkpoint. | Pin actual comparison steps; never select the largest checkpoint blindly across eras. |

## 9. Open questions and the cleanest next experiments

These are ordered by how directly they answer “when does helping one thing hurt
another?” rather than by implementation convenience.

1. **Completed first pass: example-level interference matrix.** The
   [source-addition interference study](graphs/transfer_prediction/example_interference/FINDINGS.md)
   records rescue, break, churn, margins, and stage-resolved correctness for 56
   directional additions on two targets and two episode streams. Every addition both
   rescues and breaks examples. The remaining step is to add degree, ambiguity, and
   support-health strata and extend beyond the two available replayed targets.
2. **Run matched source-addition replays on identical episodes and tensors.** Pair the
   singleton, pair, and LOO checkpoints on the same query/support inputs. This connects
   the graph-level lattice directly to individual examples without training new models.
3. **Separate coverage, dilution, and optimization interference.** For selected
   positive and negative graph pairs, match per-source exposure and total optimization
   steps, then compare joint training to checkpoint ensembling and sequential training.
4. **Trace rescued and broken examples layer by layer.** Apply the existing common
   raw/pre-metagraph/post-metagraph readouts to source-addition pairs. Ask whether a new
   source changes available evidence, destroys it in the encoder, or changes only the
   label/reference readout.
5. **Turn support sensitivity into a risk measure, not a selector first.** Estimate
   per-query decision variance over valid support/context draws and test whether high
   variance predicts cross-source disagreement and graph-addition breaks on a fresh
   stream.
6. **Use multi-objective Pareto accounting.** For every intervention, retain a vector
   of NM, classification, valid LP, regression, calibration, and compute. Count an
   intervention as non-degrading only within predeclared tolerances; do not collapse
   incompatible target sets into one average.
7. **Repair the remaining provenance boundaries before broad reruns.** Rebuild merged
   artifacts/caches containing old Suspended features, recover raw Election profiles if
   possible, and exclude invalid temporal/static-LP paths.

### Minimal new table to produce

| Base model | Intervention | Target | Rescued examples | Broken examples | Net accuracy | Margin change | Stage first diverging | Support-variance stratum |
|---|---|---|---:|---:|---:|---:|---|---|

This table would join the two bodies of work that are currently separate: graph-level
source-addition deltas and example-level error mechanisms.

## 10. Suggested reading order

For the shortest path through the evidence:

1. [Local transfer contrast](graphs/transfer_prediction/local_transfer_contrast/FINDINGS.md)
   — clearest demonstration that aggregate gains hide example-level destruction and
   complementarity.
2. [Canonical NM audit](evaluation/error_audit/FINDINGS_NM_CANONICAL_SPLIT.md) and
   [paired support resampling](evaluation/error_audit/FINDINGS_NM_SUPPORT_RESAMPLING.md)
   — corrected failure population and direct fixed-query evidence intervention.
3. [HK failure localization](evaluation/error_audit/FINDINGS_NM_HK_FAILURE_LOCALIZATION.md),
   [mechanism](evaluation/error_audit/FINDINGS_NM_HK_MECHANISM.md), and
   [bounded repair](evaluation/error_audit/FINDINGS_NM_HK_GOAL.md)
   — the most detailed current “what goes wrong and why” chain.
4. [Pairwise source additions](transfer/matrices/prodigy_nm/source_additions/nm_pairwise_source_additions/README.md)
   and [ladder-order robustness](transfer/ladders/prodigy_nm/robustness/nm_ladder_order_robustness/FINDINGS.md)
   — graph-level positive and negative transfer.
5. [Target-performance mechanisms](graphs/transfer_prediction/target_performance_mechanisms/README.md)
   — the larger intervention campaign across roles, support paths, schedules,
   checkpoints, and model families.
6. [Experiment overview](EXPERIMENT_OVERVIEW.md) — complete program context and older
   experiments not central to failure/interference.

## Maintenance rule

Add a row when a study answers a distinct causal or diagnostic question. Record the
comparison, the signed finding, the unit of replication, and the strongest validity
boundary. Do not treat another target, checkpoint, episode stream, or metric computed
from the same trained checkpoint as an independent training replicate.
