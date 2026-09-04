# Multi-graph NM intervention results — 2026-09-04

Completed **144 training runs and all 1,296 NM evaluation cells**: baseline plus 16 individual arms at eight rungs, followed by eight frozen-recipe repetitions. Each model was evaluated on all nine graphs with 512 fixed episodes per target. The last NM evaluation finished at 12:42:08 UTC, about 3.38 hours after the goal began. All required NM work is complete; CLS, LP and further seeds remain deferred.

At the eight-source endpoint, improvements on included graphs did not generally carry over to unseen TwiBot-20. The auxiliary objective, feature standardization and cross-graph classes improve the included-source mean. Proportional exposure, uniform unique-neighbor positives and degree-matched negatives improve the unseen endpoint. Proportional exposure has the largest unseen gain (+0.007492 AUC) alongside an included-source decline (-0.009611).

The table reports each arm against the fresh paired baseline at **rung eight**. Improved means Δ > 0.001, degraded means Δ < -0.001, and inconclusive means within that practical band. These are descriptive single-seed verdicts. Every row has complete results across all eight rungs; no incomplete arm is counted as a result. Baseline AUC: included-source macro mean **0.861197**, unseen TwiBot-20 **0.897545**, fixed nine-target mean **0.865236**.

| Intervention                              |   Included Δ AUC | Included status   |   Unseen Δ AUC | Unseen status   |
|:------------------------------------------|-----------------:|:------------------|---------------:|:----------------|
| Size-proportional exposure                |        -0.009611 | degraded          |       0.007492 | improved        |
| 64-episode source blocks                  |        -0.008009 | degraded          |      -0.017644 | degraded        |
| Cross-graph classes                       |         0.001848 | improved          |      -0.002219 | degraded        |
| Uniform degree-band centers               |        -0.003657 | degraded          |      -0.006133 | degraded        |
| Low-degree eligibility                    |        -0.011843 | degraded          |      -0.007042 | degraded        |
| Uniform unique-neighbor positives         |        -0.000931 | inconclusive      |       0.004084 | improved        |
| Degree-matched negatives                  |        -0.004032 | degraded          |       0.003234 | improved        |
| One-hop training context                  |        -0.016826 | degraded          |      -0.013007 | degraded        |
| Unit episode-gradient norm                |        -0.001611 | degraded          |      -0.003021 | degraded        |
| Per-node feature standardization          |         0.002235 | improved          |       0.000353 | inconclusive    |
| Source-specific affine transforms         |        -0.01566  | degraded          |      -0.058598 | degraded        |
| 512-dimensional encoder                   |         0.000362 | inconclusive      |      -0.00308  | degraded        |
| NM + masked-feature reconstruction        |         0.002528 | improved          |      -0.000192 | inconclusive    |
| Loss-adaptive degree sampling             |        -0.006232 | degraded          |      -0.005226 | degraded        |
| Cyclic center coverage                    |        -0.0022   | degraded          |      -0.001245 | degraded        |
| Per-source episode budget                 |        -0.000964 | inconclusive      |      -0.002625 | degraded        |
| Frozen recipe repetition (objective only) |         0.003363 | improved          |       0.000188 | inconclusive    |

The combined-stage selection was frozen at 11:25:33 UTC, before unseen evaluation, using only original training-source validation. **Only the auxiliary objective passed**: mean paired validation gain +0.001381, positive at six of eight rungs, worst mean per-source change -0.003473. Feature standardization missed the mean-gain gate (+0.000737). Cross-graph classes exceeded that gate (+0.002426) but failed the per-source guard (-0.022882 versus the permitted -0.01). Recomputing selection from all 136 original histories gives the identical recipe, and all input hashes match the freeze.

Consequently, the eight second-stage runs are same-seed objective repetitions; they provide no test of interactions between interventions. The recipe gains +0.003363 on included sources and +0.000188 on unseen TwiBot-20. It **does not exceed both baseline and the best individual arm by the required margin** in any panel:

| Panel                  |   Recipe Δ baseline | Best individual       |   Recipe Δ best | Versus best   | Beats both?   |
|:-----------------------|--------------------:|:----------------------|----------------:|:--------------|:--------------|
| Included-source mean   |            0.003363 | Auxiliary objective   |        0.000834 | inconclusive  | No            |
| Unseen TwiBot-20       |            0.000188 | Proportional exposure |       -0.007304 | degraded      | No            |
| Fixed nine-target mean |            0.00301  | Auxiliary objective   |        0.000784 | inconclusive  | No            |

The repeated objective runs have identical scientific parameters; differences are run metadata and some physical GPU assignments. Selected validation scores vary by as much as 0.003373 across these same-seed repetitions. This is observed repeat variation, with no inferred cause, and is not an independent-seed confidence interval. A positive table entry alone does not establish robustness across seeds or held-out graphs.

Training consumed **1,333,000 episodes**. There were 51 validation-plateau stops and 93 cap stops; 51 capped models still gained more than 0.001 at their final validation check. These are bounded-training comparisons, not universal convergence results. The frozen cap was 10,000 episodes, with the budget diagnostic capped at 1,250 × rung. Validation used 16 fixed episodes per active training source every 2,000 episodes, patience two checks and a 0.001 meaningful-gain threshold. The actual highest validation checkpoint was retained, with earlier exact ties preferred. The recipe selected 6k checkpoints at rungs four and six, 8k at rung seven, and 10k elsewhere.

All models use seed zero and one nested source order. TwiBot-20 is excluded from training and selection; the catalog confirms NM, classification and static-LP support. The 13,472 cumulative exposure records and all 144 terminal records pass source exclusion and episode-total checks. All training losses are finite. Final NM audit verifies actual checkpoint hashes, exact model/target membership, shared per-target episode fingerprints across both stages, all 648 active-source validation replay records, and the required stage ordering. Manual curve reviews and the resolved earlier evaluation interruption are documented in the run log.

Use the [fixed nine-target ladder](figures/all_targets_ladder.png) and [unseen ladder by intervention](figures/unseen_by_arm.png) for scaling comparisons. Included-source means change membership as the rung grows. The `all_rung_included_delta` column in the detailed table weights each of the 36 included graph–rung cells equally. These results use the new protocol's paired baseline; historical plots require matching budgets and metric protocols before direct comparison.

[Endpoint comparison](figures/endpoint_deltas.png) · [Per-graph effects](figures/endpoint_by_graph.png) · [Detailed findings and costs](FINDINGS.md) · [All NM cells](data/nm_results.csv) · [Exact configs and checkpoint records](data/model_records.json) · [Reproduction protocol](../../../../../setup/nm_interventions_overnight/README.md) · [Execution log](RUN_LOG.md)

[Training audit](data/training_audit.json) · [Full evaluation audit](data/final_evaluation_audit.json) · [Selection provenance](data/selection_provenance_audit.json) · [Recipe parameter parity](data/recipe_replication_audit.json) · [Frozen recipe](data/combined_selection.json)

Execution used Tucker GPUs **1–3** because GPU 0 failed allocation preflight; GPUs 4–7 were untouched. Individual training is pinned to `6260524e` in `prodigy-nmi-overnight`; its evaluator revisions are retained per cell. Recipe training and evaluation are pinned to private revision `0fc562a1` in separate `prodigy-nmi-combined` and `prodigy-nmi-combined-eval` worktrees. These Tucker directories are under `/dataMeR1/phil/gfm/`. The private code transfer used git directly to Tucker.

Local evidence lives on branch `codex/nm-interventions-overnight` in `/Users/philipp/projects/gfm/prodigy/.worktrees/nm-interventions-overnight`. Automatic approval review rejected publishing the research artifacts to the public GitHub repository without explicit publication consent. Results remain committed locally and retained on Tucker. No research computation remains pending within this NM campaign.
