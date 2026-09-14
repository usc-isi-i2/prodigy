# Three parallel hypotheses: completed, none advances

All three investigations completed under frozen rules. Twelve training cells
(two experiments × two arms × three seeds) ran 2,000 updates each, with validation
every 50 updates. The temporal-path investigation completed its observational
prerequisite and stopped before feature training. No new 2019 scoring, final refit,
or test-array access occurred. Prior extensive 2019 exploration remains disclosed.

## Results on 2018 validation

| Method | Mean selected Hits@50 | Change versus control | Decision |
|---|---:|---:|---|
| Fixed-pool BCE control | 68.8908% | — | Existing baseline |
| H1: renew negative pool every 50 updates | 68.8164% | −0.0743 pp | Does not advance |
| H2: BCE plus extreme-tail ranking loss | 68.2467% | −0.6441 pp | Does not advance |
| H3: temporal path-age replication | Not a model score | 25/27 usable groups | Insufficient support for feature training |

Each trained model contains 496,385 neural parameters and at most 496,448
conservatively counted inference scalars. Means summarize three independently
optimized models, not an ensemble and not independent dataset replications.

| Seed | Control | Renewed negatives | Renewal difference | Tail loss | Tail difference |
|---|---:|---:|---:|---:|---:|
| 0 | 68.7155% | 68.8686% | +0.1531 pp | 68.7304% | +0.0150 pp |
| 1 | 69.0250% | 68.9069% | −0.1182 pp | 67.7252% | −1.2998 pp |
| 2 | 68.9320% | 68.6739% | −0.2580 pp | 68.2844% | −0.6474 pp |

The predeclared training advancement rule required all three seed differences to
be positive and a mean gain of at least 0.5 percentage point. Neither intervention
satisfies even the all-positive condition. No earlier-year training expansion or
combination sweep followed.

## What we learned

The diagnosed fixed-panel overfitting is real, and both interventions materially
improve late-training behavior. At update 2,000, mean validation Hits@50 is
55.6615% for BCE, 62.7239% for renewal, and 66.9807% for the tail loss. Yet their
best validation-selected checkpoints do not improve. Thus reducing late overfit
is insufficient to raise the best early-checkpoint performance under these recipes.
This does not prove that negative sampling or ranking objectives cannot help.

Renewal produces 887/1,224/632 recovered positives and loses 795/1,295/787 versus
its seed-matched controls. Its small first-seed gain disappears across seeds.
Tail loss produces 1,037/751/1,047 rescues but loses 1,028/1,532/1,436. The balance
of rescues and lost existing hits remains the obstacle; recall gains alone mislead.

H1 changes only negative identities: one independently generated 100,000-pair pool
every 50 updates, with unchanged architecture, BCE, optimizer, positive draws, and
hard-mining schedule. Forty prepared pools include a separate diagnostic probe.
The probe contains 99,981 pairs after excluding 19 overlaps with all optimized
pools. Its fixed-year scores use training positives and are diagnostics only.
Feature generation matched all 100,000 archived feature vectors bit-for-bit.
Cache preparation took 853 seconds, shared across seeds.

H2 changes only the objective: original sampled BCE plus coefficient 1 mean
softplus(1 + negative score − positive score), comparing all 2,048 sampled
positives to the highest-scoring 50 training negatives, refreshed every 10 updates.
It does not use validation negatives in the loss. The result rejects this fixed
loss variant, not all ranking objectives.

For both experiments, positive/uniform-negative/mining-slot random streams match
between paired arms. Model-dependent hard-negative identities naturally differ.
Standalone checkpoint selection is identical in control and treatment: earliest
maximum validation Hits@50. This corrects the old fusion-selector mismatch without
crediting that correction to either treatment.

## Temporal path-age prerequisite

| Primary cohort | Usable negative groups | Younger-positive preference | After deleting best 20% groups |
|---|---:|---:|---:|
| 2017 | 25 | 61.87% | 52.33% |
| 2018 | 27 | 62.10% | 51.27% |

The direction survives in both years, but both fall below the frozen minimum of
30 usable groups. The smaller consistent cohorts are fragile: trimmed preference
falls to 35% and 45%. The 2018 investigation also reuses 110 of 212 cases from the
older audit; it is not independent-data replication. Different models and cohorts
across years prevent a causal interpretation of year differences.

Do not interpret this as falsifying temporal path age. It is insufficient support
to justify the proposed two-feature training ablation. No sample-size rule or
matching caliper was relaxed after outcomes appeared.

## Verification and limits

All 12 selected checkpoints replayed exactly. Each experiment's 120 BCE control
measurements exactly reproduce the archived trajectories. Parent independent audits
recomputed selected metrics by sorting saved negative scores, checked earliest-max
selection, file hashes, paired draw streams, and all aggregate decisions. The temporal
audit verified date logic, graph equality, example path enumeration and group-level
means/trimming. Raw checkpoint/score files remain private runtime artifacts.

The temporal diagnostic had an initial source/revision mismatch. That execution was
rejected, and clean repeated execution produced identical scientific outputs; the
only code difference affected independent audit case selection. Its full operations
record is retained. This is not presented as first-look evidence.

All scores in the main table are 2018 validation results, not 2019 test results.
No HyperFusion win or leaderboard acceptance is established. The best historical
fresh standalone test score remains 70.1569%; prior negative leakage and test-tuned
research remain part of the disclosed history. These negative experiments justify
stopping the tested variants rather than tuning them against 2019.

## Reproduction and locations

Integrated branch: codex/collab-fresh-ensemble in work/ensemble-repo.
H1 runtime: /dataMeR1/phil/gfm/ogbl_collab_compact_joint/negative_renewal_v1;
H2 runtime: /dataMeR1/phil/gfm/ogbl_collab_compact_joint/h2_tail_v1;
H3 admitted runtime: /dataMeR1/phil/gfm/ogbl_collab_compact_joint/h3_replication_v2.

Separate Tucker worktrees and offline W&B records were used. H1 used GPU 0,
H2 GPU 1, and H3 CPU only. The hypothesis-specific analysis leaves contain source,
frozen protocols, complete histories, findings, and audit receipts:
ogbl_collab_negative_renewal, ogbl_collab_tail_objective, and
ogbl_collab_temporal_paths under scripts/experiments/analysis/baselines/.
The parallel_hypotheses leaf adds the parent independent audits and combined summary.
