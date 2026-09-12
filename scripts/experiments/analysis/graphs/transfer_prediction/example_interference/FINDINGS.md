# Example-level source-addition interference

## Question

When adding a pretraining graph improves or degrades aggregate transfer, what happens
to the individual target examples underneath the net change? Does the pair model merely
fix errors, merely break successes, or do both at once? At what representation stage do
the pair and singleton first produce different correctness outcomes under a common
support-fitted readout?

## Main result

**Every tested source addition both rescues and breaks target examples.** Aggregate
improvement is the difference between two substantial opposing flows, not a monotone
expansion of the singleton's correct set. This holds for all 56 directional
interventions on both the original and independently sampled fresh episode streams.

For each source pair `{A,B}`, the analysis treats the saved pair checkpoint as two
directional interventions: `A → A+B` and `B → A+B`. All comparisons use bit-identical
target queries, supports, sampled subgraphs, and episode mappings within a stream.

| Stream | Target | Directional additions | Mean net accuracy | Mean churn | Median churn | Positive / negative net changes | Both rescue and break |
|---|---|---:|---:|---:|---:|---:|---:|
| Original | COVID Political | 28 | +3.94 pp | 16.68% | 13.12% | 16 / 12 | 28 / 28 |
| Fresh | COVID Political | 28 | +4.12 pp | 16.52% | 13.09% | 16 / 12 | 28 / 28 |
| Original | Facebook page-reference | 28 | +1.53 pp | 28.53% | 27.25% | 17 / 11 | 28 / 28 |
| Fresh | Facebook page-reference | 28 | +1.79 pp | 29.15% | 27.10% | 18 / 10 | 28 / 28 |

`churn = (rescued + broken) / queries`. COVID Political has 3,072 queries per
intervention; Facebook has 1,024. The means average directional interventions, not
independent training replicates.

The cancellation is large. On the fresh stream, an average COVID Political addition
rescues 46.89% of the singleton's errors while breaking 7.86% of its correct examples.
The net is positive because the starting correct/error pools differ, not because harm
vanishes. On Facebook, the corresponding means are 44.24% rescued and 20.97% broken,
producing much higher churn and a smaller net gain.

## Large improvements still contain many breaks

The largest fresh-stream improvement is adding Ukraine to the Facebook specialist on
COVID Political:

| Base → pair | Base accuracy | Pair accuracy | Rescued | Broken | Net | Churn |
|---|---:|---:|---:|---:|---:|---:|
| Facebook → Facebook+Ukraine | 56.41% | 86.65% | 1,088 | 159 | +30.24 pp | 40.59% |
| Facebook → Facebook+TwiBot | 56.41% | 77.99% | 949 | 286 | +21.58 pp | 40.20% |
| Suspended → Suspended+Ukraine | 66.80% | 88.09% | 830 | 176 | +21.29 pp | 32.75% |

Even the +30.24-point intervention breaks 159 formerly correct queries. “Better” is
therefore not set inclusion: the pair's correct set is different, not just larger.

## Net degradation also contains real rescues

The largest fresh-stream declines are likewise mixtures of benefit and harm:

| Base → pair | Base accuracy | Pair accuracy | Rescued | Broken | Net | Churn |
|---|---:|---:|---:|---:|---:|---:|
| TwiBot → TwiBot+COVID Political, on COVID Political | 83.56% | 77.51% | 206 | 392 | −6.05 pp | 19.47% |
| COVID Political → COVID Political+TwiBot, on COVID Political | 83.27% | 77.51% | 262 | 439 | −5.76 pp | 22.82% |
| TwiBot → TwiBot+HK, on COVID Political | 83.56% | 77.93% | 126 | 299 | −5.63 pp | 13.83% |
| Ukraine → Ukraine+Suspended, on Facebook | 70.02% | 65.14% | 103 | 153 | −4.88 pp | 25.00% |

Thus a negative aggregate delta does not mean that the added source contains no useful
signal. It means breaks outnumber rescues under this jointly trained checkpoint.

## Where the differences appear

Raw target inputs are identical, so the shared raw-joint ridge readout never differs.
Among queries whose **final full-model correctness differs**, the first observed
correctness disagreement between singleton and pair under the sequence of common ridge
readouts is:

| Stream / target | Encoder convolution | Encoder pooling | Pre-metagraph | Post-metagraph | Only at full native decision |
|---|---:|---:|---:|---:|---:|
| Original / COVID Political | 22.8% | 5.1% | 17.0% | 24.5% | 30.6% |
| Fresh / COVID Political | 24.1% | 5.6% | 16.1% | 23.3% | 31.0% |
| Original / Facebook | 20.4% | 14.6% | 25.7% | 16.6% | 22.6% |
| Fresh / Facebook | 18.4% | 14.6% | 27.8% | 16.2% | 23.0% |

This distribution replicates closely. It rejects a single-stage account: source
addition changes usable decision geometry in the encoder, before and after metagraph
processing, and in roughly 23–31% of final flips only the native final decision differs
while all earlier common ridge readouts agree in correctness.

“First observed” is descriptive, not a causal mediation claim. Each stage uses a
separately support-fitted ridge readout, so correctness may diverge and reconverge across
stages. The table identifies the earliest tested stage with a decision disagreement; it
does not prove that later modules caused or preserved the final flip.

## Replication

Across the same 56 directional interventions, original-versus-fresh net accuracy
changes correlate **r=.972** and churn correlates **r=.991**. Net-change signs agree in
49/56 interventions (87.5%). The fresh stream changes target episodes but reuses the
same trained checkpoints, so this is episode-stream replication, not an independent
training-seed replication.

## What changed in our interpretation

Graph-level pair deltas previously showed that source effects could be positive or
negative. This analysis reveals the hidden structure:

1. No tested source addition is uniformly helpful or uniformly harmful at the example
   level.
2. Large net improvements can coexist with 30–40% decision churn.
3. Negative additions still rescue nontrivial numbers of examples.
4. Facebook exhibits substantially more replacement of the correct set than COVID
   Political, even though its mean net effects are smaller.
5. Source interaction appears throughout the learned path rather than at one universal
   bottleneck.

The operational object to predict is therefore not only the aggregate source–target
delta. It is the pair of conditional risks:

```text
P(pair correct | singleton wrong, query/support context)
P(pair wrong   | singleton correct, query/support context)
```

A useful source-addition rule must raise the first without allowing the second to erase
the gain.

## Protocol and evidence

- Seed-0 frozen singleton and pair checkpoints from the completed classification
  lattice; no new training or inference.
- Two target tasks: COVID Political and Facebook page-reference.
- Original and fresh saved episode streams; exact cached labels were read from the
  replay batches rather than inferred from ordering.
- Full-model reconstructed accuracies match the saved official metrics.
- Singleton and pair cache receipts are equal within target/stream, and episode
  fingerprints differ between original and fresh streams.
- Seven views: raw joint ridge, convolution-center ridge, pooled ridge, pre-metagraph
  ridge, post-metagraph ridge, final-input ridge, and the native full model.
- The pair checkpoints were trained jointly; this does not separate extra source data,
  fixed-budget dilution, optimization interference, or capacity allocation.
- Directional comparisons sharing a pair checkpoint are dependent. Queries, targets,
  stages, and streams are not training-seed replicates.

Artifacts:

- [Analyzer](analyze.py)
- [Original full-model interference matrix](data/original_interference.csv)
- [Fresh full-model interference matrix](data/fresh_interference.csv)
- [Original stage outcomes](data/original_stage_outcomes.csv)
- [Fresh stage outcomes](data/fresh_stage_outcomes.csv)
- [Original validation receipt](data/original_validation.json)
- [Fresh validation receipt](data/fresh_validation.json)

Tucker source exports remain read-only under
`/dataMeR1/phil/gfm/prodigy-mechanisms/log/target_mechanisms/`. Aggregate outputs were
generated in the isolated worktree
`/dataMeR1/phil/gfm/prodigy-example-interference` at revision `36843ffd`; result commit
`cc6b5f43`. Local development used branch `main`; published review branch is
`codex/example-interference`.
