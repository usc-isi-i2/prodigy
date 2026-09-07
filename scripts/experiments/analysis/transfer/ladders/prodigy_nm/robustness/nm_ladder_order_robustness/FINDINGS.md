# NM ladder order robustness — mixture synergy

## Question

Can a mixture of source graphs transfer to a held-out graph better than every source in
that mixture does alone?

For each order, rung, and graph that is still held out, define the mixture residual as

```text
mixture residual = mixture NM AUC − max(single-source NM AUC among sources in the mixture).
```

The held-out target's own specialist is never eligible. A positive residual is the
operational definition of **mixture synergy** here: the mixture exceeds every constituent
available at that rung. This should not be called a literal "sum of parts," because AUC is
not additive.

The source table is
[`data/all_targets_heldout_mixture_vs_best_constituent_grid.csv`](data/all_targets_heldout_mixture_vs_best_constituent_grid.csv),
and the aligned per-target view is
[`figures/all_targets_heldout_mixture_vs_best_constituent_grid.png`](figures/all_targets_heldout_mixture_vs_best_constituent_grid.png).

## Main result: synergy depends on the graph combination

Rung 1 is omitted below because a one-source mixture and its only constituent are the same
model by construction.

| order | positive residuals | mean residual | range |
|---|---:|---:|---:|
| A | 8/21 | −0.0044 | −0.0201 to +0.0071 |
| B | 6/21 | −0.0042 | −0.0270 to +0.0121 |
| **C** | **20/21** | **+0.0449** | **−0.0041 to +0.0894** |

Orders A and B are well described by a **max rule**: the mixture usually matches the best
single source already present, with small losses more common than gains. Order C is
qualitatively different. Its weak/isolated early-source combinations produce NM transfer
that is consistently better than any one constituent alone.

Therefore, simply adding more graphs is not sufficient for synergy. **Some graph sets are
complementary; others are redundant or mildly interfering under this protocol.** Donor
identity and composition matter more than mixture size by itself.

## Concrete synergy: Election '20 + COVID political + CP-HK

At Order C rung 3, the training set contains only Election '20, COVID political, and
CP-HK. The mixture beats its best constituent on every graph that remains held out:

| held-out target | mixture − best constituent AUC |
|---|---:|
| TwiBot-20 | +0.0804 |
| COVID-19 | +0.0749 |
| UKR–RUS | +0.0660 |
| UKR–RUS suspended | +0.0493 |
| Midterm | +0.0440 |
| **mean** | **+0.0629** |

Thus, if these three graphs are the only available sources, joint NM pretraining learns
something useful for unseen graphs that no individual source model provides on its own.
This is the clearest evidence for source complementarity in the ladder.

## A strong donor can collapse the residual back to the max rule

COVID-19 is the final held-out target in Order C. Its mixture advantage rises from +0.0337
with `{Election '20, COVID political}` to +0.0749 after adding CP-HK. The advantage then
shrinks as stronger donors enter. Once UKR–RUS is included at rung 7, the seven-source
mixture reaches 0.9689 AUC versus 0.9730 for the UKR–RUS singleton, a residual of −0.0041.

The narrow conclusion is that this mixture does not improve on an already strong UKR–RUS
donor for COVID-19. It does **not** establish that UKR–RUS can never be complemented by any
other graph; the experiment covers only these prefixes and one training seed.

## Why small declines are plausible

Every rung receives the same 40k-step total training budget. With balanced source
sampling, increasing the number of sources reduces the expected number of updates from
each individual source to roughly `40k / number of sources`. A large mixture can therefore
finish slightly below its best singleton because the strongest donor receives less
exposure. Optimization interference is another possible explanation.

This experiment does not identify the cause. The negative residuals should be described as
**consistent with fixed-budget dilution**, not as proof that dilution caused them. A
matched-per-source-exposure experiment is required to separate composition from training
budget.

## Boundary: downstream classification is not clear

The strong Order C effect is an NM-to-NM transfer result. It does not carry cleanly to
10-shot downstream node classification. Across the three orders, only 7/11 target-entry
events improve, with mean delta +0.006 AUC and one-sided `p = 0.274`. Within Order C,
COVID political improves by +0.040 and TwiBot-20 improves by +0.039, while UKR–RUS
suspended declines by −0.011; Election '20 begins at rung 1 and has no pre-entry contrast.

Accordingly, the defensible claim is:

> Certain source combinations produce genuine synergy for held-out neighbor matching,
> while other combinations remain near the best constituent or decline slightly. The
> same composition effect is not yet established for downstream classification.

The downstream evidence and its one-seed limitation are documented in
[`nm_ladder_downstream/FINDINGS.md`](../../../../ablations/prodigy_nm/downstream/nm_ladder_downstream/FINDINGS.md).

## Limitations

- All order-robustness models use training seed 0. The target cells within a rung share one
  trained checkpoint and are not independent replicates.
- The comparison is against single-source models at the same nominal total step count, not
  at matched exposure per source.
- Synergy is defined only relative to the best constituent currently inside the mixture;
  it is not an additive decomposition or a causal interaction estimate.
- The evidence supports claims about the observed prefixes, not every possible graph
  subset or curriculum.
