# One individually audited CPU/GPU AUC tie

The first complete-mixture replay at frozen revision `420a0b05` stopped on
`nmloo_without_cp_hk` evaluated on Ukraine suspension. Original GPU AUC is
0.5087890625; CPU AUC is 0.508819580078125. Accuracy and F1 agree exactly.
The original run and failed replay are preserved on Tucker; this is not a
replacement for the historical result.

The direct audit at `4c89f1b7`, in Tucker worktree
`/dataMeR1/phil/gfm/prodigy-mechanisms-complement`, used identical checkpoint
tensors and all 32 original cached batches. One new unhooked CPU pass reproduced
every exported CPU logit bit-exactly. All three direct GPU passes on idle GPU 3
reproduced the historical AUC, accuracy and F1 exactly. The only cross-label
ordering change is positive query 233 versus negative query 21: CPU probabilities
tie at 0.8394830226898193; the GPU makes the negative score slightly larger.
The resulting AUC difference is exactly one half-pair contribution,
0.5 / (128 positive × 128 negative) = 0.000030517578125. Every predicted class is
unchanged. `audit.json` retains every repeat, numerical difference and pair score.

An earlier attempt to use the stage-observation replay on GPU stopped because
two GPU forwards differed at roundoff level; its bit-exact hook-observation guard
was not relaxed. The successful direct audit has no stage hooks and reports all
three GPU passes. GPU use was confined to this small numerical check after an
occupancy check; the substantive experiments remain CPU-only.

The mixture continuation may use an explicitly numerically aligned reference
**copy** for this one model/target, generated from this audit. The source snapshot
remains immutable. General AUC tolerance remains 1e-5 and decision tolerance
1e-6. Reports must distinguish 224 strict original mixture cells from this one
individually audited numerical cell, and preserve both AUC values. No ensemble
or complementary-error outcomes were interpreted before this amendment.
