# NM single-source downstream transfer — findings

**Setup.** Freeze each of the eight matched-40k single-source neighbor-matching
(NM) encoders from `nm_single_source_matrix` and evaluate it on:

- node classification: four labeled graphs, 10-shot ROC-AUC;
- node regression: four profile graphs × three targets
  (`followers_count`, `statuses_count`, `account_age_days`), 10-shot Spearman
  after `log1p`, averaged over the three targets below.

All 128 cells use the original `state_dict_40000.ckpt` files. The Ukraine source
is `nm_ss_ukr_rus_twitter`, not the older ladder checkpoint. This is a single-seed
frozen-encoder evaluation.

## Results

### Node classification (ROC-AUC)

| NM source | covid political | election 2020 | ukr suspended | twibot20 | mean |
|---|---:|---:|---:|---:|---:|
| ukr | .905 | .985 | .498 | **.631** | **.754** |
| covid | .907 | .983 | .498 | .602 | .748 |
| twibot20 | .890 | .979 | .495 | **.629** | .748 |
| midterm | .821 | .976 | .498 | .525 | .705 |
| ukr suspended | .732 | .885 | .466 | .567 | .662 |
| cp_hk | .722 | .799 | .505 | .579 | .651 |
| covid political | **.921** | .980 | .489 | .523 | .728 |
| election 2020 | .880 | **.985** | **.527** | .522 | .728 |

### Node regression (Spearman, mean over three targets)

| NM source | ukr | covid | midterm | twibot20 | mean |
|---|---:|---:|---:|---:|---:|
| ukr | −.018 | .014 | −.044 | **.054** | .002 |
| covid | −.082 | −.071 | −.121 | −.121 | −.099 |
| twibot20 | −.082 | −.064 | −.032 | −.159 | −.084 |
| midterm | −.024 | **.017** | −.046 | .011 | −.010 |
| ukr suspended | **.020** | .017 | −.032 | .050 | **.014** |
| cp_hk | −.035 | −.035 | −.031 | −.137 | −.060 |
| covid political | −.001 | −.011 | −.047 | −.012 | −.018 |
| election 2020 | .003 | .001 | **−.031** | −.041 | −.017 |

## Takeaways

1. **Classification transfer is mostly controlled by the evaluation graph.**
   Election 2020 is easy for nearly every encoder (.799–.985), while Ukraine
   suspended is at chance for every encoder (.466–.527). Twibot20 retains some
   source sensitivity (.522–.631), but the spread is much smaller than the
   target-to-target shift.

2. **The broad NM donors remain the safest classification sources.** Ukraine,
   covid, and twibot20 are the top three source means (.754/.748/.748), matching
   the broad-donor group in the NM transfer matrix. The narrow cp_hk and Ukraine
   suspended sources are the weakest overall (.651/.662).

3. **There is no general downstream specialist advantage.** Matching source and
   target wins clearly for covid political and election 2020. Twibot20 is
   effectively tied with the Ukraine source (.629 vs .631), and the Ukraine
   suspended specialist is actually the worst source on its own task (.466).
   The strong diagonal from the NM pretext matrix therefore does not carry over
   as a general diagonal on classification.

4. **Regression is a clean null.** Source means range only from .014 to −.099,
   and every matched in-domain cell is negative (−.018, −.071, −.046, −.159).
   No single-source NM encoder yields a useful or source-matched profile
   regression representation under this frozen 10-shot probe.

Overall: **choose a broad source such as Ukraine/covid/twibot20 if a single NM
encoder must support classification, but changing the single pretraining source
does not solve profile regression.**

## Caveats

- One seed. Eval episodes are fixed by split, so the scores are paired across
  sources but do not provide across-seed confidence intervals.
- This experiment compares pretrained sources, not pretraining against
  `random_init` or raw-feature floors. It supports source-ranking and null-transfer
  conclusions, not a causal claim that NM improves classification over no
  pretraining.
- Regression values above are dataset means over three targets. The full 8 × 12
  matrix is in `data/regression.csv`.

## Artifacts

- `data/model_manifest.csv`: exact checkpoint provenance
- `data/classification.csv`: classification matrix
- `data/regression.csv`: full dataset × target regression matrix
- `data/regression_by_dataset.csv`: regression matrix shown above
- `data/results_long.csv`: all 128 tidy result rows
- `figures/single_source_downstream_heatmaps.{png,pdf}`: final figure
