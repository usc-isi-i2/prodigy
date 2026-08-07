# All-nine source-complete batch-9 diagnostic

This experiment compared the historical all-nine batch-1 NM model with batches that
contain exactly one internally within-source episode from each of the nine sources.
All models were evaluated with the same native neighbor-matching protocol: 3-shot,
30-way, 12 queries per class, and 500 fixed test episodes on each graph.

## Result

| evaluation graph | batch 1, 40k steps | batch 9, 4,444 steps | delta | batch 9, 40k steps | delta |
|---|---:|---:|---:|---:|---:|
| COVID-19 Twitter | 0.6009 | 0.5924 | -0.0086 | 0.6163 | +0.0153 |
| COVID Political | 0.2519 | 0.2041 | -0.0478 | 0.2228 | -0.0291 |
| CP-HK Twitter | 0.2427 | 0.2189 | -0.0238 | 0.2384 | -0.0043 |
| Election 2020 | 0.2805 | 0.2127 | -0.0678 | 0.1748 | -0.1057 |
| Facebook Page Reference | 0.7688 | 0.7404 | -0.0284 | 0.7792 | +0.0104 |
| Midterm | 0.3434 | 0.3379 | -0.0055 | 0.3736 | +0.0302 |
| TwiBot-20 | 0.4192 | 0.4162 | -0.0030 | 0.4347 | +0.0155 |
| Ukraine-Russia Suspended | 0.4035 | 0.3814 | -0.0221 | 0.4626 | +0.0591 |
| Ukraine-Russia Twitter | 0.4502 | 0.4415 | -0.0087 | 0.4728 | +0.0226 |
| **mean** | **0.4179** | **0.3939** | **-0.0240** | **0.4195** | **+0.0016** |

The 4,444-step batch-9 checkpoint saw approximately the same number of episodes as
the batch-1 baseline but was 2.40 accuracy points worse. The 40k-step batch-9
checkpoint saw nine times as many episodes and only tied the baseline on average
(`+0.16` points), while Election 2020 fell by 10.57 points. Source-complete batch 9
therefore does not improve sample efficiency and should not replace batch 1 in its
current form.

## Saved runs

- batch-1 checkpoint: `state/nm_ladder_ordA_r9_facebook_06_08_2026_09_49_01/checkpoint/state_dict_40000.ckpt`
- batch-9 checkpoints: `state/nm_all9_source_complete_b9_06_08_2026_22_32_07/checkpoint/state_dict_{4444,40000}.ckpt`
- batch-9 evaluation log: `log/nm_all9_source_complete_batch/eval_all9_both_ckpts/eval.log`

All paths are relative to `/dataMeR1/phil/gfm/prodigy-facebook` on Tucker. This is a
single training run per condition; the comparison does not establish seed-level
uncertainty. A batch-9 independent-source control would be needed to separate the
effect of batch size from forcing all nine sources into every update.
