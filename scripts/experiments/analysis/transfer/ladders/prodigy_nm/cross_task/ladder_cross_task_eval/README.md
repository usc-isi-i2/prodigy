# PRODIGY ladder cross-task evaluation

This analysis closes the two missing cells in the PRODIGY ladder task/budget grid:

- the seed-0, 100-step architecture-matrix ladder evaluated on the frozen native
  neighbor-matching streams; and
- the three-seed, 2,500-step final-core ladder evaluated on the frozen downstream
  classification streams.

The joined plot also uses the already-published converse cells: 100-step downstream
classification and 2,500-step native neighbor matching. Within each task, both budgets
use exactly the same evaluation graphs, task construction, episode counts, and episode
fingerprints. Training architecture, graph set, and sampler settings match; the 100-step
and 2,500-step models come from separate training campaigns, and their configs also
differ in loader-worker and logging/checkpoint controls.

Run locally with Homebrew Python 3.11:

```bash
/opt/homebrew/bin/python3.11 -m \
  scripts.experiments.analysis.transfer.ladders.prodigy_nm.cross_task.ladder_cross_task_eval.analyze
```

The two primary figures are `figures/budget_task_ladders.png` (accuracy) and
`figures/budget_task_ladders_auc.png` (ROC-AUC). Open markers show the best rung at
each budget; the blue band is the sample standard deviation across the three
2,500-step training seeds. Step 100 has one training seed and therefore no seed band.
Step-2,500 native-NM AUC comes from the complete four-decimal metric recovery in the
original fixed-test worker logs; all other plotted cells use full-precision evaluator
outputs.

`figures/nm_auc_budget_100_2500_40000.png` is an explicitly exploratory native-NM
overlay. It restricts all means to the eight targets shared with the historical 40k
ladder. The 40k curve ends at rung 8 and uses the legacy evaluator. Only Order A has the
same source set at each rung; the historical Order B/C source sequences differ from the
current sequences and are plotted only as context. Its machine-readable inputs are in
`data/nm_auc_budget_100_2500_40000.csv`.

Two synthesis views combine this evidence with the existing saturation study:

- `figures/budget_phase_transition_auc.png` and
  `data/budget_phase_transition_auc.csv` show endpoint performance versus budget;
- `figures/mix_regret_auc.png` and `data/mix_regret_auc.csv` show the AUC cost of using
  the all-source endpoint rather than the best earlier rung.

The source experiment and launcher live under
`scripts/experiments/setup/ladder_cross_task_eval/`.
