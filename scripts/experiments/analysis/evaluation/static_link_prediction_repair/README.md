# Static-LP evaluator repair

The 2026-07-23 repair of the static-link-prediction evaluator, and the rescore of the
15 frozen checkpoints it invalidated. This folder is the **method**; the results and
their reading live with the experiment they belong to.

| what | where |
|---|---|
| the repaired evaluator | `../../../eval/pair_link_eval.py`, `pair_link_sweep.py`, `pair_link_ckpt.py` (+ `eval/tests/`) |
| how the repair was executed | [`../../setup/slp_evaluator_repair/EXECUTION.md`](../../../setup/slp_evaluator_repair/EXECUTION.md) |
| the rescored results | [`../multitask_ssl/data/pair_lp/`](../../objectives/multitask/multitask_ssl/data/pair_lp/) |
| the defect list + rescore writeup | [`../multitask_ssl/FINDINGS_rescore.md`](../../objectives/multitask/multitask_ssl/FINDINGS_rescore.md) |
| what the rescore did to the lattice | [`../multitask_ssl/FINDINGS.md`](../../objectives/multitask/multitask_ssl/FINDINGS.md) |
| this folder | `aggregate_pair_lp.py` — the summary table |

A duplicate `results/` copy and a stale copy of the findings used to sit here; both were
removed on 2026-07-26 in favour of the canonical copies above.

## The defects, in one paragraph

The episodic evaluator scored link prediction center-blind (the query node's own
embedding did not enter the score), against frozen random prototypes, with negatives
drawn without degree matching — so a model could score well by matching the degree
distribution alone. Every static-LP number produced before the repair is void; see the
rescore writeup for the full list and for which downstream findings still need
re-deriving.

## Reproduce the summary

```bash
/opt/homebrew/bin/python3.11 aggregate_pair_lp.py
```

Defaults to the canonical results and the `degree_matched` negatives. Prints the
mean-AUC and margin-over-floor table that must match `FINDINGS_rescore.md`
(NM .757 / +0.113 at the top, FP .499 / −0.145 at the bottom).
