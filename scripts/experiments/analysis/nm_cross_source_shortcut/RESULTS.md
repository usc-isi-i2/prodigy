# Results — NM cross-source-shortcut test

**Run date:** 2026-06-29/30 · **Seed:** 0 · **1 seed only.**
Merged models evaluated at two checkpoints: `@match` (step 50000 = matched total
compute vs the single-source runs) and `@full` (step 110000 = per-domain exposure).
Eval: 3-shot, 30-way.

Hypothesis: naive merged sampling lets an episode's negatives come from the *other*
source, so the model exploits a source-discrimination shortcut instead of within-source
neighborhood structure. Confining each episode to one source removes the shortcut.
(See README.md. The one variable changed vs the proportional merged run is
`neighbor_sampling_episode_source: graph_id`.)

## Comparison (3-shot, 30-way)

accuracy (most discriminative; f1 ≈ accuracy for balanced episodes)
```
regime                       test:ukr            test:covid
                             @match   @full      @match   @full
single ukr (in-domain)       0.5151    —          0.6142    —
single covid (in-domain)     0.4589    —          0.6641    —
merged proportional          0.4790  0.4955       0.6374  0.6574
merged within-source         0.4998  0.5090       0.6592  0.6698
```
roc_auc (near ceiling)
```
regime                       test:ukr            test:covid
                             @match   @full      @match   @full
single ukr                   0.9497    —          0.9741    —
single covid                 0.9245    —          0.9815    —
merged proportional          0.9373  0.9433       0.9778  0.9807
merged within-source         0.9447  0.9472       0.9811  0.9822
```

## Conclusion

Within-source episodes beat the proportional merged baseline on **both** domains at
**both compute levels**, and match/approach the best single-source model — exactly what
the cross-source-shortcut hypothesis predicts. Effect: **@match +0.021 (ukr) / +0.022
(covid)**; **@full +0.014 (ukr) / +0.012 (covid)** in accuracy. (AUC smaller, +0.002 to
+0.007, because it saturates ~0.98.)

**But the effect is small and not yet significant:**
- 1 seed; within-vs-proportional deltas are ~0.012–0.022 accuracy (~0.002–0.007 AUC).
- Context matters: the proportional merged model was *already not worse* than
  single-source (no inversion — see ../nm_transfer_matrix/RESULTS.md), so there was
  little deficit for the within-source variant to recover. The improvement is real in
  direction but marginal in size.

**Verdict:** consistent with the shortcut hypothesis, not proof of it. To confirm,
run multiple seeds (and ideally a harder/larger-n_way regime) and check the delta is
stable and significant.

## Reproduce

```bash
# (Tucker, prodigy env, repo at /dataMeR1/phil/gfm/prodigy)
cd scripts/experiments/nm_cross_source_shortcut
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
./eval_tucker.sh --device 1 --continue-on-error            # 3-shot, 30-way
python3 compare_shortcut.py --log-root /dataMeR1/phil/gfm/prodigy/log --shots 3 --n-way 30
# (reuses the nm_transfer_matrix eval logs for the single/proportional rows)
```
