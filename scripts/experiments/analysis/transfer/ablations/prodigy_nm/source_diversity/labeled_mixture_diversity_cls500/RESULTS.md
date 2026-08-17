# Results

## Answer

**Mixture diversity improves held-out CLS performance under fixed compute, and the positive macro relationship remains after continuing every model to 1,000 steps. The size of the effect is target- and donor-dependent.**

- At 500 steps, macro held-out ROC-AUC moves from 0.7101 for one source to 0.7464 for four sources (+0.0363; slope +0.0118 per added source).
- At 750 steps, macro held-out ROC-AUC moves from 0.7067 for one source to 0.7421 for four sources (+0.0355; slope +0.0115 per added source).
- At 1000 steps, macro held-out ROC-AUC moves from 0.7061 for one source to 0.7475 for four sources (+0.0414; slope +0.0133 per added source).

At 1,000 steps, target-specific diversity slopes are: `covid_political` +0.0309, `election2020` +0.0002, `facebook_page_reference` +0.0053, `ukr_rus_suspended` +0.0035, `twibot20` +0.0263.

## Convergence check

From 750 to 1,000 steps, the 75 held-out cells change by -0.0038 ROC-AUC on average; median absolute change is 0.0075, mean absolute change is 0.0156, and 29/75 cells move by more than 0.01.
A strict model-level rule that continues any model with at least one evaluation cell moving by more than 0.01 selects 20/31 models; they are listed in `data/trajectory_model_convergence.csv`.

This is a checkpoint-stability diagnostic, not proof of asymptotic convergence. The fixed-compute diversity result is stable across checkpoints, but a fully convergence-controlled comparison requires continuing the selected models.

## Endpoint controls

At 1,000 steps, macro ROC-AUC is 0.7475 for the four-source held-out model, 0.7466 for target-only training, and 0.7543 for all-five training. Adding the target to the four-source mixture changes the macro mean by +0.0067.

## Single-source transfer matrix at 1k

The five k=1 models, evaluated on all five targets, form a complete 5x5 single-source transfer matrix at 1,000 steps. Its 20 off-diagonal cells are held-out transfer evaluations and its five diagonal cells are the target-only controls. The wide and long tables are saved as `data/single_source_transfer_matrix_1000.csv` and `data/single_source_transfer_matrix_1000_long.csv`; the annotated heatmap is `figures/single_source_transfer_matrix_1000.png`.

## Same-protocol standalone transfer versus marginal value

With both quantities measured at 1k under the same seed and evaluation protocol, the 20 off-diagonal cells have Pearson `r=0.30` and overall Spearman `rho=0.11`. The mean within-target donor-rank correlation is `0.64`, showing that standalone strength is substantially more informative for choosing among donors once target difficulty is controlled. However, the top standalone donor is also the top marginal donor for only 2/5 targets, so marginal mixture value is still not a max rule.

## Comparison with final-core 2.5k single-source transfer

After restricting the final-core 2.5k three-seed single-source transfer matrix to these five graphs, standalone transfer strength is not a reliable predictor of marginal value in a mixture. Across the 20 directed off-diagonal cells, Pearson `r=0.28` and Spearman `rho=0.20`. These are descriptive coefficients because cells share donors and targets. The mean within-target donor-rank correlation is `0.00`, and the best standalone donor is also the best marginal donor for 3/5 targets. TwiBot20 is strong under both views; Facebook transfers well as a singleton here but is slightly harmful on average when added to an existing mixture.

This is still a pattern comparison rather than a controlled numerical contrast: the final-core matrix is a three-seed mean at 2.5k steps with 30-way/3-shot evaluation, whereas the marginal matrix uses seed 0 at 1k steps with 10-shot evaluation.

## Scope

All arms use training seed 0 and 500 paired 10-shot CLS evaluation episodes. Fingerprints are identical within each target across all arms and checkpoints. The experiment holds total optimizer steps fixed within each checkpoint; it does not hold per-source exposure fixed and does not estimate training-seed uncertainty.
