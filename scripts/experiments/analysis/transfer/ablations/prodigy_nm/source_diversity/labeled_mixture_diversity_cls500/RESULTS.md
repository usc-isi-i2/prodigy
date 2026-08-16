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

## Scope

All arms use training seed 0 and 500 paired 10-shot CLS evaluation episodes. Fingerprints are identical within each target across all arms and checkpoints. The experiment holds total optimizer steps fixed within each checkpoint; it does not hold per-source exposure fixed and does not estimate training-seed uncertainty.
