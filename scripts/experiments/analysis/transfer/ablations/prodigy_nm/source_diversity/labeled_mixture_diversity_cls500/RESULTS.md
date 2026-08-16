# Results

## Answer

**At a fixed 500-step budget, broader labeled mixtures improve held-out classification on average, but the effect is target- and donor-dependent rather than universal.**

Across targets, mean held-out ROC-AUC rises from 0.7101 at one source to 0.7464 at four sources (difference +0.0363; linear slope +0.0118 per added source).

Target-specific slopes: `covid_political` +0.0260, `election2020` -0.0004, `facebook_page_reference` +0.0013, `ukr_rus_suspended` -0.0042, `twibot20` +0.0361.

The positive macro curve is driven by Covid Political and TwiBot. Election 2020 is already near ceiling, Facebook is flat on average, and Ukraine Suspended remains near chance.

## Endpoint controls

The four-source held-out mean is 0.7464, versus 0.7395 for target-only pretraining and 0.7419 for all-five pretraining. Adding the target to the mixture therefore changes mean AUC by -0.0045; it does not produce a general in-domain jump under fixed total compute.

The endpoint response is also heterogeneous: all-five helps Facebook strongly, is neutral on Election, and is worse than the held-out four-source model on Covid Political, Ukraine Suspended, and TwiBot.

## What drives the curve

Subset-lattice contrasts in `data/marginal_donor_effects.csv` show donor compatibility, not source count alone. TwiBot benefits most from adding Covid Political and Election; Facebook benefits from TwiBot but is hurt by Ukraine Suspended; Covid Political benefits most from Ukraine Suspended and TwiBot.

## Scope

All models use seed 0, 500 optimizer steps, and 500 paired evaluation episodes per target. Paired fingerprints remove evaluation-set variation across arms, but there is no training-seed uncertainty yet. The experiment estimates the practical effect of diversity under fixed total compute; it does not hold per-source exposure constant.
