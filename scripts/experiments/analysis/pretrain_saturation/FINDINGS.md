# Pretrain saturation — findings

**Question.** Does downstream transfer performance saturate early in pretraining, and does
the width of the pretraining corpus change when?

**Design.** 3 pretraining corpora × 6 checkpoints (100, 500, 1000, 2000, 10 000, 40 000),
each frozen and scored on node classification (4 graphs, 10-shot) and node regression
(4 graphs × 2 targets, 10-shot). 216 eval jobs, all completed, zero failures.
Setup: [`pretrain_saturation_existing`](../../setup/pretrain_saturation_existing/) (steps
1000+, no training) and [`pretrain_saturation_dense`](../../setup/pretrain_saturation_dense/)
(steps 100/500, three 2100-step retrains).

![saturation curves](figures/pretrain_saturation.png)

## 1. Classification transfer saturates very early — before step 500

Mean ROC-AUC across the four labelled graphs:

| step | all8 | ukr | covid |
|---|---|---|---|
| 100 | 0.566 | 0.572 | 0.572 |
| 500 | **0.766** | 0.732 | 0.717 |
| 1000 | 0.754 | 0.745 | 0.727 |
| 2000 | 0.761 | 0.747 | 0.750 |
| 10 000 | 0.767 | 0.764 | 0.756 |
| 40 000 | 0.761 | 0.758 | 0.746 |

Expressed as the fraction of each arm's eventual gain (step 100 → its best checkpoint)
already realised at step 500:

| arm | gain by step 500 | plateau spread, steps 500–40 000 |
|---|---|---|
| all8 | **99.1 %** | 0.013 |
| ukr | 83.4 % | 0.032 |
| covid | 78.5 % | 0.040 |

The remaining **98.75 % of the training budget** (500 → 40 000 steps, an 80× increase)
buys `all8` nothing measurable. This is the headline the experiment was built to test, and
it is only visible because of the dense 100/500 checkpoints — no prior run in this repo
wrote a checkpoint below step 1000, so the entire rise was previously invisible.

## 2. But the mean hides that transfer only works on two of the four graphs

Per-graph ROC-AUC for `all8`:

| step | covid_political | election2020 | twibot20 | ukr_rus_suspended |
|---|---|---|---|---|
| 100 | 0.581 | 0.521 | 0.668 | 0.493 |
| 500 | 0.901 | 0.987 | 0.667 | 0.508 |
| 1000 | 0.910 | 0.983 | 0.612 | 0.512 |
| 2000 | 0.923 | 0.983 | 0.653 | 0.487 |
| 10 000 | 0.932 | 0.983 | 0.638 | 0.516 |
| 40 000 | 0.927 | 0.984 | 0.626 | 0.508 |

Three separate things are going on, and averaging them into one curve misrepresents all
three:

- **covid_political and election2020** carry the entire effect. Both jump between step 100
  and 500 and are then flat to 40 000. This is where "saturates by step 500" is true.
- **ukr_rus_suspended sits at chance (0.487–0.516) at every checkpoint.** Pretraining
  never does anything for it. A flat line at 0.5 is not saturation.
- **twibot20 gets *worse* with pretraining** — 0.668 at step 100 down to 0.626 at 40 000,
  its best checkpoint being the *least*-trained one.

**`election2020` reaching 0.987 deserves scrutiny before it is cited.** A near-perfect
frozen-probe ROC-AUC after 500 pretraining steps is more consistent with an easy or leaky
label than with representation quality.

## 3. Corpus width changes how fast, not how high

All three arms converge to 0.746–0.767 — a 0.021 spread, comparable to the plateau
wobble of a single arm. What differs is the approach: `all8` is 99 % of the way there at
step 500, the two single-source arms 79–83 %, and they keep creeping up until ~10 000.

So the prior expectation — that a wider mixture needs *more* steps — is not what the data
shows; if anything the broad corpus arrives **sooner**. State this weakly: the gaps
(0.013 vs 0.032/0.040 in plateau spread) are the size of the run-to-run noise this
experiment cannot measure (§5).

## 4. Node regression is a null result, not a saturation curve

Mean Spearman never leaves the band around zero (−0.21 to +0.16) and has no coherent
trend in step for any arm. Across all 144 regression cells: **50/144 positive**, mean
−0.029, median −0.043, range −0.38 to +0.48. Both targets behave the same
(`followers_count` mean −0.040, `account_age_days` −0.018).

The right-hand panel should be read as "these frozen encoders carry no usable signal for
10-shot profile regression", **not** as "regression saturates". Nothing saturates because
nothing rises. This has not been checked against
`../node_regression/data/features_only_floor.csv`, which is the obvious next step before
any claim is made about regression.

## 5. What this evidence cannot support

- **There are no error bars.** One pretraining run per arm. Two *identical* reruns of the
  `all8` config — same seed, same GPU, same code — land 546 apart in weight space (‖w‖ ≈
  1208), because PyG scatter message-passing uses non-deterministic CUDA atomics and
  training is chaotic. The plateau in §1 spans ±0.013; **whether that is stability or
  scatter is not determined by this data.** The three replicate runs are trained and sit
  in `prodigy-sat/state`; evaluating one costs ~48 jobs and would settle it.
- **The curve is spliced.** Steps 100/500 come from retrains done 2026-07-27; steps 1000+
  from runs of 2026-06-14 (ukr, covid) and 2026-07-09 (all8).
  `setup/pretrain_saturation_dense/check_splice.py` passes, but **weakly**: a model trained
  on a *different corpus* sits 587 away, versus 546 for the same-config null, so the test
  would also pass a model it should reject. Weight-space distance cannot resolve this
  question. The splice rests on the mechanism argument — no code drift on the plain-NM
  path between June and now, verified by reading the diffs (see the setup README) — not on
  a measurement. The sound version of the check is in metric space: evaluate the dense
  `state_dict_1001` and compare against the historical `state_dict_1000` already in the
  table (~12 jobs).
- **Steps ≥1000 are labelled one step short.** Pre-2026-07-26 checkpoints hold N+1
  completed steps under the name N. Irrelevant numerically; do not present the axis as
  exact.
- **Eval episodes are fixed by split name**, so the ± inside an eval log is the spread
  across episodes within one eval, not a confidence interval. Do not report it as one.

## Reproducing

```bash
/opt/homebrew/bin/python3.11 build_tables_and_figure.py
```

Reads the shared per-task CSVs, filters to `sat_*` model keys, and rewrites
`data/pretrain_saturation_{long,wide}.csv` and the figure.
