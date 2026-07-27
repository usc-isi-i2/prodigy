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

![classification heatmap](figures/heatmap_classification.png)

The same three-way split appears independently in all three arms. Per-graph ROC-AUC for `all8`:

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

Against a random-init encoder these two are unambiguous: covid_political 0.429 and
election2020 0.403 untrained (both *below* chance), versus 0.932 and 0.987 pretrained —
gaps of +0.50 and +0.58. Whatever else is uncertain here, this effect is real.

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

## 4. Node regression measures nothing — and there is a mechanism, not just a null

![regression heatmap](figures/heatmap_regression.png)

Mean Spearman never leaves the band around zero and has no coherent trend in step.
Across all 144 cells: 50/144 positive, mean −0.029, median −0.043, range −0.38 to +0.48.

**A random-init encoder does just as well** (`data/random_init_floor.csv`, measured
2026-07-27 with the `RANDOM_INIT` sentinel):

| | pretrained (18 ckpts × 4 cells) | random init |
|---|---|---|
| mean \|ρ\| | 0.150 | 0.069 |
| max \|ρ\| | 0.484 | **0.174** |

An **untrained** encoder reaches ρ = +0.174 on twibot20/followers_count, and **62 % of
all pretrained cells fall below that value**. There is no evidence any of these encoders
carry usable signal for 10-shot profile regression.

**Why, mechanically.** Nothing in the model was ever trained to regress. The collator
feeds the support targets in as metagraph edge attributes
(`metagraph_edge_value = y_values * (~query_mask)` — correctly masked, so no leakage), but
these encoders were pretrained on `neighbor_matching`, where that same `edge_attr` slot
carries a support/query indicator rather than a continuous magnitude. Eval runs with
`--eval_only True`, so no gradient step ever adapts the weights that read it. There is no
regression head at random init — there is no regression head at all; the scalar pathway is
simply untrained. Reading these numbers as "transfer" over-reads them.

**Consequently the apparent structure in this panel is noise**, including the one pattern
that looked alarming: `ukr` flips sign negative→positive across the 500→1000 boundary and
`covid` flips positive→negative, in 5/8 and 8/8 cells, with 500→1000 the largest adjacent
gap for both — and that boundary is exactly the splice. It aligned suspiciously with the
pre/post-code-drift split (`all8`'s historical run is post-drift and shows 1/8 cells and a
+0.006 delta). But a channel whose untrained floor is |ρ| = 0.17 has no signal to be
discontinuous; sign flips are what a quantity centred on zero does. See §5 for why the
channel that *does* carry signal shows no boundary effect at all.

Not checked against `../node_regression/data/features_only_floor.csv` — the random-init
floor above is the stronger control and reaches the same conclusion.

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
  would also pass a model it should reject. Weight-space distance cannot resolve trajectory
  identity after ~1000 chaotic steps; always compute the unrelated scale (√2·‖w‖ ≈ 1708
  here) before trusting a distance threshold.

  **The best available evidence on the splice is the classification panel itself.** It is
  the only channel with signal well above its random-init floor, and its 500→1000 step —
  the joint — moves by −0.011 / +0.012 / +0.010 for all8/ukr/covid, i.e. no more than the
  plateau wobbles anyway. The one apparent discontinuity showed up in regression (§4),
  which the random-init floor shows is measuring nothing. The stronger test remains
  available and unrun: evaluate the dense `state_dict_1001` against the historical
  `state_dict_1000` in metric space (~12 jobs per arm).
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
