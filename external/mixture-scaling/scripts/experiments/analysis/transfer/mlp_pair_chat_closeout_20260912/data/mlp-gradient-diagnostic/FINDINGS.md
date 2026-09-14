# Gradient and AdamW-update diagnostic

## Main findings

Raw training-gradient conflict is not a sufficient explanation for negative transfer. In these two cases the sign at initialization is opposite the naive expectation: Ukraine/Facebook (the harmful mixture) starts aligned, while Suspended/Political (the helpful mixture) starts opposed. The measurements instead identify early finite-step overshoot, changing training-to-validation alignment, and later ranking damage that smaller steps alone do not remove.

## Scope

A/B pairs: Ukraine/Russia → Facebook pages; Ukraine suspended → COVID political. These are the prior post-hoc diagnostic cases, not randomly selected confirmation cases. For each pair we used five saved states: original singleton initialization, unconstrained warm-start interleaving's best and final checkpoints, and ranking-constrained best and final checkpoints. The best/final continuation steps are 4k/10k for Ukraine/Facebook and 10k/16k for Suspended/Political. Arbitrary intermediate checkpoints were not retained by training and were not reconstructed.

Eight independently seeded training-batch probes per state; the same probes across states. Each probe uses up to 1,024 positive supervision edges without replacement, five uniform exact-nonedge negatives per positive. Validation uses the original deterministic source-validation sample, seed 0, up to 20×1,024 positives per graph, with disjoint context/supervision and no downstream tests. Trial results share fixed validation data; they do not constitute independent evaluation seeds.

Gradient comparisons are overall and per first linear layer, second linear layer, and decoder bias. Norms are measured before gradient clipping; the actual AdamW update uses the training clip norm 1, lr .0005, and saved optimizer state. Scalar bias cosine is sign agreement only.

For each action, weights and AdamW states are restored into an independent disposable model. Actions: A-supervised BCE, B-supervised BCE; also A+BCE-ranking penalty at rank1 checkpoints. A zero-current-gradient step per state retains momentum and weight decay as a diagnostic control. All validation losses and AUCs are remeasured after the step. No checkpoint or graph is modified. Tests verify optimizer-state non-aliasing and gradient geometry.

## 1. Gradient conflict is stage-dependent and can be compatible with benefit

Mean cosine between raw A/B supervised training gradients:

| Pair | Initial | Control best | Control final |
|---|---:|---:|---:|
| Ukraine + Facebook | +0.658 | +0.523 | -0.187 |
| Suspended + Political | -0.464 | +0.648 | +0.477 |

At initialization all eight Ukraine/Facebook probes are positively aligned; all eight Suspended/Political probes are negatively aligned. The signs are similar in both encoder layers, so this is not just a decoder-bias disagreement. At Ukraine/Facebook's final control checkpoint, all eight full-model comparisons are negative.

In the positive case at initialization:

- A-training vs A-validation gradient cosine: -0.779.
- B-training vs A-validation: +0.751.
- B-training vs B-validation: +0.997.

Thus B opposes an A training direction that locally worsens A validation under an infinitesimal raw-gradient step. Opposing A's training gradient can be useful; preserving every A training direction is not equivalent to preserving generalization. This is a local diagnostic of training/validation mismatch, not proof that all later B steps regularize A.

Raw B-gradient norms are much larger at initialization (mean trialwise B/A ratio 11.6 for Ukraine/Facebook and 8.2 for Suspended/Political). These are pre-clipping ratios, not AdamW effective learning-rate ratios. Magnitude imbalance alone does not establish harmful domination.

## 2. Actual initial AdamW steps overshoot validation loss on A

For the first B update from the initial checkpoint, first-order g_validation·delta predicts improved A BCE in both cases. The full step instead worsens A BCE in all eight probes. Holding the update direction and optimizer state fixed, shrinking the parameter displacement resolves this immediate loss tradeoff:

| Pair | Fraction of B update | ΔA validation BCE | ΔB validation BCE |
|---|---:|---:|---:|
| Ukraine + Facebook | 10% | -0.000685 | -0.013037 |
| Ukraine + Facebook | 25% | -0.000769 | -0.031182 |
| Ukraine + Facebook | 50% | +0.001672 | -0.057129 |
| Ukraine + Facebook | 100% | +0.017098 | -0.088791 |
| Suspended + Political | 10% | -0.000694 | -0.009186 |
| Suspended + Political | 25% | -0.001396 | -0.021880 |
| Suspended + Political | 50% | -0.001605 | -0.039867 |
| Suspended + Political | 100% | +0.001983 | -0.062119 |

At 10% and 25%, both validation BCEs improve in every probe for both pairs. This is direct evidence of finite-step overshoot for A's BCE at initialization. Scaling one AdamW parameter displacement equals changing that step's learning rate; it does not reproduce a full smaller-learning-rate trajectory.

Important distinction: the full initial B update still improves A's AUC (+0.029 pp for Ukraine, +0.223 pp for Suspended). The overshoot result concerns BCE/confidence, and alone cannot explain the final unseen-target AUC deficit.

## 3. Smaller steps do not fix later ranking damage

At Ukraine/Facebook's selected control checkpoint (4k), B steps reduce validation AUC on both graphs in all eight probes, at every tested size:

| Fraction | ΔA AUC (pp) | ΔB AUC (pp) | ΔA BCE | ΔB BCE |
|---|---:|---:|---:|---:|
| 10% | -0.00314 | -0.00306 | -0.000089 | +0.000188 |
| 25% | -0.00805 | -0.00813 | -0.000170 | +0.000519 |
| 50% | -0.01664 | -0.01769 | -0.000170 | +0.001205 |
| 100% | -0.03535 | -0.04118 | +0.000341 | +0.003053 |

A smaller B update can improve A BCE while degrading A AUC. This distinguishes ranking preservation from confidence/loss preservation. At this checkpoint B's training gradient has cosine -0.602 with B's own validation-loss gradient; at the final checkpoint it is -0.660. Further fitting B is locally opposed to its validation objective, consistent with overfitting. These are selected checkpoint states and cannot establish that every historical B update was harmful.

At Ukraine/Facebook's final control checkpoint, full B updates hurt A BCE in 8/8 probes and A AUC in 8/8 probes, while helping B BCE in 8/8. This is direct local evidence of a source tradeoff later in training.

## 4. Optimizer history matters

A zero-current-gradient AdamW update still moves the model because moments and weight decay remain active. For example:

- At Suspended's initial checkpoint, this control increases A BCE by .004115 and B BCE by .031517; AUC drops .124 and .842 pp respectively.
- At its selected unconstrained joint checkpoint, the same type of control improves A BCE by .003395 and B BCE by .000269.

The direction of optimizer-history effects changes with the checkpoint. This does not establish that resetting AdamW is better; it demonstrates why raw gradient cosine cannot predict the actual step by itself. The control combines momentum, second-moment preconditioning, bias correction, and weight decay; it does not isolate any one of them.

## 5. The ranking penalty is not a per-step guarantee

At the constrained Ukraine/Facebook final checkpoint, an A-with-ranking update improves A AUC by .03765 pp versus .02708 pp for A-BCE alone, and B AUC by .08591 versus .07026 pp. But at constrained best checkpoints, both actions can reduce AUC. Penalty weight 1 is a soft preference, and the optimizer state and checkpoint location matter. These local comparisons do not replace the repeated full-run penalty/control experiment.

## What this supports next

The evidence does not justify using raw A/B training-gradient cosine as a gate or assuming all negative cosines should be removed. Better targets are validation-compatible updates and ranking retention.

A focused next training experiment would use the existing warm-start control with a smaller continuation learning rate (for example 0.00005 versus 0.0005), with equal source sampling and otherwise matched settings. Report source-validation AUC as well as BCE and retain the ranking-preservation comparison. This tests whether the early overshoot contributes to final transfer loss. It is not guaranteed to solve the late ranking/overfitting behavior; that may require changing the objective or checkpoint criterion. Do not simultaneously change learning rate, optimizer reset, source weighting, and model capacity if the goal is causal attribution.

No new training has been launched by this diagnostic. Single-seed checkpoints, two post-hoc cases, and eight local batch probes limit generalization. Measurements use training-source validation only; they do not directly identify the cause of unseen-target degradation or text-preprocessing effects.

## Artifacts and provenance

- `gradient_summary.csv`: per-state/layer gradient alignment and magnitudes.
- `update_summary.csv`: full-step validation effects and harmful-probe counts.
- `step_scale_summary.csv`: scaled-step effects.
- `case0/`, `case1/`, `scale0/`, `scale1/`: scalar probe metrics and checkpoint receipts.
- Figure: `gradient_update_diagnostic.png` / `.svg`.
- Local worktree: `/tmp/mlp-pair-error-code`; branch `codex/mlp-gradient-diagnostic`.
- Tucker worktree: `/dataMeR1/phil/gfm/mixture-scaling-gradient-diagnostic`.
- Revisions: `84db0a9` (original diagnostic), `30e720a` (scaled-step follow-up).
- Cluster outputs: `/dataMeR1/phil/gfm/mixture-scaling/results/gradient_diagnostic_20260912/`.


## Literature context (2026-09-12)
These local model-specific diagnostics are consistent with existing multi-task optimization/generalization findings. They do not establish a new general mechanism. The next plan prioritizes a supervised-to-distillation loss switch after per-source convergence, distinct from our BCE-plus-ranking penalty. See /tmp/mlp-pair-error-code/docs/asynchronous_convergence_plan.md for primary references, matched-loss checks, and validation/test separation.
