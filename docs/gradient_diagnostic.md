# Gradient and disposable-update diagnostic

Two post-hoc diagnostic pairs from the ranking-preservation pilot, seed 0: Ukraine/Russia (A) + Facebook pages (B), and Ukraine suspended (A) + COVID political (B). Use original A singleton checkpoint (continuation step 0), and selected-best/final checkpoints from rank0 and rank1. These are the saved states available; intermediate periodic checkpoints were overwritten during the pilot. Checkpoint step on the singleton is its original pretraining step, separately recorded from continuation step 0.

At each state, draw eight independent source-local training batches (up to 1,024 positives without replacement, with uniform 1:5 exact-nonedge negatives). Trial samples match across states. Use the exact original deterministic source-validation sampling (up to 20×1,024 positive edges per source, seed 0). No downstream tests are read.

Compute supervised BCE gradients for A/B and aggregate validation BCE gradients over each full fixed validation sample. Record norms, dot products, and cosine similarity for all parameters, first linear, second linear, and scalar decoder bias. Scalar cosine is only a sign agreement. For rank1 states also compute effective A gradient with its frozen initial teacher's ranking penalty. Weight remains 1.

For each batch, restore a disposable copy of weights AND AdamW state, apply one clipped A-task or B-task update, and measure both validation BCE and AUC changes. At rank1 states also apply A-with-ranking updates. Once per state apply a zero-current-gradient AdamW update (momentum and weight decay remain active). This control exposes the contribution of stored optimizer state. No action is cumulative. Also record first-order validation change g_validation·parameter_delta and actual update norm. Positive BCE change is harmful; positive AUC change is helpful. Gradients use evaluation mode; this model has no dropout.

This is a local diagnostic, not a replay of historical batches (sampler state was not saved), not a training run, and not causal evidence about feature preprocessing. Repeated batches share one validation set and are not independent evaluations. Completed output contains only aggregate scalar gradient/update metrics and checkpoint receipts; checkpoints and graphs remain unchanged.

## Relation to existing work
These observations should be framed as model-specific evidence consistent with prior MTL findings, not a new general mechanism or proof that conflict never matters. See the primary references and planned matched-training-loss checks in [asynchronous_convergence_plan.md](asynchronous_convergence_plan.md). Retention on the trained graphs and transfer to graphs excluded from training remain separate outcomes.
