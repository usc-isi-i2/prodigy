# Matched warm-start ranking-preservation pilot

Four seed-0 runs on two previously diagnosed pairs, Election excluded:

- Ukraine/Russia teacher → interleave Ukraine/Russia + Facebook pages, rank weight 0 and 1.
- Ukraine suspended teacher → interleave Ukraine suspended + COVID political, rank weight 0 and 1.

Teachers are the existing singleton best.pt checkpoints, identified as stronger in the preceding target-validation diagnostic. These are exploratory, post-hoc case studies, not a new held-out confirmation. No downstream test labels select checkpoints or tune the penalty.

Both arms load identical encoder weights, bias, and AdamW state. Source order, source-specific batch/negative generators, batch counts, learning rate, and stopping rules match. Alternate A/B updates 1:1 with a single optimizer. No fresh-from-scratch arm is launched; prior scratch interleaving remains available but differs in initialization and therefore is not the penalty control.

Penalty is applied only on source-A training batches, using a frozen copy of its starting model. For each supervised positive, pair its logit with five sampled nonedge logits. Teacher preference is sigmoid(teacher_positive_logit − teacher_negative_logit), including cases where the teacher prefers a negative. Minimize Bernoulli KL from this soft teacher preference to the corresponding student preference (temperature 1, weight 1). KL is computed as student cross-entropy minus fixed teacher entropy. Bias cancels from the score differences. Pair assignments are deterministic given the already sampled negatives. There is no extra sampling and no validation/test pair in the teacher penalty.

Loss is supervised BCE + lambda × ranking KL on A, supervised BCE on B. Log supervised BCE separately from the penalty. This is a soft constraint, not a guarantee of unchanged rankings. Lambda 1 is a prespecified pilot setting; report both arms, do not select lambda from downstream tests.

Validation and selection: mean(A BCE, B BCE), every 2,000 total updates, minimum absolute improvement 1e-4, patience 3 after 2,500 updates, 100,000 total-update cap. Save and evaluate step 0 as an eligible best checkpoint. Both arms receive identical step-0 handling. No validation teacher loss is used for selection. Report source retention and held-out transfer separately; caps are not convergence and the two arms may take different numbers of updates.

All four selected checkpoints are evaluated on eight original target pair sets (32 cells), with six non-training targets per pair for transfer summaries. Optimizer state and global RNG are saved; exact sampler resume remains unsupported. Partial runs are refused and previous runs remain intact.

Launch on owned GPUs 0–3 from an isolated Tucker worktree:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_ranking_preservation.sh /dataMeR1/phil/gfm/mixture-scaling/state/ranking_preservation_s0
```

Tests cover shift-invariant ranking loss, zero loss/gradient when student matches teacher, frozen teacher gradients, matched plan, preserved AdamW counters, initial checkpoint eligibility, and existing interleaving and negative-sampling behavior.
