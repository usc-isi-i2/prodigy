# Lower continuation learning rate
Four seed-0 runs: Ukraine–Russia Twitter + Facebook pages and suspended Ukraine–Russia + COVID political, each with ranking KL weight 0 or 1. Same singleton initialization as the ranking-preservation pilot, with AdamW moments and counters restored. Override only the learning rate after loading optimizer state: 0.0005 to 0.00005.

Keep alternating equal updates, fixed graph/context caches, negative sampling, validation every 2,000 updates, patience 3, minimum 2,500 steps, min_delta 0.0001, and the 100,000-update safety cap. Select minimum mean source validation BCE, including step 0. Track source validation AUC too. A safety-cap exit is not convergence.

Launch from an isolated Tucker worktree:
bash scripts/run_ranking_preservation_low_lr.sh /dataMeR1/phil/gfm/mixture-scaling/state/ranking_preservation_low_lr_s0

Uses GPUs 0–3 and automatically evaluates all four selected models on the eight non-Election targets (32 cells). Compare against ranking_preservation_s0 at the original learning rate. This is a single-seed learning-rate ablation, not evidence of robustness across initialization seeds.
