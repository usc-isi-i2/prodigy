# Pure node MLP ladder and diagnostics — September 11, 2026

## Artifact warning and current default

**Historical results below used the defective Suspended artifact.** It was replaced at its existing canonical path on September 11. Future direct loads use the corrected graph automatically; see [repair note](../../../../../../../docs/ukraine_suspended_repair_20260911.md). The repair is not a corrected training rerun: rungs 6–9 and all Suspended-target results below must not be presented as clean corrected-artifact results. Merged artifacts and cached full graphs require independent rebuilding. Completed histories are preserved for provenance.

## What was trained

Nine independently initialized seed-0 models on cumulative source prefixes in the table order. This is **link prediction**, not feature reconstruction and not PRODIGY episodic neighbor matching. Node features enter Linear(768,256), ReLU, Dropout(0), Linear(256,256): 262,656 parameters, no message passing. Dot-product pair logits and mean BCE-with-logits; each update samples 1,024 positive edges and five source-confined uniform negatives per positive, excluding self-loops (not guaranteed to exclude all existing edges). Uniform round-robin source selection. AdamW lr 0.0005, weight decay 1e-5, gradient clipping 1.

The initial 2,500-update ladder was a short budget, not convergence. The later runs used minimum 2,500 updates/source, min_delta 1e-4, and safety limit 100,000 updates/source. Rungs 1–3 and 7–9 validated every 500 updates/source with patience 10; rungs 4–6 used 2,000 with patience 3. Stop only when every source stalls; select the checkpoint with lowest macro source-validation BCE. Rungs 1,2,3,4,6 plateaued; 5,7,8,9 hit the safety cap. Final updates: 69k,171k,294k,288k,500k,528k,700k,800k,900k. This is not an equal-budget or uniformly converged comparison.

## Evaluation

All nine targets, 81 cells. Canonical held-out edge partitions, degree-matched negative pairs, cosine scoring, ROC-AUC; leakage and endpoint-sensitivity gates passed. Checkpoints selected using source validation only. Training/validation uses dot products and uniform negatives, whereas evaluation uses cosine and degree-matched negatives; lower BCE need not imply higher evaluation AUC. Initially only six targets were evaluated; COVID Political, Election 2020 and Suspended were subsequently added.

AUC in percent. Each row adds its source to all earlier sources; † denotes safety cap.

| Source added | ukr_rus_twitter | covid19_twitter | midterm | covid_political | election2020 | ukr_rus_suspended | twibot20 | cp_hk_twitter | facebook_page_reference | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ukr_rus_twitter | 67.28 | 74.11 | 64.34 | 59.88 | 61.85 | 55.45 | 65.26 | 55.70 | 79.65 | 64.84 |
| covid19_twitter | 67.23 | 77.58 | 64.70 | 59.56 | 60.34 | 54.82 | 65.74 | 57.27 | 79.81 | 65.23 |
| midterm | 67.66 | 75.67 | 69.21 | 64.73 | 66.38 | 54.58 | 66.65 | 57.58 | 79.82 | 66.92 |
| covid_political | 67.18 | 75.68 | 69.63 | 77.07 | 69.13 | 55.27 | 66.80 | 57.02 | 77.53 | 68.37 |
| election2020 † | 66.48 | 74.91 | 68.88 | 76.29 | 75.57 | 54.97 | 65.93 | 56.33 | 76.96 | 68.48 |
| ukr_rus_suspended | 66.12 | 74.89 | 68.90 | 76.48 | 73.87 | 57.61 | 65.04 | 57.42 | 76.41 | 68.53 |
| twibot20 † | 66.31 | 75.08 | 68.16 | 77.44 | 75.42 | 57.66 | 72.39 | 56.72 | 77.07 | 69.58 |
| cp_hk_twitter † | 65.89 | 73.40 | 67.62 | 75.67 | 74.46 | 57.22 | 72.71 | 57.02 | 75.78 | 68.86 |
| facebook_page_reference † | 64.60 | 73.95 | 67.62 | 74.42 | 73.54 | 57.86 | 71.59 | 57.48 | 86.99 | 69.78 |

Mean rises from 64.84 to 69.78, but source-specific gains differ and sources join training across rows. This is not a single-source transfer matrix.

## Performance and execution lessons

Serial sampling plus bounded ordered prefetch (depth 8, four prep threads), GPU-resident features where feasible, pinned CPU batches and CUDA transfer overlap preserve sampling order. A matched 500-update Ukraine/COVID benchmark improved 99.4 to 269.2 updates/s (2.71×), with identical final weights. Four-GPU shared rung scheduling replaces static two-GPU allocation; owned GPUs are 0–3 only.

A short real-data rung-9 replay attributed ~37% wall time to next-batch calls, ~52% to model/optimizer host calls, and ~8% to scalar/synchronization calls. These include dispatch and waits, not pure GPU kernel time. Live disk reads were zero; CPU allocation/staging remained material. Caching validation batches and reusing buffers are candidates, not implemented findings.

A boundary-handoff bug mistook zombie workers for running processes and aborted the controller. Fixed by checking process state. Rung 6's interrupted work was preserved and restarted from seed in the recovery directory because checkpoints lack exact sampler resume state. Rungs 4–6 completed under the new validation regime. Evaluation overlapped completed rungs on an idle GPU, then all 81 results were aggregated.

## Activation diagnostics (old-artifact rung 9)

Selected checkpoint 859,500. Final-layer weight standard deviation 2.447, range −17.80 to 20.78; top ten singular directions account for 96.2% of squared weight magnitude. Weight concentration alone does not establish representation collapse.

On 4,096 uniformly sampled nodes/graph, hidden ReLU activations were 96–98% zero; 97/256 hidden units never activated across all samples. This is sampled inactivity, not proof of global dead units. First-layer weight std 0.433, mean bias −0.410, top-ten weight energy 33.0%. Output norm outliers distorted raw effective rank: Midterm 4.3 becomes 26.3 after unit normalization; normalized output ranks span 17.6–42.7. Effective rank is exp(entropy of centered covariance eigenvalue proportions), not a count of nonzero coordinates. Higher rank is not automatically better.

See figures for per-source training/validation curves and first/final-layer diagnostics. Training curves use a 5,000-update weighted moving mean; validation is unsmoothed.

## LeakyReLU screening

COVID Political only, seed 0, 20k updates, identical initialization and exact same batches for both arms; negative slope 0.01. Both selected 18k. Combined training took 67 seconds. Best BCE improved 0.62009 → 0.61656; own-graph AUC 78.13 → 78.59; normalized output effective rank 14.9 → 18.2. Zero activations falling to zero is expected mechanically and is not independent evidence of improvement.

Across the eight foreign graphs, mean AUC was 56.39 (ReLU) versus 56.23 (LeakyReLU), change −0.16 points. Thus the single-seed screening does not justify switching the full ladder. That cross-graph sweep also predates the Suspended repair. Full per-target scores are in data/leaky_cross_graph.json.

## Graph-distribution analysis

Saved degree comparisons are KS, not KL; separate in/out distributions, with sampling noise causing small asymmetries in the old matrix. Saved CCDFs exclude zero-degree nodes and retain up to 200 points. The degree study uses Facebook's structural graph, unlike this MLP's full page_reference_graph.pt.

New feature comparison: 36 unordered pairs × 768 coordinates, 10k nodes per graph, raw samples and exactly-all-zero-row-excluded samples, seed 2026. No coordinate normalization. These are marginal distributions, not feature correlations. After repairing Suspended, mean KS across its eight nonzero comparisons falls 0.2184 → 0.0732. UKR/RUS versus COVID remains closest (0.02275). All other graph samples stayed fixed. Corrected matrix and raw values are saved in the graph_divergence analysis data/ and figures/ folders. Node population changed, so repair comparison is not matched-node.

Existing GNN/NM transfer study (not this MLP ladder): raw-feature proxy-A mean within-target Spearman −0.738 versus node+neighbor −0.660; node+neighbor has lower donor-selection regret (0.80 versus 1.41 AUC points). In/out-degree KS are weaker (−0.434/−0.175). Those historical predictors also require artifact-provenance review before treating Suspended as corrected.

## Reproduction and locations

Training/evaluation implementation lives in the sibling mixture-scaling repository, not PRODIGY. Relevant branches: codex/mlp-ladder, codex/mlp-fast-scheduler, codex/mlp-profile, codex/mlp-overlap-eval, codex/mlp-activation-inspect, codex/mlp-leaky-test, codex/feature-dimension-ks. These implementation changes were committed and pushed during the conversation. Repair construction is in the mixture-scaling-suspended-repair worktree.

Tucker root `/dataMeR1/phil/gfm/mixture-scaling`: historical state directories `node_mlp_ladder_s0_convergence`, `_fast`, `_recovery`; final table `results/node_mlp_ladder_s0_convergence_recovery/raw/ladder_results.csv`; activation comparison `results/leaky_covid_political_s0_20k`; corrected KS `results/feature_dimension_ks_suspended_v2_s2026`. The recovery root imports completed earlier rungs through symlinks.

W&B: https://wandb.ai/eibl-usc/node-mlp-ladder — nine completed convergence runs synced, plus all 81 verified target AUCs under eval/<target>/roc_auc. Runs 1–9: 0ro6nolv, n640r8x9, x5wpxcky, jteg0go1, 1juupszu, nu64rjfk, 6nxf9yhg, pa7rh497, gdl3i40x. Offline remains the default. Interrupted rung-6 attempt was not presented as a completed run. Uploading the later LeakyReLU/KS diagnostics was not part of that sync.
