# MLP ladder and feature diagnostics — 2026-09-11

## Data-quality finding changes interpretation of earlier results

The canonical Suspended graph used in the recorded runs is corrupted by CSV parsing. Two unquoted multiline bios split into five malformed-width records. Python CSV/pandas Python read 56,443 fragments; pandas C read 72,295 rows. Rejoining the two bios produces 56,440 valid rows, matching the original graph nodes and user-ID array. The apparent 15,853-node identical embedding group originated in this parsing failure, not genuine repeated bios.

A separate corrected candidate was built on Tucker at `/dataMeR1/phil/gfm/mixture-scaling/results/suspended_csv_repair_20260911_v2/graphs/retweet_graph_suspended.pt`. Both CSV parsers agree after repair. All 354,209 edges are exactly preserved, endpoints lie in 0..56439, and all features are finite. Content hashes reuse existing embeddings except two repaired texts re-encoded with the original GTE revision/settings. 41,016 original-range feature rows change. The largest remaining nonzero duplicate group is 81; 12,283 corrected nodes have zero features. Original artifacts were not replaced and the candidate was not promoted into the catalog.

**Historical Suspended evaluations, any model trained using the corrupted graph, and derived aggregate scores are suspect and require rerunning.** In particular, ladder rungs 6–9 include Suspended. The earlier activation findings describe an affected rung-9 checkpoint. The plots after repair use corrected Suspended features only where explicitly labeled; no model was retrained on the repair in this task.

## Historical MLP ladder

Nine independently initialized cumulative-source models, seed 0; shared node MLP 768→256→256 with ReLU, dropout 0.0, 262,656 parameters. Source-confined observed edges versus five uniform approximate negatives per positive; 1,024 positive edges per update; dot-product logits and BCE. Sources rotate uniformly. Validation uses fixed held-out-edge batches. Downstream evaluation instead uses cosine scores and degree-matched negatives, so BCE and table AUC are not the same scoring protocol.

Source order: UKR/RUS Twitter, COVID Twitter, Midterm, COVID Political, Election 2020, Suspended, TwiBot20, CP/HK, Facebook. Rungs 1–4 and 6 met the plateau criterion; 5 and 7–9 hit safety limits. Final steps: 69k, 171k, 294k, 288k, 500k, 528k, 700k, 800k, 900k. Queued/recovered rungs 4–6 validated every 2,000 updates/source with patience 3; other runs used 500 with patience 10. Rung 6 restarted after a scheduler handoff failure; the interrupted attempt was preserved separately. Zombie-worker detection was corrected in commit 82ab7f5.

All 81 nine-target evaluations completed, but remain subject to the corruption caveat above. Nine completed histories and 81 AUC summaries were synced and read back from `https://wandb.ai/eibl-usc/node-mlp-ladder`. Historical mean AUC rose from 64.84% at rung 1 to 69.78% at rung 9; this is **not a clean corrected-data conclusion**. The older 2,500-update figures are preserved under `historical_fixed_budget/` and are not convergence runs.

## Training throughput

Real Ukraine/COVID benchmark: 99.44 updates/s synchronous versus 269.25 at prefetch depth 8 (2.71×), identical final weights. A brief later rung-9 replay spent 37.4% in next-batch calls, 51.9% in forward/backward/clipping/optimizer calls, and 8.1% in finite/scalar synchronization. These are host-call timings including CUDA waits, not pure GPU kernel times. Live disk reads were zero. See `fast_worker_profile/`; the short replay throughput excludes validation/W&B and is not a live-run estimate.

## Layer inspection (historical rung 9, best step 859,500)

Final 256×256 weight matrix: std 2.447, range −17.80..20.78, top 10 singular directions carry 96.2% squared weight magnitude. This alone does not show embedding collapse. First 256×768 layer: std 0.433, mean bias −0.410, top 10 weight directions 33.0% energy. Across 4,096 sampled nodes/graph, 97 hidden units never activate anywhere in the combined samples; 96–98% of ReLU activations are zero. These samples included zero-feature nodes.

Normalized output effective ranks span approximately 18–43. Midterm changes from raw rank 4.3 to normalized rank 26.3, showing substantial norm-outlier influence. Effective rank is entropy of normalized eigenvalues of centered covariance, not algebraic rank or a performance metric. These measurements describe a model trained with the corrupted Suspended data.

## Small-graph activation screening

COVID Political only, 20k updates, seed 0, identical initialization and batches; LeakyReLU slope 0.01. Both best checkpoints selected at 18k using validation BCE. Combined training took 67 seconds. ReLU versus LeakyReLU: validation BCE 0.620085 vs 0.616560; source AUC 78.13% vs 78.59%; normalized embedding rank 14.9 vs 18.2. Exact zero activation elimination is expected by construction, not itself evidence of improvement.

Mean AUC across eight other graphs: 56.39% vs 56.23% (−0.16 points). Four targets improved and four declined. This single-seed short experiment does not support a general transfer gain. Its Suspended target evaluation used the corrupted graph and must be replaced; other target evaluations and source-only training do not use that graph.

## Degree and feature distributions

Saved degree matrices are in/out KS, not KL. They come from PRODIGY's `scripts/experiments/analysis/graphs/structure/graph_divergence/data/graph_divergence_data.json`. Ordered pairs were independently sampled, producing small asymmetries. Degree CCDFs exclude zero-degree nodes and retain up to 200 points; connecting lines interpolate. Facebook's structural view differs from the MLP graph artifact. These historical degree comparisons were not recomputed after the Suspended repair.

First-coordinate histograms initially included zeros, then excluded entire zero vectors. The original sample all-zero fractions were UKR/RUS 23.23%, COVID 19.47%, Midterm 19.65%, COVID Political 0%, Election 0%, Suspended 16.99%, TwiBot20 11.53%, CP/HK 23.11%, Facebook 0.32%. Samples used up to 100k nodes, seed 2026. The corrected overlay uses all 44,157 nonzero Suspended nodes and unchanged other samples. The nonzero spike disappears after repair. Smooth first-coordinate figures use Gaussian bandwidth 0.003.

## Coordinatewise distribution ranking (corrected data)

20,000 nonzero nodes per graph, seed 2026; corrected Suspended candidate. Each coordinate gets a mean empirical KS over 36 graph pairs, or 28 pairs excluding Facebook. All 768 coordinates were scored; indices are zero-based. Direct empirical CDF computation avoids unused p-values and was checked against SciPy for coordinate 0 of every pair. Rankings are descriptive; no multiple-testing significance or task importance is claimed.

Top all-nine coordinates: 670 (0.251949), 179 (0.195211), 147 (0.191511), 359 (0.185503), 656 (0.184738). Top without Facebook: 670 (0.199018), 656 (0.193848), 119 (0.188698), 194 (0.188336), 147 (0.188311). Median scores: 0.076946 and 0.069927 respectively. Coordinate 0 scores 0.082874 and 0.051241. Density plots are smoothed; KS uses original unsmoothed samples. Sorted, index-order line, and index-order scatter plots are retained. Distribution shift is separate from task relevance. Matryoshka support of the exact GTE checkpoint was not verified.

## Reproduction and artifacts

Figures are in `figures/`; supporting JSON/JSONL is gzip-compressed under `data/` where large. `data/cluster_reports.json` preserves exact CSV results and JSON summaries as strings. `node_mlp_ladder_s0_convergence/` contains earlier finished-rung curves and data. `manifest.json` lists file sizes and SHA-256 hashes.

Relevant code branches: `codex/mlp-fast-scheduler`, `codex/mlp-profile`, `codex/mlp-overlap-eval`, `codex/mlp-activation-inspect`, `codex/mlp-leaky-test`, `codex/mlp-feature-hist`, `codex/suspended-csv-repair`, `codex/coordinate-ks`. Cluster scripts live in the corresponding isolated mixture-scaling worktrees. Full coordinate samples remain in `/dataMeR1/phil/gfm/mixture-scaling/results/coordinate_ks_nonzero_corrected_s2026/samples.npz`; they are not committed. No model checkpoints or raw bio text are included here.

This archive was assembled in `/tmp/mlp-session-findings`, branch `codex/mlp-session-findings`, without staging unrelated worktrees' changes.
