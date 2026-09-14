# Negative renewal: mechanism partly repaired, selected performance unchanged

Completed2026-09-14. All6 frozen cells complete; no test scoring or earlier-fold expansion. Prior repeated2019 exploration remains disclosed. No leaderboard-accepted win established.

| Seed | Fixed best2018 Hits@50 | Renewal best2018 Hits@50 | Delta pp | Steps fixed/renewal |
|---|---:|---:|---:|---:|
|0|68.715465%|68.868584%|+0.153119|150/200|
|1|69.025032%|68.906864%|−0.118168|100/200|
|2|68.931829%|68.673857%|−0.257972|100/100|
|Mean|68.890775%|68.816435%|−0.074340|—|

The frozen gate required all3 paired gains positive and mean gain>=0.5 percentage point. It failed. This exact renewal recipe is not an improvement over the validation-selected standalone control. No retuning or test evaluation followed.

Renewal does materially reduce late negative-panel overfitting. At2000updates, same-positive independent-probe Hits@50 improves for every seed (62.56→78.01%,62.80→80.44%,68.13→79.48%). Final2018 scores also improve for every seed, but remain below early checkpoint peaks. This distinguishes a real diagnostic mechanism repair from a useful selected-performance gain. Fixed-year probe scores use training positives and a99,981-pair diagnostic negative pool, not benchmark results. The pool is disjoint from the union of optimized pools;19 overlaps were masked using training-only identities before training. Cutoffs are within-checkpoint diagnostics and not calibrated across models.

## Contract and evidence

Unchanged496,385 neural weights, frozen14-input feature standardization and2015calibration, Adam.001/weight decay.0001/clip5, BCE,2048positive+1024uniform+1024hard samples, top2048 mining every10updates. Initial negative pool matches the historical fresh control;39 independently generated100k pools replace it at updates51,101,...1951. Renewal seeds2026091401..1439, separate probe seed2026091440. Same-year positives and2018validation negatives excluded; no test identities used. Positive/uniform/hard-slot RNG streams match within seed; actual hard-negative identities necessarily differ.

Controls were rerun, not compared against the previous fusion-selected checkpoints. All120 control validation history entries exactly reproduce archived full standalone histories. Root's independent audit verified hashes,240completed evaluation rows, earliest-maximum selection, per-seed sampling streams, sort-based Hits parity, and selected-score recovery/loss counts. Training verified official OGB evaluator parity and bitwise checkpoint replay. Exact100k feature replay and pre2017 adjacency equality establish shared-preprocessing parity. Original prepared input hashes and pinned AA source/revision were verified. Forty cached panels have checksummed loader verification. Conservative inference state496,448scalars includes the neural network and existing small fitted state; well below1M.

No operational failures or outcome-dependent changes occurred. Audit hardening before launch strengthened full-panel parity, pool checksums, and probe disjointness. This was one frozen run at e7d696ac83a4325e50805dee11a675033e9535ea.

## Runtime and reproduction

One-time reusable feature preparation853.02seconds. Six substantive training cells368.67seconds total,57.7–71.6seconds each; GPU0 roughly4GB and94% measured utilization. Offline W&B for preparation and each training cell; run locations preserved in data/*results.json. Source branch codex/collab-negative-renewal, local work/h1; dedicated Tucker worktree /dataMeR1/phil/gfm/prodigy-negative-renewal; tmux collab-negative-renewal exited after completion. Runtime cache, checkpoints, and diagnostics retained at /dataMeR1/phil/gfm/ogbl_collab_compact_joint/negative_renewal_v1.

Reproduce with scripts/experiments/setup/ogbl_collab_negative_renewal/launch.sh using a new output root and the pinned inputs. Canonical machine-readable results: data/summary.json; independent audit:data/independent_audit.json; protocol/prepared/feature_parity/pools and six complete histories/results retained alongside them. The experiment supports rejecting renewal every50updates as a standalone route forward; it does not establish that every possible renewal schedule fails.
