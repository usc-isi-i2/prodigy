# Updated per-coordinate feature KS — September 11, 2026

Current result: `data/feature_ks_political_updated.json`; figure: `figures/feature_ks_matrix_political_updated.png`. This supersedes the Suspended-only updated matrix for COVID Political comparisons. Both historical matrices remain available.

COVID Political uses the rebuilt canonical graph, SHA256 `755219daede78f913ab897fec51c98f261cff797a9869d80fb7ed35ce0b4d516`. Its eight pairs were rerun. The Suspended comparison uses the verified repaired v2 graph (56,440 nodes), resampled with the same seed and procedure as the prior repaired result. All other comparisons and graph samples remain fixed from the earlier sweep, including Election 2020's original sampled features; this is not a claim that every graph was reloaded from its latest canonical artifact.

Each comparison uses 10,000 sampled nodes per graph and two-sided empirical KS per each of 768 raw coordinates. Separate all-row and nonzero-row comparisons use fixed seed 2026, excluding exactly all-zero feature vectors in the latter. Mean KS is an average of marginal distances, not multivariate divergence. Finite samples produce nonzero distances even under identical populations.

Nonzero mean KS for COVID Political, old → new: UKR/RUS 0.0915 → 0.0993; COVID 0.0972 → 0.1065; Midterm 0.0675 → 0.0589; Election 2020 0.0669 → 0.0903; Suspended 0.0808 → 0.0825; TwiBot20 0.0820 → 0.0738; CP/HK 0.1084 → 0.1194; Facebook 0.1281 → 0.1425. These are modest mixed changes, unlike the large reduction after Suspended's repair.

The earlier MLP ladder, activation diagnostics and LeakyReLU screening predate this COVID Political rebuild. Their numerical results are historical; this distribution update does not update those model results.

Reproduction: mixture-scaling branch `codex/feature-dimension-ks`, commit `053b6be`, `scripts/rerun_covid_political_ks.py`; Tucker results `/dataMeR1/phil/gfm/mixture-scaling/results/feature_dimension_ks_political_updated_s2026`. Existing `elapsed_seconds` in the cumulative JSON is inherited from the original full sweep, not timing for this partial update.

Follow-up: [joint MMD, reduced KS views and September 12 closeout](CHAT_CLOSEOUT_20260912.md).
