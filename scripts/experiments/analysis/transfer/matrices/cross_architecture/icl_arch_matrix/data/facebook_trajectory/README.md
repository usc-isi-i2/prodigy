# PRODIGY Facebook classification trajectory

- Evaluated: 2026-08-15 18:28:32–18:29:20 PDT (2026-08-16 01:28:32–01:29:20 UTC).
- Code: `a7b59fa` on `codex/final-core-three-seed-sync`.
- Target: `facebook_page_reference`, primary `page_category_top30` labels.
- Protocol: no adaptation; 128 fixed seed-0 2-way/10-shot/4-query test episodes.
- Checkpoints: updates 20, 60, and 100 for four PRODIGY single-source models, plus one deterministic random-init update-0 control.
- Metric: episodic binary ROC-AUC. The 30-class global diagnostic is not used because each episode samples only two page categories.
- All 13 rows share episode fingerprint `df2330ab1920415c5c98c8ec00ef3e640a961ff90514a04bf5074d09529a0afe`.
- Tucker source: `/dataMeR1/phil/gfm/prodigy-archtraj/log/icl_arch_matrix/prodigy_facebook_trajectory_v4/results/`.

`prodigy.jsonl` is sorted by checkpoint and model ID; it consolidates the seven raw Tucker result files without changing their row payloads.
