# Campaign execution log

- 2026-09-04 09:19 UTC: goal activated; deadline 17:19 UTC. Dedicated local and Tucker worktrees created on `codex/nm-interventions-overnight` from fast-launcher revision `9ea87fe`.
- 09:25 UTC: source-restriction tests pass; initial campaign implementation committed and pushed through git.
- 09:27 UTC: first full-graph smoke stopped at CUDA preflight before data load. GPU 0 reports idle/default mode but rejects standalone allocation. GPUs 1–3 allocate successfully. No production training occurred.
- 09:29 UTC: second smoke launched on GPUs 1–3 in `nmi_smoke`; all 17 methods included. No test-source feedback enters stopping or selection.
- 09:33 UTC: second smoke completed all 17 configurations on GPUs 1–3. Every arm trained 20 steps, validated on eight included sources, and wrote a selected checkpoint. Strict reload/fingerprint gate follows before production.
- 09:34 UTC: strict checkpoint reload / finite-weight / eight-source fingerprint gate passed all 17 smoke arms. Production launched in `nmi_train` at revision `6260524e`, GPUs 1–3, four concurrent models/GPU, 48 loader workers. 136 configs queued with rung eight first. GPU 0 remains unavailable; no GPU reset attempted.
