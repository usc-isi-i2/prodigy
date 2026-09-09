# Proxy-A seed stability

See FINDINGS.md for results and figures/proxy_a_seed_stability.png for the scatterplot.
Run `python scripts/experiments/analysis/graphs/transfer_prediction/proxy_a_seed_stability/analyze.py` from the repo root to reproduce summaries and figures.

Execution: Tucker worktree `/dataMeR1/phil/gfm/prodigy-proxy-a-seeds`, branch
`codex/proxy-a-seeds-clean-20260909`, runner revision `020c7f39`.
CPU only, four threads, tmux `proxy-a-seeds`. All 216 fits completed with no
convergence warnings. Small result/provenance files were downloaded from
`log/proxy_a_seed_stability/`; sampled node IDs and feature caches remain there.
Local analysis worktree: `/Users/philipp/projects/gfm/prodigy-proxy-a-seeds`.

Sampling regression check passed in the local prodigy environment. Analysis
validated all six unique seed/policy matrices, symmetry, finite values, and
complete fixed NM outcome alignment. Plot visually checked. Historical results
were not overwritten. The historical sample-selection implementation is reused
as a sensitivity control; its sorting/truncation creates node-ID selection bias.
