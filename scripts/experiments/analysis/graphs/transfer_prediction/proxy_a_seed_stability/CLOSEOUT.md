# Chat and worktree closeout — 2026-09-12

## Preservation

All experiment code, 4k/40k results, findings, derived tables, figures and the
incremental fixed-effects/graph-holdout analysis were committed and remotely
verified through `f42bc9c8` on `codex/proxy-a-seeds-clean-20260909`.
This closeout additionally preserves the original historical scatterplot and its
portable plotting script, the archive manifest/verification receipt, and the
conversation synthesis below. The branch is the recovery point; this closeout
makes no claim that the branch has been merged into main.

The original local `/Users/philipp/projects/gfm/prodigy-proxy-a-seeds` worktree
was already absent when checked. A temporary `/private/tmp/proxy-a-closeout`
worktree was used for this closeout. Main had an unrelated in-progress merge and
was not edited or resolved here.

All 120 files from Tucker's experiment `log/` have been moved intact to:

`/dataMeR1/phil/gfm/experiment_archives/proxy_a_seed_stability_20260912/log/`

This includes 108 sampled-ID/feature NPZ caches, both completed runs' raw JSON,
protocols and DONE markers, all console logs and the failed initial 40k launch
protocol. SHA-256 hashes were recorded before moving and all verified afterward.
The eight result/provenance files already committed match the Tucker originals
byte-for-byte. See `data/closeout/receipt.json`, `SHA256SUMS`, `verification.txt`.
Caches are preserved on Tucker rather than committed to Git (4 GB total archive).
Older paths in run READMEs and protocols describe historical locations; the
archive above is now authoritative. Runner revision remains `020c7f39`.

No active tmux session or relevant process remained. The intended cleanup removes
only `/dataMeR1/phil/gfm/prodigy-proxy-a-seeds` plus this temporary local worktree;
main and other experiment worktrees/branches are retained.

## Interpretation saved from the conversation

Raw-feature proxy-A is a repeatable predictor of NM-to-NM transfer. The 40k
uniform repeats preserve the negative association (within-target Spearman
−0.725/−0.706/−0.722) and the 4k-versus-40k pair ranking (0.995), while reducing
mean domain-accuracy SD from 0.53 to 0.21 percentage points. This establishes
stability over the tested sample sizes, not rare-region coverage or causality.
The original sorted-candidate sampler favors low IDs; uniform results are primary.

Beyond source and target fixed effects, residual Pearson r is −0.576, but removing
just Election↔COVID Political's two directed cells reduces it to −0.196. Graph
holdout improves average donor ranking and centered prediction error, not best-
donor regret. Similarity therefore provides uneven incremental compatibility
information rather than an established universal explanation of source quality.

The query/episode analysis synthesis discussed here is a hypothesis, not a new
experiment: target episodes supply relative true-versus-rival neighborhood
separation; source-trained encoders preserve/use that evidence differently;
learned support processing then converts it into class decisions. Proxy-A may be
a coarse indicator of compatibility with these discrimination problems.

The evidence cited on September 9 came from main's error-audit files:
`FINDINGS_NM_COMPLETE_INPUTS.md`, `FINDINGS_NM_SOURCE_STAGES.md`,
`FINDINGS_NM_EPISODE_DETAIL.md`, `FINDINGS_NM_SUPPORT_GEOMETRY.md`,
`FINDINGS_NM_HK_MECHANISM.md`, and `FINDINGS_NM_HK_GOAL.md`.
They belong to separately maintained analyses, not new outputs of this chat.
Those audits cover two targets/one checkpoint seed per source, unlike the
nine-graph/three-training-seed transfer matrix. They show source differences
before and after the metagraph, useful native readout behavior on original HK
supports, and fragility under replacement; a fixed geometry repair did not
improve the primary HK benchmark. This is a record of the discussion, not an
assertion that no subsequent evidence exists.

The proposed bridge—whether source–target proxy-A predicts preservation of
true-versus-rival separation before the metagraph on matched episodes—was not
run in this chat. No further experiment is pending in this worktree.
