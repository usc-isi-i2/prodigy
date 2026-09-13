# Experiment overview conversation — closeout, 12 September 2026

## Work preserved

This conversation used the local `main` checkout at
`/Users/philipp/projects/gfm/prodigy`. It created no local experiment worktree,
Tucker worktree, training job, checkpoint, or new evaluation dataset.

- [June–September hierarchical overview](../../EXPERIMENT_OVERVIEW.md), including
  archived June studies, sibling projects and branch-only evidence.
- [Corrected historical synthesis](PROGRAM_FINDINGS.md): invalid LP numbers and
  MIX-synergy conclusions replaced or excluded; unresolved comparisons explicit.
- Entry links in the analysis and synthesis indexes.
- The linked [HK LP baseline record](../../evaluation/lp_baselines/hk_lp_baselines/FINDINGS.md),
  its two JSON evidence files and verification script copied byte-for-byte from
  retained branch `codex/hk-lp-baselines-20260908`; these are historical results,
  not a new run or a newly repeated validation.

The overview's temporary source links were repaired to main-tree evidence,
immutable remote commits, or surviving sibling-project records. The research
inventory remains dated September 11; later cleanup imports are not silently
counted as part of that original inventory.

## Paper assessment from the conversation

The judgment was that there is sufficient material for a focused empirical paper,
not yet a demonstrated broadly superior method or a general scaling law. The
recommended first paper is source composition versus native/downstream transfer;
the representation-versus-inference mechanism study is a separate, more ambitious
candidate. Nonmonotonic data scaling alone is not novel. Best-constituent comparisons
use a target-informed oracle and need practical donor baselines. The main remaining
submission work is a frozen central claim/protocol, a predeclared multi-seed check of
the decisive comparison, and auditable corrected evidence. This is a research
assessment, not a prediction of acceptance or authorization for new runs.

## Worktree safety

At audit, both local and Tucker Git registries contained main and only one linked
worktree: `codex/pilot-transfer-selection` (revision `b461cb38`). Tucker had an
active `pilot-transfer-graph-recovery` tmux session. This is not a worktree created
or used for experiments by this conversation. Neither it nor main was deleted.

Other cleanup and preservation operations are recorded separately in
`cleanup_records/` and the analysis archive. This closeout does not claim to have
revalidated every historical checkpoint or ignored artifact. Unrelated dirty files
in main were intentionally left untouched.
