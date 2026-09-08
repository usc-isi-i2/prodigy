# TRACE health-guided source allocation

This experiment asks whether a target-unlabeled transfer-health signal can improve
the allocation of a fixed graph-pretraining budget.  The primary target is
`facebook_page_reference`, selected before the new runs because it has headroom,
exact inspected examples, and the clearest replicated TRACE routing gain.

Every training arm uses the same PRODIGY architecture, NM objective, four-source
pool, three seeds, and exactly 2,500 optimizer updates.  The comparison contains:

- the existing r4 uniform interleaving, replay-100, and blocked sequential arms;
- four newly trained single-source specialists per seed;
- a true naive merged arm per seed, sampling nodes proportionally from the union of
  the same four sources and allowing cross-source pseudo-classes;
- a health-guided weighted interleave per seed (stage 2).

Stage 1 trains the specialists and merged baselines.  The specialist checkpoints
are then replayed on a fixed original Facebook stream.  The schedule ranks sources
within each seed by full-model/U1 support-readout agreement, provided the target
passes a preregistered 0.55 support-LOO competence gate.  The fixed 2,500-step
allocation is 40/30/20/10 percent in rank order and is smoothly interleaved.  This
uses target inputs and episode support labels but never target query labels; it is
therefore target-unlabeled, not target-agnostic.

Run the smoke phase before the full phase.  All Tucker runs belong in the dedicated
`/dataMeR1/phil/gfm/prodigy-trace-health-guided` worktree on branch
`codex/trace-health-guided`; never update that worktree while its tmux session is
running.

`run_pipeline_tucker.sh` executes stage 1, verifies every consumed episode, derives
the target-unlabeled schedule, trains and verifies stage 2, replays the complete
27-model lattice on original and disjoint fresh streams, and writes the comparison
tables.  The table reports blocked sequential, replay-100, uniform interleaving,
naive merged, health-guided interleaving, health-selected single source,
validation-selected best single source, and a clearly marked fresh-test oracle
best single source.
