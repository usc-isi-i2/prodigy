# Canonical NM support resampling

Bounded, outcome-conditioned diagnostic, not a deployable support selector.
Use the two seed-0 step-2500 specialists and the canonical test episode plans.
Reconstruct all 512 episodes per target and require the original complete input
tensor hash. Select 100 native-model failed single-anchor query occurrences and
100 correct controls, matched exactly in incident-degree and query-frequency
bins. Every selected query identity is distinct within a target.

Each selected anchor must have at least three alternative static-test neighbors
outside all episode query identities, the original support triple and the anchor
itself. Sample five uniform triples without replacement within each draw. Verify
every replacement pair against all three edge splits. This intentionally changes
the historical lowest-ID support-selection policy; it is conditional on sufficient
alternative neighbors and not representative of every NM occurrence.

For the paired context control, retain the original three support IDs but draw
fresh static-train neighborhoods using the same per-slot seeds as the replacement
condition. Both models consume the same saved support draws. Only the selected
class's three pre-metagraph rows change; all query rows, other supports, candidate
classes and the ORIGINAL within-batch learned label vectors are retained.

Full-forward baseline parity, manual-encoder parity, selected historical decisions,
and whole-batch witnesses for both intervention conditions gate the run. Numerical
logit tolerances are fixed at 1e-5 for manual encoding and 1e-4 for differently
batched suffix/full comparisons, with identical selected baseline decisions and
witness decisions required. Immutable input hashes and model-state hashes are
rechecked after each model. Context-only graph change rates verify manipulation.

Output reports average correctness across all five draws, any/all-draw correctness,
rescue and break rates conditional on each model's baseline, and true-class rank
changes. The failed/correct case split is selected on the NATIVE model and stays
fixed for the foreign model. Any-draw success is oracle headroom, not measured
algorithmic gain. Tests condition replacements on the known true anchor, so they
cannot be used to claim deployment without a separate label-free selection rule.

Run in an isolated Tucker worktree, using a free owned GPU and the prodigy env:

```sh
python scripts/experiments/setup/nm_support_resampling/run.py --out /dataMeR1/phil/gfm/error_audit/nm_support_resampling_20260908 --dry-run
```

Remove `--dry-run` to execute. `--cases`, `--targets`, `--device`, `--threads`,
`--audit-root`, `--bio-root` and `--seed` are overrideable. Small case runs are
labelled smoke. The output directory must be new. Large support graphs, case IDs,
and per-case predictions stay private on Tucker; import only aggregate evidence.
