# HK NM mechanism and bounded repair

Authorized overnight goal, 8 September 2026. HK→HK NM first; source controls
remain part of interpretation. No classification work. Existing jobs, including
the queued flagship launcher waiting for all GPUs, have priority. Begin with
two CPU threads and no CUDA allocation. Use only an idle owned GPU (0–3) after
the priority queue clears; never alter or stop another process.

## Frozen first question

On the existing 200 selected cases, compare the original and nearest-support
endpoints using the same cached HK embeddings. Substitute only the three
changed supports' actual metagraph keys, values, or both. Retain every other
row/block, the original native decoder, and evaluation normalization. All Q
blocks remain unchanged. Query labels select metrics, never the forward pass.

The score-only analysis has already shown that true-class score movement is
sufficient to reproduce all 200 nearest-support correctness outcomes. The next
question is whether that movement follows changed attention, changed values,
or their interaction. No K/V outcome from these NM inputs has yet been seen.

Required checks: exact unchanged query outputs; joint K/V matches the full
replacement's final label/query outputs (1e-5 tensor, 1e-4 logit tolerances);
value-only attention identical to original; keys-only attention identical to
replacement; original/replacement predictions and probabilities match cached
references (1e-5 probability, 1e-4 tied-winner logit tolerances). Count numerical
differences, do not silently loosen gates. Preserve weights, inputs and donors.

Report both failed cases and correct controls. Five random trials remain
correlated draws if used later. K/V results on prior classification experiments
are not NM evidence. The 71 persistent cases are a diagnostic subset, not a
benchmark denominator.

After this first result, record the repair rationale and its fixed evaluation
protocol before testing it. Prefer a small, source-trained or source-validation
selected change; never fit to the known canonical test labels. Include unchanged
full canonical accuracy, recoveries and breakages, and source controls. A failed
repair remains a valid result of a bounded test; it must not be reported as a
successful remedy.

Initial setup estimate: 30–60 minutes. CPU compute estimate: 5–15 minutes.
New graph loading, sampling and encoding are unnecessary for this first stage.

Runtime branch: `codex/nm-hk-goal-20260908`; local worktree
`/private/tmp/prodigy-nm-hk-goal`. Transfer source via private git and use a new
Tucker worktree. All paths, methods and thread counts must be explicit/overridable.
