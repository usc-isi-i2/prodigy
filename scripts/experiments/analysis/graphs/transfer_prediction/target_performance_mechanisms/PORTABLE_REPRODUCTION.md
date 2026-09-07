# Portable inference reproduction: code candidate complete, full replay running

7 September 2026, 03:23 UTC progress checkpoint. Do not interpret this file as
the completed full replay; require the live run's terminal receipt first.

## Completed

- A 27-file, code-only archive builds from explicitly selected model and
  intervention sources. It contains no real profiles, account IDs, feature
  tensors, checkpoints or platform-specific training launchers.
- Model source changes are import-namespace changes only; selected diagnostic
  functions are copied verbatim with minimal imports. Source hashes and symbol
  lists are included. A direct S,U,M constructor and tensor schema replace the
  catalog/training-runner dependencies, not the model computation.
- Nine synthetic tests pass from extracted archives on local Python 3.10 and
  Tucker Python 3.11. They exercise all 45 conditions, 37 direct/factored role
  checks, query-label invariance, mask/degree contracts, tensor serialization
  and restoration after exceptions. Four independent full-grid analysis tests
  reject partial, missing, duplicate, mismatched and nonfinite result panels.
- The private adapter exports all 320 fixed batches and six final checkpoints,
  with all 60 model/target/stream first-batch constructor/encoding/logit
  comparisons bit-exact to the research implementation. Account identifiers
  are replaced by local node indices. Real feature tensors remain private;
  removing account IDs is not asserted to anonymize those features.
- The extracted runtime's three-target smoke (political, pages, bot; HK seed 0;
  one batch each) matches all 129 historical condition-logit comparisons
  exactly. Its 135 metric cells include six additional prototype diagnostics.
  It is explicitly partial, not full-study reproduction.

Code archive SHA-256:
`9eab3ff790a906d037354742be2d19e1c65f3273009a8c7e3402f6a85839951b`.
Code manifest SHA-256:
`d19c2d76ca1905d3ead4d6aaee98195b31a7fd29704695e2d78151f7af4a5c77`.
Local archive: `output/inference/graph_role_inference_code.zip` in this worktree.
The compact completed receipts are `data/portable_inference_20260907/`.

## Running full verification

Runtime revision `0054731a`, pinned detached on Tucker in
`/dataMeR1/phil/gfm/prodigy-role-topology`.
The extracted runtime and all private inputs are outside that Git checkout:
`/dataMeR1/phil/gfm/inference-reproduction-20260907/`.
Tmux: `mechanism-portable-inference`; pane PID 3937440, Python PID 3938570.
The Python process was confirmed live; the latest inspected log had four full
model/input comparisons complete, all historical logits and metrics exact.

The full target is six update-2500 checkpoints, all five targets, both streams,
32 batches per stream, 45 conditions: 2700 metric cells. Historical references
cover 2580 cells (43 conditions per model/input); 120 raw/encoder prototype
cells are additional diagnostics. The run does not repeat training or select
new interventions; it verifies numerical portability of existing results.
The compatible architecture can load other steps, but this declared validation
does not claim the complete five-step panel or historical nine-source map.

The runtime checks every input file and model tensor digest, frozen-state and
operator restoration, query invariance and historical predictions. It also
checks that imported inference modules come from the extracted package, not
the research checkout. The source library's actual sum aggregation is retained;
actual mean remains a named intervention. No production model code was fixed.

Do not restart on an observation timeout, and do not checkout/pull in the pinned
worktree while its job is live. Read `full/DONE.json` and the live process/logs.
After completion, retrieve `full/{protocol,DONE,metrics,receipts}.json` and
`private_inputs/export_receipt.json`, then run `analyze_portable_inference`
as a module. It independently compares all 2580 historical metric cells to
the canonical topology/message/dose evidence and checks complete coverage.

## Remaining publication scope

No public upload or license grant is asserted. The inspected research checkout
contains no repository-level license file; upstream code licensing and derived
data rights need resolution before publication. The code package's attribution
notice preserves that distinction. Portable frozen inference is narrower than
training, graph reconstruction from raw platform records, or public access to
private model inputs. The manuscript is unchanged until the full validation
supports a precise reproducibility claim.

Branch: `codex/role-topology-interactions`.
Local worktree: `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
Build/run instructions: setup's `portable_inference/README.md`.
