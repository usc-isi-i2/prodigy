# Portable inference reproduction: complete final-checkpoint replay

7 September 2026. The extracted-package run completed in 2,965.24 seconds on
four CPU threads. Its terminal receipt and all 2,700 metric cells passed the
independent canonical-evidence validator. This is completed frozen inference,
not an end-to-end training or public-data release claim.

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
  and restoration after exceptions. Six independent analysis tests check the
  real canonical CSV/JSON loaders and reject partial, missing, duplicate,
  mismatched and nonfinite result panels or invalid numerical-error receipts.
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

## Completed full verification

Runtime revision `0054731a`, pinned detached on Tucker in
`/dataMeR1/phil/gfm/prodigy-role-topology`.
The extracted runtime and all private inputs are outside that Git checkout:
`/dataMeR1/phil/gfm/inference-reproduction-20260907/`.
The job ran in tmux `mechanism-portable-inference`; Python PID 3938570 exited
after writing `full/DONE.json`. No restart or runtime checkout mutation was
needed. All 60 model/input comparisons passed. Maximum historical logit error
and maximum canonical AUC/accuracy/F1/NLL error are both **0.0**.

The completed grid is six update-2500 checkpoints, all five targets, both streams,
32 batches per stream, 45 conditions: 2700 metric cells. Historical references
cover 2580 cells (43 conditions per model/input); 120 raw/encoder prototype
cells are additional diagnostics. The run does not repeat training or select
new interventions; it verifies numerical portability of existing results.
The compatible architecture can load other steps, but this declared validation
does not claim the complete five-step panel or historical nine-source map.

The runtime checked every input file and model tensor digest, frozen-state and
operator restoration, query invariance and historical predictions. It also
checked that imported inference modules came from the extracted package, not
the research checkout. The source library's actual sum aggregation is retained;
actual mean remains a named intervention. No production model code was fixed.

All four terminal JSONs are retained in `data/portable_inference_20260907/full/`;
`validation.json` records their hashes and the three canonical reference-file
hashes. The independent comparison uses the topology/message `cells.csv` files
and dose `metrics.json`, not regenerated references from the new package. The
first analysis invocation exposed an incorrect JSON filename assumption for
the two CSV panels; it was corrected and covered by a real-file loader test.
The frozen runtime and its outputs were unchanged.

The final receipts contain 17,280 exact pre-metagraph query checks, 26,880 exact
post-metagraph checks and 2,220 direct/factored role checks. Raw prototype
metrics also agree across all source checkpoints for each fixed input stream.
To revalidate the saved completed evidence from the worktree:

```bash
/opt/homebrew/bin/python3.11 -m scripts.experiments.analysis.graphs.transfer_prediction.target_performance_mechanisms.analyze_portable_inference --input scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/portable_inference_20260907 --output scripts/experiments/analysis/graphs/transfer_prediction/target_performance_mechanisms/data/portable_inference_20260907
```

## Remaining publication scope

No public upload or license grant is asserted. The inspected research checkout
contains no repository-level license file; upstream code licensing and derived
data rights need resolution before publication. The code package's attribution
notice preserves that distinction. Portable frozen inference is narrower than
training, graph reconstruction from raw platform records, or public access to
private model inputs. The paper's reproducibility update can now report this
completed final-checkpoint panel, without describing it as training or raw-graph
reproduction. The original [PRODIGY repository](https://github.com/snap-stanford/prodigy)
was also inspected on 7 September 2026; its visible root listing supplied no
repository-level license file. This observation is not a legal determination or
a basis for inventing a license grant.

Branch: `codex/role-topology-interactions`.
Local worktree: `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
Build/run instructions: setup's `portable_inference/README.md`.
