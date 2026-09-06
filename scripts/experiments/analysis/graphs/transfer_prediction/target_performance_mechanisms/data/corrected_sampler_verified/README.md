# Corrected singleton reuse: input provenance

`checkpoint_inventory.json` was produced by the isolated verifier at `311383ab`.
It covers nine sources and saved updates 100/300/900/2500. All 36 weights-only
files match their saved training-state tensors; completed and optimizer step
counts, finite common architecture, source restriction and effective training
configuration checks passed before classification replay began. Original training
revision: `edd1649e`, seed zero, 2,500 updates, four episodes per update.

`source_pipeline.txt` is the unedited log of the separate source pipeline. All
nine singleton trainings completed; the subsequent NM evaluation stopped on a
missing reference-ledger path. Pair/LOO stages did not run in that pipeline.
The archived ledger does exist under
`/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core-cache/files/log/final_core_cached_test/production/bs32/summary/episode_fingerprints.tsv`.
Pointing to it is **not** a verified repair: the corrected NM selector also
changes generated NM test episodes, so historical fingerprint compatibility
must be checked separately. We have not restarted or altered that pipeline.

Our classification replay launched 11:43:00 UTC in
`/dataMeR1/phil/gfm/prodigy-mechanisms-corrected`, revision `311383ab`, tmux
`mechanism-corrected-sampler`, four CPU threads and no visible GPUs. The complete
6,120-row replay and exact comparison of its 320 cached batches are required
before interpreting results. `DONE.json` and `input_validation.json` will be
collected only once those gates pass. The inventory alone is not an eval result.

The correction changes retention, role order and random-number consumption
jointly. This single-seed production-recipe comparison has matched test inputs,
not matched training inputs or guaranteed matched initialization. It does not
replace the dedicated three-seed factorial.
