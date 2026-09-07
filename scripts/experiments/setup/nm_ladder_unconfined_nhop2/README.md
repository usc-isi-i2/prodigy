# Unconfined two-hop NM graph ladder

This experiment adds a naive merged-source arm to the registered sequential versus
interleaved schedule comparison. It keeps the same canonical source order, seed 0,
40,000 optimizer steps per rung, 2-hop `9,9` sampling, node limit 101, one-hop NM
positive walks, 256-dimensional `S,U,M` GraphSAGE encoder, and terminal-checkpoint
evaluation on the same eight NM graphs.

The sole intended treatment difference is episode formation. These configs omit
`neighbor_sampling_episode_source`, source weighting, source subsets, and source
sequences, so support/query centers and negatives are sampled naively from each rung's
merged artifact without `graph_id` confinement.

`run_pipeline_tucker.sh` trains all eight rungs on GPUs 2 and 3, evaluates the 64
terminal cells, adds `auc_unconfined` to the registered paired table, and regenerates
`figures/interleaved_vs_sequential_mean.png` with the third line.
