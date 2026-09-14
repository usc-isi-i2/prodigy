# Nonzero mini transfer matrices

Requested order: node-only feature reconstruction, node-only LP, node+neighbor LP.
The first runner is `mini_transfer` and launches only feature reconstruction.
Graphs are the nine verified nonzero views, replacing Ukraine and COVID with
one-hop seed-0 500k-node induced minis. Historical matrices are not overwritten.

FR uses the existing MaskedFeatureMLP: 768→256→256 encoder, 256→768 decoder,
learned mask token, 50% coordinate masking, squared cosine error on masked
coordinates. AdamW lr .001, weight decay 1e-5, clip 1, batch 1024, seed 0.
A seeded node permutation assigns disjoint 70% training / 15% source validation /
15% final test sets. Validation uses at most 20×1024 fixed nodes and fixed masks.
Validate every 2000 updates; stop after 3 checks without an absolute 1e-4
improvement after 2500 updates; safety limit 100k. Select the lowest source
validation loss. Cap hits are explicitly not convergence.

GPU-resident features, indexing, masks and batches avoid per-update host transfers.
New CUDA RNG streams are seeded; this is a fresh protocol, not an assertion of
bitwise equivalence with historical CPU-mask runs. Float32, TF32 off.
Four workers claim individual sources through file locks on GPUs 0–3. After all
nine finish, evaluation is sharded by target, with the same ten mask seeds per
model. Primary metric is squared cosine error (lower better); masked MSE and
cosine error are also retained. Replicate standard deviation is mask variability,
not a confidence interval across training seeds. W&B is offline by default.

Run from its dedicated Tucker worktree in tmux:
`bash scripts/run_mini_fr_tucker.sh /dataMeR1/phil/gfm/mixture-scaling-mini-transfer/state/fr_nonzero_walk1_s0`

Model state, optimizer, runtime RNG, graph identity and code revision are saved.
Exact training resume is not implemented; partial runs refuse overwrite.

## LP stages

`mini_lp` runs `--view node` then `--view node_neighbors` (fanout 10).
Both share a fresh unique-undirected nonself edge partition: 70% train, 15%
early-stopping validation, 15% final heldout. Exact CUDA negative sampling rejects
self-loops and all original edges in either direction, including both heldouts.
Positive/negative training ratio is 1:5. Features, edge lists, and known-edge
membership keys stay on GPU. Same 256-dimensional NodeMLP, dot-product score,
AdamW .0005 and prior clipping/decay/batch/early-stopping settings.
Neighbor means use a fixed sample of up to ten distinct neighbors, without
replacement, from the 70% training topology only. Isolates receive zero context.
The target graph uses its own training-only context at evaluation.

All nine targets use the canonical pair evaluator with 2000 final-heldout
positives and 2000 degree-matched negatives. The latter reject train, source
validation and final-test edges. A fixed 30% pair subset selects calibration;
raw-dot AUC/BCE are reported on the remaining 70%. Final evaluation edges never
select the source checkpoint. Predictions and metric JSONs retain provenance.
Fresh caches include graph identity, seed and split protocol and are shared only
between the two LP views. These redesigned splits and exact negatives mean the
results are not a rerun of the old matrix under identical data semantics.
