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
