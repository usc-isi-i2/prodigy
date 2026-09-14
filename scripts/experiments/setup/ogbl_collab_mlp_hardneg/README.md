# ogbl-collab feature-only interaction and hard-negative campaign

This frozen 2x2 campaign tests whether the feature-only nonlinear MLP's Collab
Hits@50 error is caused by its cosine-only pair interface, its easy uniform
training negatives, or their interaction. `cosine_uniform` is reused from the
validated `official_v1` campaign; the other nine seed-level cells are trained here.

All cells retain the official temporal split and fixed official validation/test
negatives. Test remains closed until validation Hits@50 selects a checkpoint.
Production defaults to offline W&B and writes authoritative local artifacts under
`/dataMeR1/phil/gfm/ogbl_collab_mlp_hardneg/hardneg_v1/`.

Dry-run one cell before launch:

```bash
python -m scripts.experiments.setup.ogbl_collab_mlp_hardneg.run \
  --scorer interaction --negative-policy hard8 --seed 0 \
  --out /tmp/not-created --device cuda:0 --dry-run
```

After a timed smoke cell establishes resource use, launch one seed per owned GPU:

```bash
bash scripts/experiments/setup/ogbl_collab_mlp_hardneg/launch_tucker.sh 0 0
bash scripts/experiments/setup/ogbl_collab_mlp_hardneg/launch_tucker.sh 1 1
bash scripts/experiments/setup/ogbl_collab_mlp_hardneg/launch_tucker.sh 2 2
```
