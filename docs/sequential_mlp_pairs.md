# Sequential neighbor MLP pairs

72 ordered distinct source pairs, seed 0, using the nine graph sources of the bias LP gallery. Stage A is reused from its existing source-validation-selected `best.pt`. Both encoder and learned decoder bias initialize stage B; AdamW moments, step counters, and parameter settings are preserved at the boundary (default `--optimizer-policy preserve`). The earlier reset-optimizer run remains separately archived in its original output directory. Only B supervision is sampled during stage B. This is sequential continuation, with no replay or interleaving.

The architecture, disjoint fixed neighbor context, 1:5 uniform exact-nonedge sampling, learning rate, and stopping rule match the bias gallery. Stage B selects minimum B-validation BCE, checks every 2,000 updates, stops after three stale checks (absolute improvement threshold 1e-4), with a 100,000-update cap. Caps are recorded separately from convergence. Stage counters are B-only; first-stage step and checkpoint identity are retained in metadata.

After training, each of the 72 frozen models is evaluated on the same nine uniform target pair sets as the gallery (648 cells). A and B targets measure retention and adaptation; the other seven are held-out-source transfer. Headline AUC/BCE use the frozen source decoder bias. Target calibration is diagnostic only. Existing singleton results provide the first-stage baseline.

Run from a dedicated Tucker worktree in tmux:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_sequential_mlp_pairs.sh /dataMeR1/phil/gfm/mixture-scaling/state/sequential_mlp_pairs_adam_s0
```

Four lock-protected workers share the training queue on GPUs 0–3. All workers must succeed before evaluation begins. Completed runs are skipped after first-stage provenance validation; partial runs are refused. No interleaved jobs are launched by this script.

Read-only preflight: `python -m mixture_scaling.sequential_mlp_pairs plan --root <root>`. Aggregation writes `results/matrix.csv` and a completion receipt only after all cells exist.
