# Historical model-size evaluation

This evaluates only the retained 7,502,083-parameter UKR+COVID+midterm checkpoint.
The 1,640,514-parameter checkpoint is intentionally not rerun.

The runner reuses `icl_arch_matrix.evaluate_prodigy`, so each target gets the current
fast classification protocol: 128 fixed 2-way/10-shot test episodes, with the exact
episode stream fingerprinted in the JSONL result. The default four-target panel is
`covid_political`, `election2020`, `ukr_rus_suspended`, and `twibot20`.

On Tucker, from a checkout of this branch:

```bash
DRY_RUN=1 bash scripts/experiments/setup/nm_model_size_eval/run_tucker.sh
bash scripts/experiments/setup/nm_model_size_eval/run_tucker.sh
```

The job defaults to physical GPU 2 and writes beneath
`log/nm_model_size_eval/`. Set `GPU=3` to use the other owned GPU. To make a
small-versus-big comparison auditable, set `REFERENCE_RESULTS` to the existing small
model JSONL. The evaluator then refuses any target whose fixed-episode fingerprint
does not match:

```bash
REFERENCE_RESULTS=/absolute/path/to/small_1p64m.jsonl \
  bash scripts/experiments/setup/nm_model_size_eval/run_tucker.sh
```

The historical checkpoint and architecture are pinned by default:

- checkpoint: `state/merged_ukr_rus_covid_nm_11_06_2026_18_03_41/checkpoint/state_dict_110000.ckpt`
- architecture: `emb_dim=512`, `layers=S2,U,M2`, `dropout=0.1`, `n_hop=1`
- recorded pretraining sources: `ukr_rus,covid,midterm`

This is a one-seed historical ablation because only one retained large-model training
run exists. It supports a paired evaluation-episode comparison, not a multi-seed
capacity claim.
