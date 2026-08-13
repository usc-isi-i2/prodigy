# Pretrain saturation — the dense early-training half

Three short retrains that supply the two points the surviving trajectories cannot:
**step 100 and step 500**. Every historical run used `checkpoint_step: 1000`, so nothing
below 1000 was ever written, and that is precisely the region where a saturation curve
either bends or does not.

The other twelve points come from
[`../pretrain_saturation_existing/`](../pretrain_saturation_existing/); both halves feed
`analysis/transfer/ablations/prodigy_nm/saturation/pretrain_saturation/`. Arm definitions and the `sat_<arm>_s<step>` key
convention are imported from that folder's [`arms.py`](../pretrain_saturation_existing/arms.py)
so the two halves join instead of producing disjoint rows.

## The runs

| config | arm | clone of | steps |
|---|---|---|---|
| `train_all8_dense.yaml` | `all8` | `covid_ukr/merged_ukr_rus_covid_midterm_all8_nm.yaml` | 2100 |
| `train_ukr_dense.yaml` | `ukr` | `ukr_only/ukr_only_nm.yaml` | 2100 |
| `train_covid_dense.yaml` | `covid` | `covid_only/covid_only_nm.yaml` | 2100 |

Each is its source config with four changes: `epochs: 1` + `dataset_len_cap: 2100`
(budget), `checkpoint_steps`, `eval_step`, and a new `prefix`. Everything the
optimization trajectory depends on is identical, so each run is a genuine **prefix** of
its historical counterpart:

- `BatchSampler` seeds its RNG once from the split name and draws episodes lazily
  (`data/dataloader.py:449`), so `dataset_len_cap` sets only *how many* episodes are
  drawn, never *which*. 2100 < 10000, so neither run reaches an epoch boundary.
- The LR scheduler is commented out (`experiments/trainer.py:437`), so a shorter budget
  does not change the learning rate at any step.

Cost, from the historical runs' checkpoint mtimes: all8 ~7.6 steps/s (~5 min), ukr/covid
~4.5 steps/s (~8 min). Loading the graphs dominates — covid is 73 GB on disk, all8 104 GB.

## `checkpoint_steps: "100,500,1000,1001,2000,2001"`

- **100, 500** — what these runs exist to produce.
- **1000, 2000** — duplicate surviving checkpoints; a free cross-check.
- **1001, 2001** — splice probes, never evaluated. Because the old trainer named a save
  by the pre-increment loop variable, a historical `state_dict_1000` holds 1001 steps, so
  it should equal our `state_dict_1001` tensor-for-tensor.

This schedule needs `-ckpt_steps/--checkpoint_steps` (`b75bae9`); a modulo cadence cannot
express it. A terminal `state_dict_2100` is also written, as it is for any run.

## `eval_step` deliberately differs for `ukr` and `covid`

Their source configs set `eval_step: 1000`; the dense configs use `100000`. Two reasons,
both about checkpoint integrity:

1. The in-loop **best**-checkpoint save (`experiments/trainer.py:1647`) still names by the
   pre-increment loop variable — the 2026-07-26 fix covered the periodic and terminal
   saves, not this one. With `eval_step: 1000` it would overwrite our clean
   `state_dict_1000` (1000 steps) with a 1001-step model.
2. A val eval consumes global torch RNG (`Collator` → `linearize` → `torch.rand`), so the
   training stream diverges after the first one fires.

Consequence, and it is a real limitation: for `ukr` and `covid` the **2001 probe is not
comparable** — their historical runs evaluated at step 1001 and these do not.
`check_splice.py` skips it for those arms and says so rather than reporting a spurious
failure. The **1001 probe is** comparable for all three arms, because the historical
periodic save fired *before* the eval block in the same iteration. `all8` used
`eval_step: 100000` and is checked at both probes.

## Running it

```bash
DRY_RUN=1 bash run_all_train_tucker.sh
```

```bash
tmux new-session -d -s sat_dense 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash scripts/experiments/setup/pretrain_saturation_dense/run_all_train_tucker.sh'
```

Then prove the splice before trusting anything downstream:

```bash
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state python3 check_splice.py
```

It compares each probe against a reference distance — how far the historical run itself
moved between its own steps 1000 and 2000 — and calls "same trajectory" only when the
difference is at least 100× smaller. Exact equality is not expected: different days,
possibly different GPUs, nondeterministic scatter kernels. **If it fails, do not splice**;
retrain the affected arm to 40000 with
`--checkpoint_steps 100,500,1000,2000,10000,40000` instead.

Finally:

```bash
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state python3 make_model_list.py
```

```bash
DRY_RUN=1 bash run_eval_sweep.sh --gpus 0,1
```

`run_eval_sweep.sh` is a thin wrapper that execs the existing half's sweep script with a
different model list — the two halves must be measured with identical flags, and two
copies of a flag list drift. 6 ckpts × 12 = **72 jobs**.

**GPU etiquette:** checked 2026-07-27 — a vLLM worker is pinned across GPUs **2 and 3**
(~76 GB each), so only 0 and 1 are free. Verify before launching:

```bash
ssh tucker nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

Give this its own worktree if anything else is running out of the main checkout —
`git worktree add ../prodigy-sat experiment/pretrain-saturation`. `state/` is per-worktree
and gitignored, so the checkpoints land wherever the job ran; `STATE_DIR` must point at
that tree, while the *historical* checkpoints stay in the main checkout's `state/`.
