# Source-unaware Order-A ladder on the final-core protocol

This is the naive merged-pretraining treatment for final-core Order A, seed 0. Rung 1
is identical to the existing Ukraine specialist and is reused. Rungs 2–9 train on the
same immutable all-nine final-core split artifact while restricting eligibility to the
active source union. Every episode then takes the `p=1` node-uniform mixed-source path.

The controlled comparison is the existing within-source final-core ladder:

- `static_train` context and disjoint `static_test` positives;
- fair 2-hop `9,9` / 101-node context;
- batch size 4, learning rate 0.002, weight decay 0.001;
- 2,500 optimizer updates and checkpoints 100/300/900/2500; and
- the exact 512-episode fixed-test stream with published fingerprint validation.

## Execution

Only GPUs 0 and 1 are used. Two workers split the eight physical models. Each worker
trains one rung, immediately evaluates that completed model on all nine final-core
targets, then takes its next rung. Thus evaluation of one rung overlaps training of the
other GPU whenever their phases differ.

```bash
bash scripts/experiments/setup/nm_ladder_global_nhop2/check_inputs_tucker.sh
bash scripts/experiments/setup/nm_ladder_global_nhop2/run_smoke_tucker.sh

DRY_RUN=1 GPUS="0 1" \
  bash scripts/experiments/setup/nm_ladder_global_nhop2/run_all_train_tucker.sh

tmux new-session -d -s nmglobal-finalcore \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   cd /dataMeR1/phil/gfm/prodigy-nmglobal; \
   GPUS="0 1" bash scripts/experiments/setup/nm_ladder_global_nhop2/run_all_train_tucker.sh \
   > log/nm_ladder_global_finalcore/orchestrator.log 2>&1'
```

Training checkpoints live under `state/nm_ladder_global_finalcore/`. Exact fixed-test
JSONs live under `log/nm_ladder_global_finalcore/fixed_test/results/`.
