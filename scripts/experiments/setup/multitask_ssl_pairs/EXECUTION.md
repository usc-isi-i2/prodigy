# multitask_ssl_pairs — execution

Run on Tucker, prodigy env, in tmux. All 3 arms are independent → one GPU each,
~1h wall (see throughput below). Paths below match the `multitask_ssl_rotation`
run; **verify against the current Tucker tree/worktree before launching.**

```bash
REPO=/dataMeR1/phil/gfm/prodigy            # tree holding this branch's code (verify)
PAIRS=scripts/experiments/multitask_ssl_pairs
```

## 0. Smoke test first (always)

```bash
# dry run — print the resolved command, no training:
DRY_RUN=1 $PAIRS/train_arm_tucker.sh NMCL --device 0

# tiny real run — 1 arm, ~40 episodes, verify the fp/cl dispatch + a ckpt is written:
$PAIRS/train_arm_tucker.sh NMFP --device 0 --epochs 1 -ds_cap 20 \
  -eval_step 20 -ckpt_step 20 --prefix mtp_NMFP_smoke
# expect the loader banner "nm_fp_cl rotation (per-episode): nm:cl:fp counts = 1:0:1"
# and a state_dict_*.ckpt under state/mtp_NMFP_smoke_*/checkpoint/. Then delete it.
```

## 1. Train the 3 pair arms (1 GPU each, in tmux)

**Interactive / synchronous ssh** — the committed script activates conda itself:
```bash
cd $REPO
tmux new -s mtp_NMCL -d "$PAIRS/train_arm_tucker.sh NMCL --device 0"
tmux new -s mtp_NMFP -d "$PAIRS/train_arm_tucker.sh NMFP --device 1"
tmux new -s mtp_CLFP -d "$PAIRS/train_arm_tucker.sh CLFP --device 2"
```

**Detached tmux gotcha** (from the rotation run): conda is NOT initialized in a
detached, non-interactive tmux (`.bashrc` short-circuits; `conda` is a shell fn not
inherited). If the committed script fails with a conda error, bypass conda and call
the env-python directly:
```bash
cd $REPO
LD_LIBRARY_PATH=/home/mhchu/miniconda3/envs/prodigy/lib \
  /home/mhchu/miniconda3/envs/prodigy/bin/python experiments/run_single_experiment.py \
  --config $PAIRS/configs/NMCL.yaml --device 0   # repeat NMFP dev1, CLFP dev2
```

Budget: 40k episodes (epochs 4 × cap 10000), `checkpoint_step=10000` → ckpts at
10k/20k/30k; step 39999 is 0-indexed so the 40k boundary never saves — the final
usable ckpt is **30k**, matching the rotation arms (see the rotation FINDINGS /
`trainer-checkpoint-off-by-one`). Throughput: nm ~7 it/s, cl/fp ~30 it/s ⇒
NMCL/NMFP ≈ 1h (nm is the long pole), CLFP ≈ 22 min. State under
`state/mtp_<ARM>_<ts>/checkpoint/`.

## 2. Build the merged model list (all 7 arms)

Score pairs + single/MIX controls in ONE sweep so every arm is evaluated under
identical conditions. The rotation checkpoints live under the `mtr` worktree's
state dir; point `ROTATION_STATE_DIR` at it (verify the path):
```bash
ARMS="NMCL NMFP CLFP NM CL FP MIX" \
  STATE_DIR=$REPO/state \
  ROTATION_STATE_DIR=/dataMeR1/phil/gfm/prodigy-mtr/state \
  bash $PAIRS/make_model_list.sh
cat $PAIRS/model_list.txt          # expect 7 lines: <ARM> <30k ckpt>
```
(For pairs-only, drop `ARMS`/`ROTATION_STATE_DIR` → defaults to NMCL NMFP CLFP.)

## 3. Frozen-encoder eval sweep

```bash
MODEL_LIST=$PAIRS/model_list.txt bash $PAIRS/run_eval_sweep.sh --gpus 0,1,2,3
# writes reg/slp/cls rows keyed by model=arm into scripts/experiments/analysis/evaluation/task_tables/*/data/*.csv
# and prints MULTITASK_SSL_PAIRS_EVAL_SWEEP_DONE. Runs from $REPO so --log-root
# resolves to $REPO/log (the parser derives it from REPO_ROOT — do not hardcode).
```

## 4. Aggregate → the subset-lattice table

```bash
python3 $PAIRS/aggregate_results.py --plotting-root scripts/experiments/analysis/evaluation/task_tables
```
Reads the 7 arms and prints: the lattice table (cls/reg/sLP + min-bar by k),
capability-vs-#objectives, the ranked generalist bar, and the HEADLINE — which
arms clear static-LP and the marginal sLP per objective (with vs without). Write
the reading into `FINDINGS.md` (mirror `../multitask_ssl_rotation/FINDINGS.md`).

## Notes
- Single seed (seed 0), matching the rotation run — "spread" is across the 5 eval
  datasets, not across seeds.
- If the rotation CSVs are stale/unavailable, re-eval those 4 arms here too (they
  are in the merged model_list above) rather than trusting old rows — identical
  eval code is the point.
