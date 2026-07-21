# multitask_ssl_corpora — execution

Operational companion to `README.md`. Run on Tucker from an **isolated worktree**
holding branch `exp/multitask-ssl-corpora` (never the main tree):

```bash
cd /dataMeR1/phil/gfm/prodigy && git fetch origin
git worktree add /dataMeR1/phil/gfm/prodigy-msc exp/multitask-ssl-corpora
cd /dataMeR1/phil/gfm/prodigy-msc && git config core.hooksPath .githooks
```

## 1. Launch the 8 pretrains (detached tmux, 2 per GPU on 0-3)

```bash
cd /dataMeR1/phil/gfm/prodigy-msc
nvidia-smi   # confirm GPUs 0-3 have headroom (4-7 are other users' — never touch)
bash scripts/experiments/setup/multitask_ssl_corpora/launch_all_tucker.sh
```

GPU packing (long poles overlap; lightest pair on the partially-occupied GPU 3):
GPU0 cov_NM+all8_NM, GPU1 cov_MIX+all8_MIX, GPU2 cov_CL+all8_CL, GPU3
cov_FP+all8_FP. The script uses env-python directly (conda is not initialized in
detached tmux). Logs: `/tmp/msc_<corpus>_<ARM>.log`; state:
`state/msc_<corpus>_<ARM>_<ts>/checkpoint/` (ckpts 10k/20k/30k; no 40k — the
off-by-one is intended, see README).

Expected throughput (from the 3-way run: nm ~7 it/s, cl/fp ~30, mix ~15; all8
somewhat slower, and 2 runs/GPU contend): NM arms are the long pole, ~2-3h.

Verify ~5-10 min in: `tmux ls | grep msc_`, `tail /tmp/msc_*.log` (it/s visible),
`nvidia-smi` shows 8 python processes on GPUs 0-3.

## 2. Arm the watcher (auto model-list + eval + CSV commit)

```bash
cd /dataMeR1/phil/gfm/prodigy-msc
git config user.name  # must be sane; the watcher commits result CSVs
tmux new-session -d -s msc_watcher \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   bash scripts/experiments/setup/multitask_ssl_corpora/watch_and_eval.sh \
     > /tmp/msc_watcher.log 2>&1'
```

The watcher polls every 5 min until every run has
`checkpoint/state_dict_30000.ckpt` AND its tmux session has exited, then runs
`make_model_list.sh` (30k ckpts, keyed cov_NM..all8_MIX), `run_eval_sweep.sh
--gpus 0,1,2,3`, and commits+pushes the refreshed CSVs under
`scripts/experiments/analysis/*/data/`. Success marker in `/tmp/msc_watcher.log`:
`ALL COMPLETE`. Timeout (8h default): `WATCHER TIMEOUT — INCOMPLETE`, no eval.

## 3. Manual fallback (if the watcher dies)

```bash
cd /dataMeR1/phil/gfm/prodigy-msc
STATE_DIR=$PWD/state bash scripts/experiments/setup/multitask_ssl_corpora/make_model_list.sh
source /home/mhchu/miniconda3/etc/profile.d/conda.sh && conda activate prodigy
MODEL_LIST=scripts/experiments/setup/multitask_ssl_corpora/model_list.txt \
  bash scripts/experiments/setup/multitask_ssl_corpora/run_eval_sweep.sh --gpus 0,1,2,3
```

## Relaunching a crashed arm

```bash
tmux kill-session -t msc_<corpus>_<ARM> 2>/dev/null
ONLY="<corpus>_<ARM>" bash scripts/experiments/setup/multitask_ssl_corpora/launch_all_tucker.sh
```
(Stale partial state dirs are harmless: make_model_list picks the newest
`msc_<run>_<ts>` dir and demands the 30k ckpt.)

## Run log (as executed)

Filled in by the launching session — see the final workstream report.
