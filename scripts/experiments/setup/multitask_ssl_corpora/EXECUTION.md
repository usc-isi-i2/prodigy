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
`scripts/experiments/analysis/evaluation/shared_task_tables/*/data/`. Success marker in `/tmp/msc_watcher.log`:
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

## Run log (as executed, 2026-07-21)

- Worktree `/dataMeR1/phil/gfm/prodigy-msc` created at 9271c2e; hooks enabled.
- 09:39 all 8 trainings launched via `launch_all_tucker.sh` (packing as above).
  Graph loading took ~5 min (cov, 78 GB) / ~30 min (all8, 111 GB); host RAM fine.
- Wall times (40k steps, final it/s): cov_NM 1:27 (7.6), cov_MIX 0:52 (12.9),
  cov_FP 0:48 (13.9), cov_CL 0:35 (19.0), all8_NM 1:36 (7.0), all8_MIX 0:55
  (12.2), all8_FP 0:50 (13.2), all8_CL 0:38 (17.4). All saved 10k/20k/30k ckpts
  and exited cleanly ("Saved best model").
- The auto-eval watcher was **not** used (session interruptions meant the eval
  was driven manually): 12:20 `make_model_list.sh` (8 x 30k ckpts) +
  `run_eval_sweep.sh --gpus 0,1,2,3` in tmux `msc_eval`; 144 eval runs, done
  ~15:00, marker `MULTITASK_SSL_CORPORA_EVAL_SWEEP_DONE` in `/tmp/msc_eval.log`.
- Bug found+fixed mid-run (4cbd4a4): `scripts/eval/eval_ckpts_all_graph_tasks_tucker.py`
  resolved the graph catalog with a stale pre-reorg depth (`parents[3]`) and
  crashed on startup; fixed to `parents[2]`, pulled on Tucker, relaunched.
- Parse gotcha (worked around in e7dd1c7): `parse_benchmark_eval_logs.py`
  REPLACES the shared CSVs with whatever the current log root contains — from a
  fresh worktree this drops all historical rows. The msc rows were re-merged
  with the HEAD rows before committing; msc-only copies live in
  `../../analysis/objectives/multitask_ssl/multitask_ssl_corpora/data/`.
- Tucker cannot push to GitHub non-interactively; result commits were fetched
  to the laptop over ssh (`git fetch ssh://tucker/dataMeR1/phil/gfm/prodigy-msc`)
  and pushed from there.
