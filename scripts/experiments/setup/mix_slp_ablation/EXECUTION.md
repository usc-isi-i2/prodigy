# Execution tracker — mix_slp_ablation

Companion to `README.md` (design + interpretation matrix). Exact commands as
run, checkpoints, and seeds. Findings land in
`scripts/experiments/analysis/mix_slp_ablation/FINDINGS.md`.

## Fixed facts

- **Branch:** `exp/mix-slp-ablation` (from `refactor/experiments-analysis-split`
  @ `2c03936`). Tucker worktree: `/dataMeR1/phil/gfm/prodigy-abl`.
- **Seed:** `SEED=0` (runner `--seed 0` → `run_single_experiment.py` seeds
  torch/np/random; the ablation augs draw from the seeded torch RNG). Eval
  episodes are seeded by split name (`sum(ord(split))`) — identical across
  conditions by construction.
- **Checkpoints (pinned step 30000, the FINDINGS checkpoints):**
  - MIX: `/dataMeR1/phil/gfm/prodigy-mtr/state/mtr_MIX_09_07_2026_15_09_52/checkpoint/state_dict_30000.ckpt`
  - NM:  `/dataMeR1/phil/gfm/prodigy-mtr/state/mtr_NM_09_07_2026_15_09_52/checkpoint/state_dict_30000.ckpt`
- **Eval:** static-LP only, 0-shot, `--slp-n-query 4`, hard negatives,
  datasets `midterm,ukr_rus_twitter,covid19_twitter,twibot20`.

## Status

| Step | Status |
|---|---|
| Implementation (runner catalog-path fix, setup scripts, parser) | ✅ laptop, committed |
| Tucker worktree + model list (pinned 30k) | ✅ |
| Sanity: `none`, MIX, covid19 → must reproduce ≈0.755 | ✅ 0.75508, exact |
| Full grid 2 arms × 4 conditions × 4 datasets (32 runs) | ✅ 2026-07-21, 0 errors |
| Parse → `data/slp_ablation_2x2.csv`, commit on Tucker, push | ✅ (pulled to laptop via `git fetch ssh://tucker/...` — Tucker has no push credentials) |
| FINDINGS.md verdict (laptop) | ✅ **topological** — rewire collapses MIX (0.759→0.576), permute is a no-op (0.762) |

## Step 0 — Tucker worktree (once)

```bash
ssh tucker   # or ssh -i ~/.ssh/id_ed25519 mhchu@10.137.32.100
cd /dataMeR1/phil/gfm/prodigy
git fetch origin
git worktree add /dataMeR1/phil/gfm/prodigy-abl origin/exp/mix-slp-ablation
cd /dataMeR1/phil/gfm/prodigy-abl
git checkout -B exp/mix-slp-ablation origin/exp/mix-slp-ablation
git config core.hooksPath .githooks
```

## Step 1 — Model list (pinned 30k ckpts)

```bash
cd /dataMeR1/phil/gfm/prodigy-abl
STATE_DIR=/dataMeR1/phil/gfm/prodigy-mtr/state \
  bash scripts/experiments/setup/mix_slp_ablation/make_model_list.sh
cat scripts/experiments/setup/mix_slp_ablation/model_list.txt
```

## Step 2 — Sanity anchor (MIX, none, covid19 only)

Must land near the FINDINGS value 0.755 before ablations are worth running
(exact reproduction expected: same ckpt, same split-seeded episodes, same
eval params as the original sweep).

```bash
source /home/mhchu/miniconda3/etc/profile.d/conda.sh && conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd /dataMeR1/phil/gfm/prodigy-abl
grep '^MIX ' scripts/experiments/setup/mix_slp_ablation/model_list.txt \
  > /tmp/abl_mix_only.txt
MODEL_LIST=/tmp/abl_mix_only.txt DATASETS=covid19_twitter CONDITIONS=none \
  SKIP_CONDA=1 SEED=0 \
  bash scripts/experiments/setup/mix_slp_ablation/run_2x2_slp.sh --gpus 0 \
  2>&1 | tee /tmp/abl_sanity.log
```

## Step 3 — Full grid (tmux; ~32 light eval runs, one GPU)

```bash
tmux new-session -d -s abl2x2 'export PATH="/home/mhchu/miniconda3/bin:$PATH"; cd /dataMeR1/phil/gfm/prodigy-abl && bash scripts/experiments/setup/mix_slp_ablation/run_2x2_slp.sh --gpus 0 > /tmp/abl_2x2.log 2>&1'
tail -f /tmp/abl_2x2.log   # ends with MIX_SLP_ABLATION_2X2_DONE
```

## Step 4 — Results back

```bash
cd /dataMeR1/phil/gfm/prodigy-abl
git add scripts/experiments/analysis/mix_slp_ablation/data/slp_ablation_2x2.csv
git commit -m "mix_slp_ablation: 2x2 static-LP results from Tucker"
git push origin exp/mix-slp-ablation
# laptop: git pull, write FINDINGS.md
```

## As-run log

- **2026-07-21** Laptop: implementation committed (`24a92f4`), pushed.
- **2026-07-21** Tucker: worktree `/dataMeR1/phil/gfm/prodigy-abl` created on
  `exp/mix-slp-ablation` @ `24a92f4`; hooksPath set; model list pinned to the
  two 30k checkpoints above (both verified present).
- **2026-07-21 09:39** Sanity anchor (Step 2, GPU 1):
  `eval_MIX_to_covid19_twitter_slp_0shot_21_07_2026_09_39_45` →
  `test_roc_auc = 0.75508`, **exactly** equal to the original sweep's
  `eval_MIX_to_covid19_twitter_slp_0shot_09_07_2026_17_34_25`
  (`test_roc_auc = 0.75508`, same accuracy 0.6765 / f1 0.57905). Split-name
  episode seeding + identical runner flags ⇒ bit-identical anchor. Proceed.
- **2026-07-21 ~10:0x** Full grid (Step 3) launched in tmux `abl2x2`,
  `--gpus 0,1,2,3`, `SEED=0`, log `/tmp/abl_2x2.log`
  (32 runs = 2 arms × 4 conditions × 4 datasets; the `none` cells re-run
  rather than reusing the July-09 sweep so all four conditions share one
  code state; the parser keeps the latest timestamp per cell).
- **2026-07-21 12:20–12:55 (Tucker clock)** Grid ran as 4 sequential runner
  invocations (one per condition, 8 jobs each, 4 parallel GPU slots), sharing
  GPUs politely with the sibling `msc_eval` sweep launched 12:22. 32/32 jobs
  done, 0 errors, ~35 min wall. Metrics gotcha: each run writes
  `log/<run>/data/metrics_test_step0.json` (step-suffixed; no unstepped
  `metrics_test.json`, no val file) — `parse_slp_2x2.py` handles this.
- **2026-07-21** `parse_slp_2x2.py` → 32-row
  `analysis/mix_slp_ablation/data/slp_ablation_2x2.csv`; committed on Tucker
  (ambient Davidchu identity, matching prior Tucker-side result commits),
  fetched to the laptop worktree over ssh (`git fetch
  ssh://tucker/dataMeR1/phil/gfm/prodigy-abl exp/mix-slp-ablation` +
  cherry-pick; Tucker's https remote has no credentials), pushed; Tucker
  worktree hard-reset to `origin/exp/mix-slp-ablation` to reconverge.
- **2026-07-21** FINDINGS written: verdict **topological** (see
  `scripts/experiments/analysis/mix_slp_ablation/FINDINGS.md`).
