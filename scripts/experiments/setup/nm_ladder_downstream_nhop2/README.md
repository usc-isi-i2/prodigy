# Fair-two-hop ladder downstream evaluation

This experiment scores the completed fair-two-hop NM ladder variants on node
classification and repaired static link prediction. It does not train a model.

The historical one-hop downstream ladder found a static-LP entry effect but no
classification entry effect. The recent fair-two-hop experiments were evaluated only
on NM. This sweep asks whether the downstream result survives the protocol changes.

## Registered scope

| variant | logical rows | physical checkpoints | order(s) | intervention |
|---|---:|---:|---|---|
| matched40k | 8 | 8 | A | interleaved, 40k total |
| sequential | 8 | 8 | A | blocked source schedule, 40k total |
| split | 8 | 8 | A | NM trained on background edges and tested on holdout edges |
| fixed10k | 16 | 15 | A, C | 10k expected episodes per active source |
| **total** | **40** | **39** | | |

Fixed-exposure A8 and C8 are one shared all-eight model. The analysis expands that model
into two logical rows but never counts it as two physical encoders.

Tasks:

- **node classification:** 10-shot episodic evaluation on `covid_political`,
  `election2020`, `ukr_rus_suspended`, and `twibot20`; 39 × 4 = 156 jobs;
- **static link prediction:** repaired pair-conditioned cosine evaluator on
  `ukr_rus_twitter`, `covid19_twitter`, `midterm`, `twibot20`, and `cp_hk_twitter`;
  degree-matched negatives are primary, with random and hard-2-hop sets retained as
  robustness reads.

Temporal LP is excluded. Its episodic evaluator has the same known defects as the old
static-LP path and has never been repaired.

## Locked evaluation protocol

Every encoder readout uses the fair-two-hop context: `n_hop=2`, fanouts `9,9`, node cap
101, and the unchanged one-hop NM-positive walk setting. Classification uses the shared
runner. Static LP uses `scripts/eval/pair_link_sweep.py` over `static_background` and
`static_holdout`; the sweep now accepts fanouts and node cap explicitly, rather than
silently falling back to its old sampler defaults.

All classification arms receive the same split-derived evaluation episodes. All static-LP
arms on a graph receive one shared positive/negative pair set. Changing `--seed` does not
resample classification episodes; the comparison is paired, not a seed-confidence study.

## Git and Tucker isolation

Implementation branch: `codex/nm-ladder-downstream-nhop2`.

Create a dedicated Tucker worktree after the branch is pushed. Check `tmux ls`,
`git worktree list`, `nvidia-smi`, and `free -h` first; never pull a worktree with a live
job. Name the branch explicitly:

```bash
cd /dataMeR1/phil/gfm
git -C prodigy fetch origin codex/nm-ladder-downstream-nhop2
git -C prodigy worktree add -b codex/nm-ladder-downstream-nhop2 \
  ../prodigy-nmld-h2 origin/codex/nm-ladder-downstream-nhop2
cd /dataMeR1/phil/gfm/prodigy-nmld-h2
git config core.hooksPath .githooks
```

The resolver reads checkpoints from the worktrees that trained them:

```text
/dataMeR1/phil/gfm/prodigy-nmlh2/state
/dataMeR1/phil/gfm/prodigy-nmlh2seq/state
/dataMeR1/phil/gfm/prodigy-nmlsplit-h2/state
/dataMeR1/phil/gfm/prodigy-nmlfxh2/state
```

Each root is overrideable with `--<variant>-state-root` or the corresponding environment
variable, such as `FIXED10K_STATE_ROOT`.

## Preflight and dry runs

The plan is derived from the four committed manifests rather than duplicated by hand:

```bash
python3 scripts/experiments/setup/nm_ladder_downstream_nhop2/make_model_list.py --dry-run
```

On Tucker, resolve all 39 checkpoints and inspect both command families without touching
a GPU:

```bash
python3 scripts/experiments/setup/nm_ladder_downstream_nhop2/make_model_list.py
DRY_RUN=1 bash scripts/experiments/setup/nm_ladder_downstream_nhop2/run_classification_sweep.sh
DRY_RUN=1 bash scripts/experiments/setup/nm_ladder_downstream_nhop2/run_pair_lp_parallel.sh
```

## Four-GPU run

The pipeline uses all four owned GPUs in each expensive phase. Classification launches a
four-slot queue over GPUs 0–3. Static LP launches four graph workers concurrently:

| GPU slot | graph assignment |
|---:|---|
| 0 | COVID |
| 1 | Ukraine/Russia |
| 2 | Midterm, then Hong Kong |
| 3 | TwiBot-20 |

This deliberately overlaps the large graph loads and assumes the host RAM check is green.
Override `GPUS` only with four IDs from 0–3.

```bash
tmux new-session -d -s nmld_h2 \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0,1,2,3" bash scripts/experiments/setup/nm_ladder_downstream_nhop2/run_pipeline_tucker.sh'
```

Monitor:

```bash
cat scripts/experiments/setup/nm_ladder_downstream_nhop2/run_logs/pipeline_status.txt
tail -f scripts/experiments/setup/nm_ladder_downstream_nhop2/run_logs/pipeline.log
```

Phases are `resolve`, `smoke`, `classification`, `static_lp`, and `assemble`. Run one with
`ONLY=<phase>`. Pair-LP workers use `--resume`: a retry preserves validity-clean completed
models and evaluates only missing or invalid models.

Raw evidence lands under `analysis/nm_ladder_downstream_nhop2/data/raw/`; the strict
assembler refuses partial physical matrices by default.
