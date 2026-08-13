# NM interpolation ladder — fill-in rungs (4→7 sources)

Completes the merged-graph NM "interpolation ladder" by filling the gap between
rung 3 (`ukr+cov+mid`) and rung 8 (`all8`). The original ladder jumped from 3
sources straight to 8; this experiment adds the remaining 5 graphs **one at a time,
in table-column order**, so each new rung lights up exactly the next test column — a
clean diagonal staircase.

## The full ladder

| Rung | Sources | Adds (column that enters) | Where the numbers come from |
|-----:|--------:|---------------------------|-----------------------------|
| 1 | 1 | ukr_rus_twitter | existing (ladder CSV) |
| 2 | 2 | covid19_twitter | existing (ladder CSV) |
| 3 | 3 | midterm | existing (ladder CSV) |
| **4** | **4** | **covid_political** | **trained + evaluated here** |
| **5** | **5** | **election2020** | **trained + evaluated here** |
| **6** | **6** | **ukr_rus_suspended** | **trained + evaluated here** |
| **7** | **7** | **twibot20** | **trained + evaluated here** |
| 8 | 8 | cp_hk_twitter (= all8) | existing (ladder CSV) |

Every rung is evaluated on **all 8** single-source graphs (NM, 30-way, 3-shot,
matched-40k), so the assembled table is 8 rungs × 8 columns.

## Protocol (matched to the existing ladder)

- **Merges** are disjoint block-concat, `drop_edge_features:true` (structure + node
  features only), carrying `graph_id` for within-source NM sampling. The 4 intermediate
  merges are literal **prefixes of the all8 input list** (same source order → same
  `graph_id`s), so rungs 4–7 interpolate cleanly between rung 3 and rung 8.
- **Training** clones `covid_ukr/merged_ukr_rus_covid_midterm_all8_nm.yaml` exactly
  (256·S,U,M base, no aug, `attr_regression_weight=0`, within-balanced episodes:
  `neighbor_sampling_episode_source: graph_id` + `balanced`). Only `graph_filename` +
  `prefix` change, plus `epochs:5`/`checkpoint_step:10000` so each run **self-terminates**
  at 50k with `state_dict_40000` as its final ckpt. The LR scheduler is disabled
  (constant LR), so 50k-planned vs the ladder's 120k-planned-then-killed are identical
  at the 40k checkpoint — no watcher needed.
- **Eval** reuses the shared harness `eval/eval_ckpts_all_graph_tasks_tucker.py` with
  the same flags as the ladder + single-source matrix (`--tasks nm --shots 3
  --nm-n-way 30 --data-root /dataMeR1/phil/data`).

## Run order (Tucker)

```bash
cd scripts/experiments/nm_ladder_fillin

# 0. sanity-check without touching GPUs
DRY_RUN=1 ./build_merges_tucker.sh
DRY_RUN=1 ./run_all_train_tucker.sh

# 1. build the 4 intermediate merged graphs (idempotent; ~minutes each, CPU/RAM)
./build_merges_tucker.sh

# 2. train the 4 rungs. NM is ~2GB so these coexist with the running nmss jobs.
#    In tmux (see AGENTS.md conda-on-PATH gotcha for detached sessions):
mkdir -p run_logs
tmux new-session -d -s nmladder_fillin \
  'export PATH="/home/mhchu/miniconda3/bin:$PATH"; \
   GPUS="0 1 2 3" bash scripts/experiments/setup/nm_ladder_fillin/run_all_train_tucker.sh \
   > scripts/experiments/setup/nm_ladder_fillin/run_logs/orchestrator.log 2>&1'
#    ~40k steps at ~7 it/s ≈ 1.5h/rung; one rung per GPU => ~1.5h wall.

# 3. once all four state_dict_40000.ckpt exist, point the model list at them
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh

# 4. eval the 4 rungs on all 8 graphs (32 NM jobs)
GPUS="0,1,2,3" ./eval_ladder_tucker.sh

# 5. assemble the complete 8x8 table (pulls existing rungs from the ladder CSV)
python3 assemble_full_table.py --log-root /dataMeR1/phil/gfm/prodigy/log
#    -> nm_ladder_full.csv + a printed staircase diagnostic
```

## Outputs

- `model_list.txt` — the 4 rungs at `state_dict_40000` (written by `make_model_list.sh`).
- `nm_ladder_full.csv` — the complete 8-rung × 8-column table (the deliverable).
- Eval logs land in `<repo>/log/eval_nm_ladder_<N>src_to_<test>_nm_3shot_30way_*/`.

## Notes / gotchas

- **Coexistence:** these share GPUs with the `nmss` single-source-matrix run; NM
  episodes are tiny, so no need to wait for or kill it. `nvidia-smi` first anyway.
- **Existing rungs:** `assemble_full_table.py` reads them from
  `scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/data/nmladder_results.csv`. If that CSV isn't on Tucker (it
  may be gitignored), the script falls back to the published values baked into it and
  prints a warning — the table is still complete.
- **Fill-in order** = the table's column order (twitter-like graphs first, the two
  social_llm graphs twibot20/cp_hk last). To use a different curriculum, reorder the
  `inputs:` in the `merge_*.yaml` files and the `added` column in `assemble_full_table.py`.
- **1 seed**, matched-40k — same caveats as the rest of the ladder; hedge sub-1% gaps.
