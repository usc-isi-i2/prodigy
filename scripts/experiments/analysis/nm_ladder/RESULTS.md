# NM interpolation ladder — complete 8-rung results

The findings file for the whole merged-graph NM ladder. `nm_ladder/` and
`nm_ladder_fillin/` were merged into this one folder on 2026-07-26; setup for the
fill-in runs stays in `../../setup/nm_ladder_fillin/`. Robustness of the rung order
is a separate experiment: [`../nm_ladder_order_robustness/`](../nm_ladder_order_robustness/).

Fills rungs 4–7 of the merged-graph NM ladder (the original jumped rung 3 → rung 8).
Each new rung adds one source graph, in table-column order. NM 3-shot / 30-way,
matched-40k (`state_dict_40000`), within-balanced episodes, evaluated on all 8
single-source graphs with the shared eval harness. **1 seed.**

- Rungs 1/2/3/8: existing ladder (`scripts/experiments/analysis/nm_ladder/data/nmladder_results.csv`).
- Rungs 4/5/6/7: trained + evaluated here (2026-07-13). Cross-checked against raw
  `metrics_test` and the fallback-vs-CSV agreement (rungs 1/2/3/8 identical to the digit).

## The full table (test roc_auc; **bold** = the graph that first enters at that rung)

| rung | sources | train adds | ukr | covid | midterm | cov_pol | elec20 | ukr_susp | twibot20 | cp_hk |
|-----:|--------:|------------|----:|------:|--------:|--------:|-------:|---------:|---------:|------:|
| 1 | 1 | ukr | **.948** | .973 | .874 | .849 | .828 | .771 | .921 | .724 |
| 2 | 2 | +covid | .945 | **.980** | .885 | .843 | .828 | .775 | .925 | .726 |
| 3 | 3 | +midterm | .941 | .978 | **.915** | .830 | .815 | .777 | .927 | .720 |
| 4 | 4 | +covid_political | .934 | .975 | .909 | **.911** | .830 | .777 | .923 | .724 |
| 5 | 5 | +election2020 | .935 | .975 | .909 | .910 | **.926** | .769 | .925 | .726 |
| 6 | 6 | +ukr_rus_suspended | .933 | .974 | .907 | .911 | .924 | **.934** | .924 | .724 |
| 7 | 7 | +twibot20 | .932 | .975 | .903 | .908 | .920 | .926 | **.938** | .727 |
| 8 | 8 | +cp_hk (=all8) | .934 | .975 | .908 | .906 | .920 | .931 | .937 | **.867** |

## The interpolation staircase (added column, rung below → this rung)

| rung | enters | before → after | Δ |
|-----:|--------|----------------|---:|
| 4 | covid_political | .830 → .911 | **+.081** |
| 5 | election2020 | .830 → .926 | **+.096** |
| 6 | ukr_rus_suspended | .769 → .934 | **+.165** |
| 7 | twibot20 | .924 → .938 | +.013 |
| 8 | cp_hk | .727 → .867 | **+.140** |

## Extended with the single-source specialists

The concurrent `nm_single_source_matrix` run (train NM on **one** graph, test on all 8;
same matched-40k / 30-way / 3-shot protocol) stacks directly under the ladder. **Bold** =
in-domain diagonal (train == test = specialist ceiling).

| train (1 graph) | ukr | covid | midterm | cov_pol | elec20 | ukr_susp | twibot20 | cp_hk |
|-----------------|----:|------:|--------:|--------:|-------:|---------:|---------:|------:|
| ukr | **.947** | .973 | .881 | .839 | .826 | .789 | .922 | .714 |
| covid | .926 | **.981** | .884 | .850 | .835 | .786 | .926 | .720 |
| midterm | .797 | .879 | **.925** | .835 | .805 | .644 | .861 | .626 |
| covid_political | .630 | .699 | .688 | **.915** | .783 | .551 | .737 | .544 |
| election2020 | .602 | .655 | .680 | .787 | **.952** | .563 | .710 | .548 |
| ukr_rus_suspended | .770 | .831 | .725 | .733 | .728 | **.964** | .765 | .623 |
| twibot20 | .869 | .947 | .860 | .843 | .803 | .712 | **.949** | .690 |
| cp_hk | .681 | .758 | .763 | .683 | .641 | .603 | .738 | **.906** |

**Specialist ceiling vs all-8 merged** (does merging cost accuracy in-domain?):

| col | specialist | all8 | Δ (spec − merged) |
|-----|-----------:|-----:|------------------:|
| ukr | .947 | .934 | +.013 |
| covid | .981 | .975 | +.006 |
| midterm | .925 | .908 | +.017 |
| cov_pol | .915 | .906 | +.009 |
| elec20 | .952 | .920 | **+.032** |
| ukr_susp | .964 | .931 | **+.033** |
| twibot20 | .949 | .937 | +.012 |
| cp_hk | .906 | .867 | **+.039** |

- **Merging costs little in-domain**, and the cost is smallest on the big twitter graphs
  (covid +.006, ukr +.013) and largest on the small/topical ones (cp_hk +.039, ukr_susp
  +.033, elec20 +.032) — the small domains get diluted in the shared budget.
- **covid and ukr are near-universal donors** (each transfers ≥.83 to most targets); the
  small/topical specialists (cov_pol, election2020, cp_hk) transfer narrowly — their
  off-diagonal rows collapse to ~.55–.70.
- Consistency: single-source `ukr` (.947) ≈ ladder rung 1 (.948) — same ukr-only model,
  independent run, reproduces to ±.001.

## Read

- **Clean staircase.** Every "rest" graph's column stays flat at its zero-shot transfer
  level until that graph enters the training merge, then jumps and holds. The intermediate
  rungs interpolate smoothly between rung 3 and rung 8 — no surprises.
- **twibot20 is the exception** (+.013): it already sits ~.92 before entering, i.e. NM
  pretrained on the twitter retweet graphs already transfers to it well. It's a retweet
  graph like the others, so its own inclusion adds little.
- **cp_hk is the hardest target** — flat ~.72 across rungs 1–7, only reaching .867 once it's
  in training (rung 8). It's the most out-of-distribution source (HK political).
- **Adding sources barely taxes the in-training columns:** ukr .948 → .932 over 7 additions,
  covid .980 → .975, midterm .915 → .903 — small shared-budget dilution, not collapse
  (the within-balanced sampling protects the small domains).
- **Column, not count, drives AUC:** rung 7 (everything but cp_hk) ≈ rung 8 (all8) on every
  column *except* cp_hk — a graph's score is set by whether it's in training, not by how many
  graphs are.
- Cross-check: ukr→ukr = .948 at rung 1; the new rungs' ukr column (.932–.935) sits at the
  all8 level (.934) — no protocol drift.

**Caveats:** 1 seed, matched-40k; treat sub-.02 gaps as noise.

## Reproduce

See `README.md`. Build merges → `run_all_train_tucker.sh` (GPUs 4–7) →
`make_model_list.sh` → `eval_ladder_tucker.sh` → `assemble_full_table.py`
(→ `nm_ladder_full.csv`, gitignored).
