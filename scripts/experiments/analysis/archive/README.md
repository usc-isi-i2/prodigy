# Archive

Retired analyses and superseded work. Nothing here is current — check
[`../_cross/README.md`](../_cross/README.md) for the live tree.

## In the working tree

| folder | what |
|---|---|
| `multitask_ssl_superseded/` | the rotation/pairs work voided by the 2026-07-23 static-LP evaluator rescore; see its `README.md` |
| `outputs_old/` | pre-reorg setup and eval docs from `outputs/old/`, superseded by `AGENTS.md` |
| `misc/` | loose artifacts that belonged to no experiment folder |

## Off the working tree — 23 analyses, ~22 MB (removed 2026-07-26)

The May–June analyses (`train1`, `train2`, `train3`, `social_llm`, `episode_viewer`,
`eval_merged_11_06_2026`, `embedding_ablation`, `iohunter`, `aug`,
`compare_amandeep_train2`, the `covid_*`/`cp_hk_*`/`twibot20_*` studies, the
`transfer_trajectory_*` study, and the three `runs_cleaned_may*.csv` exports) were
mostly notebooks with embedded outputs. They are preserved in git, not deleted:

- branch `archive/retired-analyses-2026-07`
- tag `archive/retired-analyses-2026-07-26`

Get one back:

```bash
git checkout archive/retired-analyses-2026-07-26 -- scripts/experiments/analysis/archive/train2
```

Browse without restoring:

```bash
git ls-tree -r --name-only archive/retired-analyses-2026-07-26 scripts/experiments/analysis/archive/
```

**The branch and tag are local until pushed.** Push them before deleting or re-cloning
this working copy:

```bash
git push origin archive/retired-analyses-2026-07 archive/retired-analyses-2026-07-26
```
