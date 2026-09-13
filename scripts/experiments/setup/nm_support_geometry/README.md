# Paired support changes and complete-input geometry

Read-only CPU analysis of the saved five-draw support replacement/context
intervention. No new episodes, support draws, model forwards, or training.

Uses the complete-input archives from `nm_complete_input_audit` and saved
support graphs/predictions from `nm_support_resampling`. For every selected case:

1. Verify compact archive SHA-256 hashes against the completed input receipt.
2. Recover the fixed query and all 90 original support feature summaries from
   saved global IDs and the specified backing feature artifact.
3. Verify baseline geometry against the previous independently computed table.
4. Verify every intervention support ID against its case manifest, and every
   saved real-node feature tensor against the backing graph.
5. Replace only the true class's three summaries. The query and largest competing
   class score stay fixed. Join to both models' existing trial predictions.

Full-neighborhood mean cosine and exact node-set Jaccard are computed using the
same definitions as the complete-input audit. Input hashes use exact equality;
independently recomputed float32 summary scores allow absolute error below 1e-6.

Runtime revision `2f4df077` on the isolated Tucker worktree
`/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`, branch
`codex/nm-complete-input-audit-20260908`. Code moved through private Git. The
initial attempt used unsupported memory-mapped loading in Tucker's PyTorch and
produced no results. The successful version uses its standard CPU loader;
the full feature artifact occupies host memory, but no graph adjacency is built.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -u scripts/experiments/setup/nm_support_geometry/run.py \
  --out /dataMeR1/phil/gfm/error_audit/nm_support_geometry_20260908_v2
```

`--inputs` and `--intervention` override private input roots. The output directory
must not exist. Use an idle isolated worktree and tmux. Output is an 8,000-row
private joined CSV and a verification receipt. Only aggregates belong in git.

Local aggregation:

```bash
/opt/homebrew/bin/python3.11 \
  scripts/experiments/analysis/evaluation/error_audit/summarize_nm_support_geometry.py \
  --input-root /path/to/downloaded/private/output
```

Cosine conclusions use paired-valid comparisons (original query and all support
summaries, and replacement summaries, nonzero). Jaccard covers all cases.
Five draws per case are repeated observations, not independent examples.
