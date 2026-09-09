# HK-only encoder and metagraph diagnostic

Frozen HK specialist, original 200 selected HK cases, both saved support
interventions, five draws each. No training or new sampling. Uses standalone
HK feature rows and exact saved merged-ID subgraphs; never loads the merged
34.6-million-node graph.

The original architecture is instantiated with the shared `get_module_list`
and `SingleLayerGeneralGNN` constructors and the saved effective config.
Checkpoint model weights load strictly. Only the configured HK checkpoint is
used. Standalone source-local feature rows are indexed by saved merged IDs
minus the canonical source offset; pooling IDs receive zero rows. The full
reconstructed HK input hash must equal the earlier canonical audit exactly.

Encode all 210 subgraphs for each selected episode, not just the selected query:
other queries can influence the metagraph. Cache the original within-batch label
vectors (no reindexing of the learned label table), metagraph edges and roles.
One full-batch forward verifies selected-episode manual encoder and metagraph
replay within fixed tolerances of 1e-5 and 1e-4. All 200 original and 2,000
intervention decisions must reproduce the earlier native outcomes, with true
probabilities within 1e-5. Check model-state hashes before and after.

Three fixed heads are recorded:

- Native metagraph and decoder.
- Cosine to the mean of each class's three pre-metagraph support embeddings,
  normalizing the mean prototype and query. This is the primary simple head.
- Mean cosine to the three individually normalized support embeddings, matching
  the scoring rule of the raw full-neighborhood feature diagnostic.

Both simple heads use the native learned logit scale for saved probabilities,
without calibration. Top-1 comparisons are scale-independent; head-to-head NLL
differences are not a calibrated model comparison. Native NLL changes remain
valid. No prototypes, temperatures or methods are fitted on these test cases.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -u scripts/experiments/setup/nm_hk_mechanism/run.py \
  --device cuda:0 --out /dataMeR1/phil/gfm/error_audit/nm_hk_mechanism_20260908
```

Input roots, standalone feature path and device are overrideable. Use a free
owned GPU and an idle isolated worktree/tmux session. Runtime revision
`40d7b5ef` on branch `codex/nm-complete-input-audit-20260908` in Tucker worktree
`/dataMeR1/phil/gfm/prodigy-nm-complete-input-audit-20260908`.

Outputs remain private: `head_comparison_private.csv` (2,200 rows),
`embeddings_and_logits_private.pt` (selected complete pre-metagraph episodes,
all 6,000 replacement support embeddings, labels/edges/roles, case manifests,
three sets of logits and row keys), and a verification receipt. Cache logits
support subsequent loss/temperature/rank analyses without inference; cached
embeddings support additional decision-rule comparisons without re-encoding.

The aggregation helper is
`scripts/experiments/analysis/evaluation/error_audit/summarize_nm_hk_mechanism.py`.
It takes `--input-root` and `--geometry` (the private paired geometry CSV from
`nm_support_geometry`). Only aggregate results belong in git.
