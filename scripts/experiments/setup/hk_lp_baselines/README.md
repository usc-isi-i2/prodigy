# Hong Kong direct-link baseline pilot

This is a direct, undirected edge-prediction comparison. It is not the 30-way
neighbor-matching benchmark. Every method scores the same saved validation and
test endpoint pairs. No supports or metagraph are used at LP inference.

## Frozen protocol, before inspecting pilot outcomes

- Reuse the canonical HK 70/15/15 unordered-edge split already extracted from the
  original final-core artifact. The edge-cache SHA must match its saved receipt.
- Use only static_train for learned-model training and neighbor aggregation.
  Choose 2,000 validation positives and 2,000 test positives from their separate
  canonical views, plus one negative per positive. Negatives are absent from
  **all three** graph views and unique across train/validation/test.
- Randomly orient each positive before corruption; replace one endpoint with
  a node in its same log2 training-degree bin. Degree-zero has its own bin.
  Record infeasible positive proposals, with no uniform-negative fallback.
- MLP and plain GraphSAGE share 100,000 sampled training positives/nonedges and
  the exact batch index stream: 2,500 updates, 256 positives and their 256 matched
  negatives per update; AdamW lr .002, weight decay .001, initialization seed 0.
  Use the terminal checkpoint. No hyperparameter or checkpoint selection on test.
- Both learned LP models use cosine embeddings with a learned positive scale and
  bias for BCE training. Report the cosine ranking. Score orientation is selected
  using validation only; both signed and original-sign test results are retained.
- Primary ROC-AUC, secondary average precision on balanced pair sets. AP uses
  sklearn's tie-aware implementation. These are one-seed pilot results, not an
  independently replicated or compute-matched architecture study.

## Models

1. Raw 768-dimensional node-feature cosine.
2. Raw center + mean one-hop neighbor features: normalize each block, concatenate,
   normalize the concatenation, then cosine. Means use all undirected training
   neighbors. Empty means/zero features remain zero blocks, giving tied scores
   when no feature evidence exists; report zero-feature subsets separately.
3. Shared two-layer node MLP, 768 -> 256 -> 256, ReLU between layers, LP-trained.
4. One-hop mean GraphSAGE: PyG SAGEConv(768,256), ReLU, Linear(256,256), LP-trained.
   Since this is one hop, the raw neighbor mean can be computed once and reused;
   a synthetic gate checks equivalence to the actual PyG convolution.
5. Existing HK seed-0, step-2500 PRODIGY/NM checkpoint, frozen encoder + cosine.
   Strict weight loading, unchanged model hash. Preserve its two-hop [9,9]
   sampled-context recipe (101-node cap) and directed message-passing behavior.
   This differs from the simpler baselines' full undirected one-hop aggregation:
   the question is comparative LP utility, not an isolated encoder ablation.
6. Common neighbors, Adamic-Adar, Jaccard, preferential attachment, using the
   existing pair-LP evaluator's implementations on the training adjacency.

PRODIGY's original NM objective, source exposure and training compute are not
matched to the LP-trained models. No conclusion about the metagraph follows
from this encoder-only LP comparison. Existing historical LP numbers use another
pair protocol and must not be mixed with this pilot table.

## Execution

Use an isolated worktree and free owned GPU. Example after validating the code:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python -m unittest scripts.experiments.setup.hk_lp_baselines.test_protocol
python scripts/experiments/setup/hk_lp_baselines/run.py --self-test
python scripts/experiments/setup/hk_lp_baselines/run.py --dry-run --out /dataMeR1/phil/gfm/lp_baselines/hk_20260908
python -u scripts/experiments/setup/hk_lp_baselines/run.py --device cuda:0 --out /dataMeR1/phil/gfm/lp_baselines/hk_20260908
```

The output directory must be new. Save the protocol before training, all pair
sets, learned checkpoints, sampled PRODIGY neighborhoods without feature copies,
endpoint embeddings, every pair score, training losses, parameter/input hashes,
permutation controls, and aggregate results. Private node-level files remain on
Tucker. Setup, pair preparation, training and final scoring times are separate.
