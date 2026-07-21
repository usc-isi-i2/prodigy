# Retweet NM Pretraining Discussion Notes

Question: why might neighbor-matching (NM) pretraining on a disjoint UKR/COVID merged retweet graph transfer worse than NM pretraining on a single source graph?

Working hypotheses:

1. **Episode mixture imbalance.**
   The merged graph is not an equal mixture of datasets. COVID has many more nodes than UKR, so uniform center sampling gives more COVID NM episodes before any rejection effects. If strict NM sampling rejects sparse centers, the denser component can dominate even more.

2. **Neighborhood distributions differ.**
   UKR and COVID may have different retweet degree distributions, reciprocity, bot behavior, language communities, and event dynamics. NM may be learning source-specific local graph motifs rather than a general retweet-neighborhood prior.

3. **Merged training reduces per-domain updates.**
   If the merged run uses the same number of total episodes as a single-source run, each component effectively gets fewer training examples. A fixed-compute merged result and a matched-per-source merged result answer different questions.

4. **Edge-view mismatch.**
   Comparisons are cleaner if single-source and merged NM use the same edge view, especially `temporal_history` versus `default`. Otherwise the observed effect might be graph construction/evaluation mismatch rather than pretraining strategy.

5. **Feature distribution mismatch.**
   If node features include bio/text embeddings or graph-derived features that separate datasets, the model may fit dataset-specific neighborhoods. That can help in-domain but hurt broad transfer.

Useful next discussion:

- Decide whether the paper question is fixed compute, fixed per-domain exposure, or best achievable merged pretraining.
- Decide whether merged pretraining should sample datasets proportional to size or deliberately balance datasets.
- Compare checkpoint trajectories rather than only final checkpoints; a merged model may learn slower but catch up later.
