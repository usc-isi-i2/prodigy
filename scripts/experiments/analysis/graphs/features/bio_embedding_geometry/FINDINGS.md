# Findings: geometry of bio embeddings across nine graphs

## Scope and method

All results use the same `Alibaba-NLP/gte-multilingual-base` revision and
`bio-text-v001` normalization. Empty bios and zero vectors are excluded. Facebook
contains page descriptions rather than Twitter user bios and is reported separately.

Exact overlap uses the complete unique normalized-bio catalogs. Geometry uses up to
50,000 unique embeddings per graph, with a fixed seed (`20260807`); all 43,659
available unique non-zero embeddings are used for Ukraine-suspended.

## Exact normalized-bio overlap

- 40,776,170 distinct non-empty descriptions across all nine stores.
- 2,545,745 (6.24%) occur in at least two stores.
- Across the eight Twitter stores, 2,544,308 of 40,638,978 (6.26%) occur in at
  least two stores.
- No description occurs in all nine; the maximum is seven stores.
- The largest pair is Ukraine–COVID: 2,271,751 shared descriptions (19.22% of
  Ukraine and 7.45% of COVID).
- Ukraine-suspended is almost contained in Ukraine: 43,645 of 43,659 unique bios
  (99.97%).

These are identical normalized texts and therefore identical embeddings under the
shared model revision. They are not necessarily identical accounts; generic bios can
belong to unrelated users. See `data/pairwise_bio_overlap.csv`.

## Concentration and dimensionality

Election2020-political is the tightest cloud (mean random-pair cosine 0.590;
effective dimension 82.6). COVID-political and Midterm are also relatively
concentrated. Ukraine and COVID are broader and occupy the most directions
(effective dimensions 132.3 and 134.1). Facebook descriptions have the lowest
random-pair cosine (0.473), hence the greatest angular spread.

The effective PCA subspaces overlap far beyond random expectation: pairwise soft
intersection covers 73.7–95.0% of the smaller subspace. Ukraine–COVID shares 125.3
soft dimensions (94.9% of the smaller subspace), versus 22.9 expected for random
subspaces of the same ranks. See `data/concentration.csv` and
`data/subspace_overlap.csv`.

## Held-out centroid geometry

Centroid cosine distances were estimated in the original normalized 768-dimensional
GTE space, not in the separation-optimized 3D display. Each graph used 50,000 vectors
except Ukraine-suspended (43,659), split into two independent halves.

- The two distance matrices correlate at 0.99992.
- Mean absolute split difference is 0.00025; maximum difference is 0.00101.
- Stable components are Ukraine+COVID+TwiBot-20, Midterm+Ukraine-suspended,
  COVID-political+Election2020-political, Facebook, and Hong Kong.
- Forcing five components into exactly four is unstable: one half merges the first
  two Twitter components, while the other merges Hong Kong into the first component.

Thus the distances are well converged, but four clusters are not uniquely supported.
Five components are the more faithful summary. See `data/centroid_cosine_distance.csv`.
