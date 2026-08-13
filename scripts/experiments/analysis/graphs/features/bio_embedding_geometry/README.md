# Bio-embedding geometry across the nine social graphs

This analysis compares the GTE multilingual bio/page-description embedding stores
used by the nine social graphs. It excludes empty bios and zero vectors.

Run on Tucker in the `bio-embeddings-v001` environment:

```bash
python scripts/experiments/analysis/graphs/features/bio_embedding_geometry/analyze.py \
  --data-root /dataMeR1/phil/data \
  --output-dir scripts/experiments/analysis/graphs/features/bio_embedding_geometry/data
```

The default geometry sample is 50,000 unique non-zero embeddings per graph (all
43,659 for `ukr_rus_suspended`). The script writes the overlap, concentration,
PCA-subspace-overlap, and held-out centroid-distance tables used in `FINDINGS.md`.

The checked-in `bio_overlap_summary.csv` additionally records the union/multiplicity
totals from the same full-catalog run.

`bio_hash` overlap is computed over complete unique normalized-text catalogs. The
geometry calculations use embedding rows and give every unique bio equal weight;
they do not weight repeated bios or include zero-filled missing profiles.
