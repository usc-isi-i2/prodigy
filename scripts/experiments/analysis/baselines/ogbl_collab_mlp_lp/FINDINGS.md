# Feature-only MLP link prediction on ogbl-collab

## Evidence status

The official cosine campaign is admissible baseline evidence under the frozen
OGB temporal split and validation-only checkpoint selection. The symmetric
concatenation campaign is an exploratory comparison. The ordered-concatenation
campaign is an order-sensitivity diagnostic and is not a valid undirected
benchmark model. The test-oracle campaign is diagnostic-only because it scores
the test panel at every evaluation epoch.

All production and diagnostic runs used seeds 0--2, dataset fingerprint
`07f7af8e654bda27caad60ed74c479f48780343826543dd613d90cb4e979f9f4`,
and the official fixed negative arrays. The authoritative runtime artifacts are
retained on Tucker at `/dataMeR1/phil/gfm/ogbl_collab_mlp_lp/`.

## Results

Official test Hits@50, reported as mean and sample standard deviation across
three seeds for learned models:

| Scorer | Hits@50 (%) | Classification |
| --- | ---: | --- |
| Raw feature cosine | 26.271 | admissible deterministic baseline |
| Linear map then cosine | 34.520 +/- 0.138 | admissible baseline |
| Nonlinear MLP then cosine | 36.945 +/- 1.124 | admissible baseline |
| Symmetric linear concat | 2.231 +/- 0.005 | exploratory |
| Symmetric nonlinear concat | 22.270 +/- 0.594 | exploratory |
| Ordered linear concat | 2.247 +/- 0.012 | diagnostic-only |
| Ordered nonlinear concat | 21.040 +/- 0.694 | diagnostic-only |

Removing symmetry did not rescue concatenation: the ordered nonlinear scorer
was 1.230 percentage points below the symmetric nonlinear scorer. A linear
layer on `[x_u, x_v]` remains additive in the endpoints and cannot represent
pairwise compatibility; allowing different endpoint weights adds order
sensitivity but no interaction term.

The retrospective test-oracle diagnostic found no Hits@50 inflation for either
linear-cosine seed and none for nonlinear seeds 0 and 2. Nonlinear seed 1 could
gain 0.168 percentage points by selecting epoch 121 using test Hits@50 instead
of the validation-selected epoch 109. The mean oracle advantage was therefore
0.056 percentage points, but this is non-admissible diagnostic evidence.

Machine-readable headline results and the validation receipt are in
[`data/results.csv`](data/results.csv) and
[`data/validation_receipt.json`](data/validation_receipt.json).
