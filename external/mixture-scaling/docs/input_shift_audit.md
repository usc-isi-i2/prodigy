# Input distribution audit

Runs on eight current nonzero mini graph views, excluding Election. All input
caches are read-only and their graph identities and context receipts must match.
There is no checkpoint fitting, selection, or reading of validation/test outcomes.

Run from an isolated Tucker worktree in the prodigy environment:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u scripts/audit_input_shift.py --output /dataMeR1/phil/gfm/mixture-scaling/results/input_shift_audit_20260912
```

Uniform-node views: 20,000 distinct nodes per graph, raw 768-dimensional embeddings
and concatenation with the exact cached up-to-ten context-neighbor mean (1536).
Distances: 128 fixed random unit projections, empirical 1-Wasserstein averaged over
projections after dividing each projection by its equal-graph pooled standard
deviation. Cross-graph comparisons use 10,000 nodes per graph; independent halves
within each graph provide an equal-sample-size reference. These references quantify
sample variation for these fixed graphs, not uncertainty over independent graphs.
Scale is shared across graph pairs within each view; values across views should
not be interpreted as the additional causal effect of neighbor features.

Domain classifier: standardized linear logistic SGD, fixed alpha=.001, averaged
weights, up to 300 epochs, fixed seed; 3,000 training and 2,000 held-out center nodes
per graph. Scaling uses training inputs only. Within-graph pseudo-domains are a
negative control. Center IDs are disjoint between train and evaluation, but context
neighbors can overlap: this measures discrimination within fixed graphs, not
generalization to independent graphs. AUC measures this classifier's ability to
separate graph identity; chance AUC is not proof of equal distributions or overlap.

Pair view: 10,000 supervised positive edges sampled with replacement and 50,000
uniform nonedges rejecting self-pairs and all known edges. No held-out positive
edges are used. Compare 30,000-pair halves using symmetric [endpoint mean,
absolute endpoint difference] features, then random projections as above, for both
node representations. This is a diagnostic summary of the task input distribution,
not the MLP's exact pair representation: absolute differences lose cross-coordinate
endpoint alignment. Node and pair sampling measures differ by design.

Connectivity: raw-embedding cosine on those edges/nonedges; report means,
histograms, KS and cosine-as-edge-score AUC. AUC is a descriptive discrimination
measure under this sampled background, not a fitted model result. Shared 24-cluster
MiniBatchKMeans fits L2-normalized raw embeddings (2,000 nodes per graph). Count
undirected supervision-edge cluster mixing; expected mixing is the product of
observed endpoint cluster marginals (configuration expectation), correcting for
cluster edge volume, not a simple-graph degree-preserving rewiring simulation.
Positive-edge distributions involve connectivity, not solely marginal node p(x).

Raw sampled IDs, features, edges, projection matrices and cluster centers stay in
the remote raw/ directory. Completion writes audit.json plus COMPLETE.json.
Completed roots skip; partial roots refuse to run. No existing graph/cache/run is
replaced. Figures and the compact JSON are copied into the analysis result folder.

Local rendering (Homebrew Python 3.11):

```bash
MPLCONFIGDIR=/tmp/mlp-mpl-cache /opt/homebrew/bin/python3.11 scripts/plot_input_shift.py --input results/input_shift_audit/data/audit.json --output results/input_shift_audit
```

`scripts/compare_input_shift_transfer.py` optionally overlays the already-completed
interleaving and singleton test tables. It does not rerun evaluation or tune the
audit. The baseline is the better of the two constituent singleton means on the
six other active graphs, not a per-target oracle. Pairwise Spearman correlations
are descriptive: graph pairs share sources and different pairs omit different
targets. No independence-based confidence intervals or p-values are reported.
