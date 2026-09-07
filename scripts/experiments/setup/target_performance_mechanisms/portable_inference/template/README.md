# Graph-role inference reproduction candidate

This package runs the studied PRODIGY S,U,M inference implementation and its
query/support topology, message-content and suppression-dose controls without
the research repository, a graph catalog, a training launcher or network access.
It is a local release candidate, not a claim of public availability or a license
grant. No private data or weights are included in this code archive.

## Quick check

The local synthetic check uses Python 3.10 with PyTorch 2.0.1,
torch-geometric 2.3.1 and torch-scatter 2.1.2 (`requirements.txt`). The historical
real-data replay uses Python 3.11.15, PyTorch 2.0.1+cu118,
torch-scatter 2.1.2+pt20cu118 and scikit-learn 1.9.0
(`requirements-reference.txt`), with the same PyG and NumPy versions.
Install PyTorch and the matching torch-scatter wheel for your CPU/CUDA platform
before the remaining requirements. CPU inference is used regardless of GPU
availability. Do not upgrade the graph runtime for a numerical replication:
the actual SAGE operator in this evaluated version sums messages despite the
configuration requesting mean. The package checks and preserves that behavior;
actual mean is an explicit inference intervention, not a silent correction.

From a freshly extracted package:

```bash
python -m unittest test_contracts
python -m graph_role.run --inputs /path/to/authorized-tensor-inputs --output /path/to/new-output
```

The first command requires only this code and dependencies. It checks 45
conditions on synthetic episodes, direct versus factored role intervention,
query-label invariance, degree-preserving swaps, nested masks, tensor round trips,
state/operator restoration and file-integrity/path contracts. Synthetic tests
are implementation tests, not evidence of performance on real targets.

The second command evaluates every model and dataset in `inputs.json`. Optional
`--max-batches`, `--model-ids` and `--dataset-ids` create explicitly partial
replays. Output paths must be new. `--threads` defaults to four. The code never
trains, chooses a checkpoint or intervention using query outcomes, uploads data,
or reads the full source graph. No GPU is needed.

## Input schema and scope

`inputs.json` declares models and datasets, each with a relative path and SHA-256.
Models contain `id`, `source`, `seed`, `step`, `path`, `sha256`, `tensor_sha256`
and `config`. Configuration supplies input/label feature dimensions and every
key in `graph_role.model.PARAM_KEYS`. Model files are tensor state dictionaries;
loading is strict. The schema can load any compatible saved checkpoint, but
only a declared completed replay demonstrates a particular checkpoint panel.

Datasets contain `id`, `target`, `stream`, `use_global`, an ordered `batches`
list, and optional per-model `references`. A batch file is a tensor-only record
with `format=1`, `graph` and `arguments`. See `graph_role.io.GRAPH_FIELDS` and
`pack_batch`. Model arguments are the label features, one-hot labels, metagraph
edges/attributes, query-role mask, sequence tensors and optional task mask.
Graphs retain node features, disjoint subgraph/pooling structure and episode
class mappings. Account IDs are replaced by local indices; virtual nodes remain
marked -1. Tensor features can still be sensitive and are not anonymized merely
by removing IDs. Only load authorized inputs; do not publish them by default.
All runtime loads use `weights_only=True`, not arbitrary object-pickle loading.

Each batch produces 19 role/topology conditions (three paired rewiring draws),
15 message controls (five replacements times three roles), nine partial support
doses (three fractions times three draws), and raw/encoder prototype readouts.
Complete support removal is the topology endpoint, not a duplicated dose.
Masks use the exact target/stream/batch-index seed schedule of the source replay.
The complete reference panel has 43 graph-inference conditions; prototype
readouts are additional diagnostics, not claimed reference replications.

Labels enter scoring and support-conditioned inference; query label values do
not enter predictions. Scoring preserves the source protocol: global binary
orientation when `use_global=true`, otherwise pooled episode-local binary class
pairs. AUC, accuracy, F1 and NLL are reported. This is not the original TwiBot-20
follow-graph benchmark, an end-to-end training pipeline, graph construction from
raw records, or automatic selection of the best rule.

## Outputs and verification

The runner writes metrics, per-query logits, frozen-input receipts, the exact
runtime environment and a terminal `DONE.json`. When historical references are
supplied, every declared reference logit is checked (absolute tolerance 1e-5),
and full-stream metrics must agree within 1e-6. References are used only after
prediction. Missing references are explicit and do not count as successful
historical reproduction. A partial replay never becomes a full-study result.

`manifest.json` hashes the code files. `SOURCE_PROVENANCE.json` identifies exact
source files/symbols and transformations. Model implementations are copied with
only import-namespace changes; selected diagnostic functions are copied verbatim
with minimal imports. The explicit constructor and tensor schema are adapters.
No training code, logging service or cluster path is required at runtime.

Third-party attribution and release scope are in `NOTICE.md`. Input availability,
redistribution rights and anonymous/public hosting remain separate from a
successful local execution. The aggregate-results companion independently
rebuilds the manuscript tables and figures; it is a different package.
