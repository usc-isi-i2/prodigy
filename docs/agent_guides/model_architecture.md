# PRODIGY Model Architecture

Read this before changing model, sampler, dataloader, episode construction, or
objective plumbing.

PRODIGY is an in-context few-shot learner. One episode contains `n_way` labels times
`n_shots` support and `n_query` query center nodes. Each center expands into an
`n_hop` sampled subgraph with a pooling supernode. Sampling starts in
`experiments/sampler.py`; dataset and batching logic live in `data/dataset.py` and
`data/dataloader.py`.

`models/general_gnn.py` executes the `--layers` string, assembled in
`experiments/layers.py` and normally `S,U,M`:

- `S` message-passes over each sampled subgraph in `models/multilayer_gnn.py`.
- `U` pools the subgraph into a supernode representing one data point.
- `M` runs the attention GNN in `models/metaGNN.py` over the bipartite metagraph of
  data-point nodes and label nodes. Edge attributes carry `(is_query, +/-1 support
  label)`, so this stage is where support labels reach query representations.

Prediction uses scaled cosine similarity between query and label embeddings. The
loss is cross-entropy for `n_way > 1` and BCE or margin ranking for one-way tasks;
regression replaces this with an MLP head. Label embeddings are sentence-BERT text
embeddings unless `ignore_label_embeddings` or `zero_label_embeddings` is enabled.

Pretraining objectives such as neighbor matching, contrastive learning, masked
feature prediction, or mixtures change episode and label construction rather than
the encoder. The `fp` and `e4` objectives add an auxiliary MLP head over node
embeddings.
