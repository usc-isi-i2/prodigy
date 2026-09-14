# Matched GraphSAGE-cosine on ogbl-collab

This campaign isolates neighborhood aggregation against the completed nonlinear
feature-only MLP. It uses the same official temporal split, supplied 128-D node
features, scaled-cosine decoder, deterministic 1:1 negative stream, optimizer,
update count, validation selection rule, three seeds, and test-once policy.

The only intended representational change is two mean-GraphSAGE layers. Message
passing uses the unique, unweighted, undirected official training pairs. It excludes
validation edges, test edges, edge weights, years, node IDs, and repeated-event
multiplicity. Training positives remain in the message-passing graph, as in standard
transductive link prediction.

This is a matched topology ablation, not the official example's stronger and more
confounded three-layer GraphSAGE plus three-layer edge MLP. That recipe should be a
separately named arm if we run it later.

The frozen contract is [`protocol.yaml`](protocol.yaml). Dry-run first:

```bash
python scripts/experiments/setup/ogbl_collab_sage_lp/run.py \
  --dataset-root /dataMeR1/phil/data/ogb \
  --out /tmp/not-created \
  --seed 0 --device cuda:0 --dry-run
```

Production uses one dedicated Tucker tmux session per seed on GPUs 0-2:

```bash
bash scripts/experiments/setup/ogbl_collab_sage_lp/run_tucker.sh 0 0
bash scripts/experiments/setup/ogbl_collab_sage_lp/run_tucker.sh 1 1
bash scripts/experiments/setup/ogbl_collab_sage_lp/run_tucker.sh 2 2
```

Outputs live under `/dataMeR1/phil/gfm/ogbl_collab_sage_lp/matched_v1/` and W&B
project `ogbl-collab-sage-lp`. Aggregate only after all three result files validate:

Each seed stores atomic `checkpoints/best.pt` and `checkpoints/last.pt` files. The
last checkpoint includes optimizer and early-stopping state; both checkpoint hashes
are recorded in `selection_frozen.json`. Per-epoch W&B metrics include training loss,
validation Hits@10/50/100, ROC-AUC, AP, score distribution summaries, gradient and
parameter norms, learned score scale/bias, epoch time, throughput, learning rate, and
peak GPU memory. Final test summaries include the same predictive metrics plus the
predeclared repeat-versus-novel strata.

```bash
python scripts/experiments/setup/ogbl_collab_sage_lp/aggregate.py \
  --root /dataMeR1/phil/gfm/ogbl_collab_sage_lp/matched_v1 \
  --out /dataMeR1/phil/gfm/ogbl_collab_sage_lp/matched_v1/aggregate.json
```
