# Mixture Scaling

Controlled GraphSAGE experiments measuring how the scale and diversity of a graph
pretraining mixture affect downstream node classification.

## Research questions

1. How does downstream performance change with pretraining-mixture scale and diversity?
2. How do those relationships change with model scale?
3. Can the resulting scaling relationships select a compute-efficient mixture
   that beats naive choices?

PRODIGY is out of scope. Every experimental row uses the same plain GraphSAGE
architecture and native edge negative-sampling SSL objective.

## First pilot

`configs/twibot20_sage_ssl_pilot.yaml` specifies a plain GraphSAGE random-walk SSL
convergence run on TwiBot-20. It deliberately contains no PRODIGY pooling,
metagraph, label embeddings, or episodic classification machinery.

The pilot is centered on the existing 2.5k evidence. It saves checkpoints at 100,
300, 600, 900, 1.5k, 2.5k, 5k, and 10k updates.
Convergence is selected only from a fixed validation stream; test metrics are read
once after selection.

## Repository layout

```text
configs/       versioned experiment definitions
src/           model, SSL objective, training, and evaluation code
tests/         protocol and implementation checks
results/       derived tables and figures (not raw checkpoints)
```

Large graph artifacts, checkpoints, and logs stay on Tucker under `/dataMeR1`.

Start with [FINDINGS_INDEX.md](FINDINGS_INDEX.md) for the consolidated GraphSAGE,
node-MLP, failure-analysis, intervention, and preservation record. Historical
worktree narratives are retained for provenance but are not duplicate canonical
results.

Verified aggregate tables and the current RQ1 interpretation live in
`results/RESULTS.md`. The later strict 70/15/15 study, including frozen probes,
supervised and logistic baselines, structural features, and the TwiBot-20 and
UKR/RUS findings, is documented in `results/STRICT_STUDY.md`. Rebuild the original
ladder-derived tables from the four aggregate inputs with:

```bash
PYTHONPATH=src python -m mixture_scaling.analyze --root .
PYTHONPATH=src python -m mixture_scaling.plot_results --root .
```

## Primary overnight study

Seven seed-0 single-source specialists and seven leave-one-target-out six-source
mixtures are trained with uniform source rotation. For each labeled target, the
target specialist and its target-held-out mixture are evaluated at steps 100, 300,
900, and 2,500 with the same ten labeled nodes per class. The full cross-source
matrix and additional seeds follow only after this primary gate is complete.

On Tucker, `scripts/pipeline_tucker.sh` waits for GPUs 2–3, builds missing citation
artifacts non-destructively, runs tests and a two-step smoke, then executes the primary
training/evaluation/aggregation sequence. Every stage stops the pipeline on failure.

## Follow-up sweeps

After the primary gate, `scripts/pipeline_followups_tucker.sh` runs three declared
extensions on GPUs 2–3: the complete 14-model by 7-target seed-0 transfer matrix;
the held-out mixture ladder at sizes 2–5; and full primary replications at seeds 1
and 2. The ladder order is the graph order in `configs/graphs.yaml` with the held-out
target removed. Its size-1 and size-6 endpoints reuse the primary specialist and
leave-one-out models, so only 18 distinct intermediate mixtures require new training.

## Strict TwiBot pilot

`configs/twibot_strict_pilot.yaml` defines the replacement protocol. Every graph has
a permanent stratified 70/15/15 node split. SSL gradients use induced training
subgraphs, and convergence is selected on a fixed 10% of training-graph edges removed
from message passing and gradient updates. Validation-node and test-node subgraphs
remain untouched by SSL; test is used only for one final downstream evaluation. The
six-source mixture rotates
uniformly across sources and confines every positive, negative, and minibatch to one
source graph.

The pilot compares scratch, a TwiBot specialist, UKR/RUS, Cora, and Facebook
single-source initializations, and an all-non-target mixture. Each initialization is
evaluated as a frozen linear probe and after supervised TwiBot fine-tuning. Both head
selection and fine-tuning early stopping use TwiBot validation nodes; test nodes are
scored once after selection. Run it on Tucker with
`scripts/twibot_strict_pilot_tucker.sh`.
