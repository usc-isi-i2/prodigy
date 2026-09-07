# Frozen citation readout breadth test

Cora is a pre-nominated new-domain episodic node-classification test of two
social-pretrained native-objective checkpoints (blocked/interleaved, step2500).
It is not a standard Cora split benchmark: the existing loader makes custom
stratified60/20/20 node pools; support/query centers are sampled without overlap
within each test episode, from the test pool. The full graph remains available
as unlabeled context, so graph access is transductive.

Protocol:7-way,3-shot,4-query-per-class,32 batches of4 episodes, seed offset400009
(native test448 →400457). This offset was chosen before citation outcomes.
No Cora pretraining, checkpoint selection, query-label fitting, or readout tuning.
The sources are ukraine, COVID, midterm, and COVID-political. Input compatibility
must pass strict checkpoint loading. All forwards use model.eval(), matching
the social baseline. Native/traced parity is checked on every batch at absolute
tolerance1e-5, zero relative tolerance.

Compare native inference with raw-center and actual U1 representations, each
decoded by fixed ridgeλ1, scale1, no intercept, with none/support centering before
row-L2 normalization. Metrics are calculated per episode (accuracy, macro-F1,
OVR-AUC, NLL), never pooling arbitrary local class columns. Cached input tensors,
class mapping, identities, predictions, embeddings, and SHA256 receipts are saved.

Run `python -m scripts.experiments.setup.citation_readout_breadth.run --blocked
<absolute checkpoint> --interleaved <absolute checkpoint> --output <new directory>`
for a dry run. Add `--execute --gpu 3` only after reviewing the plan and checking
GPU ownership. Use an idle dedicated worktree. No training is performed.
