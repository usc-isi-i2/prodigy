# PinSAGE final-core transfer

## Evidence contract

Four seed-0 PinSAGE models completed 2,500 training steps. The admissible evaluation
is the corrected fixed-test run at `c4e19eb180`: 4 models by 4 targets, 512 episodes
per cell, split `test`, training edge view `static_train`, target edge view
`static_test`, and metric contract `accuracy_f1_macro_roc_auc_ovr_macro_v1`.
All 16 result JSON files and the four training result/config pairs are committed
under `data/`.

Earlier `eval_nm_*` files labeled `step0` used the pre-fix evaluation path and give
radically inconsistent accuracies. They are rejected and intentionally excluded.
This is neighbor-matching classification, not the historically invalid episodic
static-link-prediction evaluator.

## Corrected fixed-test results

Values are accuracy / macro ROC-AUC.

| model | COVID Political | Election 2020 | TwiBot-20 | Ukraine/Russia Suspended |
|---|---:|---:|---:|---:|
| PinSAGE COVID | .143245 / .832812 | .117513 / .804719 | .290169 / .893033 | .162386 / .807280 |
| PinSAGE Election | .122184 / .794455 | .231917 / .886771 | .134163 / .767266 | .084440 / .712241 |
| PinSAGE LOO TwiBot-20 | .161377 / .812544 | .138493 / .819411 | .266829 / .870413 | .273617 / .861198 |
| PinSAGE TwiBot-20 | .116520 / .791792 | .114681 / .797639 | .319531 / .905473 | .113509 / .765624 |

Ranking signal is substantial (macro ROC-AUC .712--.905), while argmax accuracy is
very poor (.084--.320). Under this fixed readout, PinSAGE therefore exhibits a
ranking/decision mismatch and does not provide a usable calibrated classifier.
The result is limited to one training seed and the fixed episode panel.
