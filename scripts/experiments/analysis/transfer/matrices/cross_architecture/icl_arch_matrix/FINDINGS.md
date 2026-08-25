# Findings: one-seed, 100-update architecture matrix

## Registered scope

- Architectures: PRODIGY, VISION, GILT.
- Source sets: the same 31 final-core physical sets.
- Training: seed 0, 100 optimizer updates, effective batch size 4 (400 neighbor-matching episodes).
- Evaluation: 128 fixed 2-way/10-shot binary-classification episodes on four targets; ROC-AUC primary.
- Completeness: 372/372 architecture × source-set × target cells.
- Episode audit: one support/query-node fingerprint per target across all architectures and models.

This is a descriptive low-budget system comparison, not a converged architecture ranking or a parameter-matched backbone ablation.

## Aggregate result

Mean ROC-AUC over all 124 model-target cells per architecture:

| Architecture | Mean ROC-AUC | Specialist mean | Mixture mean |
|---|---:|---:|---:|
| VISION | 0.7391 | 0.7118 | 0.7502 |
| GILT | 0.6831 | 0.6511 | 0.6963 |
| PRODIGY | 0.5840 | 0.5963 | 0.5790 |

Paired over the same 124 model-target cells, VISION minus PRODIGY averages +0.1550 ROC-AUC (positive in 71.0% of cells), GILT minus PRODIGY +0.0991 (58.1%), and GILT minus VISION -0.0559 (GILT positive in 8.9%). These cells are dependent measurements from one training seed and four fixed evaluation streams, not 124 independent replicates.

The architecture effect is target-dependent. VISION/GILT are strongest on Covid Political and Election 2020, whereas PRODIGY is strongest on TwiBot-20; all systems are near chance on Ukraine Suspended at this budget.

## Composition rule

At update 100, the best-included-specialist envelope is strong for VISION and GILT but not PRODIGY:

| Architecture | Best-specialist MAE | Pooled r | Target-demeaned r | Mean residual |
|---|---:|---:|---:|---:|
| VISION | 0.0192 | 0.997 | 0.623 | -0.0186 |
| GILT | 0.0304 | 0.993 | 0.866 | -0.0304 |
| PRODIGY | 0.1471 | 0.243 | -0.094 | -0.1386 |

Thus the primary 2,500-update PRODIGY composition result is not an architecture-free property at update 100: early-budget composition behavior differs across these systems. The later pilot contains only one source set, so it cannot establish that PRODIGY learns the full composition envelope later.

The reconstructed nine-rung trajectories for all three registered source orders are
shown in [`figures/prodigy_ladder_orders.png`](figures/prodigy_ladder_orders.png),
[`figures/vision_ladder_orders.png`](figures/vision_ladder_orders.png), and
[`figures/gilt_ladder_orders.png`](figures/gilt_ladder_orders.png). Rung 1 uses the
appropriate single-source checkpoint, rungs 2--8 use the order-specific cumulative
mixtures, and the shared `all9` checkpoint supplies rung 9.

[`figures/joint_downstream_ladder_orders.png`](figures/joint_downstream_ladder_orders.png)
places the matched PRODIGY/VISION/GILT ladders beside the separately run SAMGPT ladder.
The shared panel is for qualitative comparison only: SAMGPT uses nine evaluation targets
and a different training/evaluation protocol, whereas the other three systems share the
same four fixed downstream episode streams.

## Random-initialization control

One deterministic, untrained instance per architecture was evaluated on the exact same four episode streams. The model has no associated source set; it retains each architecture's native raw-feature transforms, random message passing, and support-label mechanism. The comparison with update 100 therefore measures the end-to-end effect of 100 pretraining updates, not gain over a raw-feature-only floor.

| Architecture | Random-init mean | Update-100 mean | Mean delta | Update-100 cells above random |
|---|---:|---:|---:|---:|
| PRODIGY | 0.4169 | 0.5840 | +0.1671 | 91.1% |
| VISION | 0.6485 | 0.7391 | +0.0906 | 73.4% |
| GILT | 0.5348 | 0.6831 | +0.1483 | 82.3% |

All three update-100 means exceed their random-init anchor. VISION is already strong at random initialization on Covid Political (0.7312) and Election 2020 (0.7849), so its early lead is partly architectural rather than wholly learned in pretraining. Ukraine Suspended is the exception: update-100 means change by only +0.0046 for PRODIGY and decrease by 0.0350/0.0050 for VISION/GILT. The aggregate is one initialization and the update-100 side averages 31 source-set models; it is not a paired checkpoint trajectory.

## Budget sensitivity

The preserved 500-update single-source (`ss_ukr_rus`) pilot uses the same four targets and evaluation fingerprints. Its architecture means are PRODIGY 0.7592, VISION 0.7578, and GILT 0.6829, versus VISION 0.6609, GILT 0.6261, and PRODIGY 0.5896 at the 100-update pilot. The rank change on one source set confirms that 100 updates is pre-saturation and prevents a claim of architecture superiority.

## Raw-feature-only controls

Two no-topology controls use only each center node's L2-normalized raw 768-dimensional text feature. Cosine prototypes are support-class means; fixed L2 logistic regression (`C=1`, `liblinear`) is fit independently on the same 20 support nodes in each episode. Neither control pretrains, tunes on query labels, aggregates neighborhoods, or fits across the target dataset. All eight control rows exactly match the trained matrix's four episode fingerprints.

| Control | Covid Political | Election 2020 | Ukraine Suspended | TwiBot-20 | Mean |
|---|---:|---:|---:|---:|---:|
| Raw cosine prototype | 0.8228 | 0.8354 | 0.5119 | 0.5609 | 0.6827 |
| Raw logistic | 0.8242 | 0.8369 | 0.5165 | 0.5542 | 0.6830 |

The two raw controls agree within 0.0002 in their four-target means. Relative to raw logistic, update-100 VISION averages +0.0561 ROC-AUC and 75.0% of its source-set/target cells are higher. GILT is effectively tied in the mean (+0.0002), although only 33.1% of its cells are higher; this reflects strong target dependence rather than uniform equivalence. PRODIGY averages -0.0989 and only 29.0% of its cells are higher. Thus the update-100 comparison does **not** establish learned graph-representation value for GILT or PRODIGY over raw text; only VISION clears this floor on average at the tested budget. These are transductive few-shot episode controls, not trained supervised target baselines.

## Topology-only degree controls

The topology controls discard node text and represent each center by three full-target-graph features: target-graph-z-scored `log1p` directed in-degree, out-degree, and total degree. Cosine prototypes and fixed L2 logistic regression are again fit only on each episode's 20 support nodes. The edge index is visible transductively, but no query labels or target-wide label fit are used. All eight rows exactly match the trained matrix's episode fingerprints.

| Control | Covid Political | Election 2020 | Ukraine Suspended | TwiBot-20 | Mean |
|---|---:|---:|---:|---:|---:|
| Degree cosine prototype | 0.6047 | 0.5034 | 0.5391 | 0.6669 | 0.5785 |
| Degree logistic | 0.5986 | 0.5275 | 0.5247 | 0.7148 | 0.5914 |

Degree logistic is the stronger topology control in the four-target mean. Relative to it, update-100 VISION averages +0.1477 ROC-AUC, GILT +0.0918, and PRODIGY -0.0073; 53.2%, 50.0%, and 26.6% of their respective source-set/target cells are higher. The cell fractions show that even the positive VISION/GILT mean gaps are target-dependent rather than uniform.

Taken together, VISION clears both the raw-text and degree floors on average. GILT clears degree but is tied with raw text, while PRODIGY clears neither stronger floor at update 100. This sharpens the result boundary: the low-budget architecture matrix establishes system differences, but it does not establish learned graph-representation value for every system. The degree control is not a trained structural encoder, shallow GNN, or label-propagation model, so it also does not exhaust topology-only alternatives.

## Target-supervised references

The target-supervised references use the repository's deterministic 60/20/20 stratified node splits. A raw-feature MLP and two-layer GraphSAGE each train for 100 updates at two fixed learning rates (`1e-3`, `3e-4`); 32 validation episodes select the learning rate, then the chosen update-100 model scores only query nodes from the unchanged 128 test episodes. Test support labels and query labels are never used for fitting or selection. These models see thousands of labeled training centers and are supervised references, not matched-label-budget competitors for 10-shot in-context learning.

| Reference | Covid Political | Election 2020 | Ukraine Suspended | TwiBot-20 | Mean |
|---|---:|---:|---:|---:|---:|
| Supervised MLP | 0.9117 | 0.9188 | 0.6125 | 0.7188 | 0.7905 |
| Supervised GraphSAGE | 0.9556 | 0.9885 | 0.5971 | 0.7955 | 0.8342 |

Relative to supervised GraphSAGE, update-100 VISION/GILT/PRODIGY average -0.0951/-0.1510/-0.2501 ROC-AUC; 7.3%/0.0%/0.0% of their source-set/target cells are higher. Against the supervised MLP, the gaps are -0.0514/-0.1073/-0.2064 and 41.9%/16.9%/1.6% of cells are higher. GraphSAGE improves the four-target mean over the MLP by 0.0437, although the MLP is better on Ukraine Suspended.

The supervised result supplies the missing conventional reference and shows substantial headroom under target labels. It does not imply that target-specific training is preferable in the intended few-shot setting: the label budgets and adaptation regimes differ by design. It does establish that the one-seed update-100 architecture audit should be presented as an early-budget in-context systems comparison, not state-of-the-art downstream performance.

## Protocol boundaries

- Parameter counts differ: PRODIGY 1.64M, VISION 2.05M, GILT 2.12M.
- PRODIGY uses directed sampled edges and raw 768D features; VISION/GILT symmetrize, VISION centers 768D features, and GILT uses one frozen seed-0 orthogonal 768→128 projection.
- Label mechanisms, losses, and optimizers remain architecture-native.
- Memory-bound adapter runs accumulate four per-episode gradients before one optimizer step, preserving the effective batch and update count.
- One seed and unequal target query counts rule out confidence intervals or a converged leaderboard interpretation.
