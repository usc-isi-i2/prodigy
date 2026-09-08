# Social-specificity transfer pilot

## Question

Does single-source PRODIGY neighbor-matching pretraining learn representations that are specific to social-media graphs, or do they transfer to non-social citation graphs?

## Pilot design

Four single-source checkpoints were evaluated on all four source/target combinations: UKR–RUS Twitter, Facebook Page–Page, Cora, and PubMed. All checkpoints used one training seed, 2,500 updates, 2-hop sampling, 3-shot support, and fixed evaluation episodes. The social targets use 30-way episodes. Cora's held-out graph has only nine eligible distinct split-aware NM centers, so the citation targets use 5-way episodes; PubMed was also evaluated 5-way to keep the two citation columns comparable.

Raw accuracy must therefore be compared **within target columns**, not directly across social and citation columns. The companion CSV includes chance accuracy and normalized accuracy above chance, `(accuracy - chance) / (1 - chance)`, for cross-protocol orientation.

## Final accuracy matrix

Rows are pretraining sources and columns are evaluation targets.

| Pretraining source | Twitter (30-way) | Facebook (30-way) | Cora (5-way) | PubMed (5-way) |
|---|---:|---:|---:|---:|
| **Twitter** | **24.76%** | **22.27%** | **49.32%** | 55.17% |
| **Facebook** | 14.39% | 21.02% | 38.20% | 46.48% |
| **Cora** | 6.03% | 6.99% | 46.31% | 40.24% |
| **PubMed** | 6.22% | 7.65% | 38.61% | **61.23%** |
| Chance | 3.33% | 3.33% | 20.00% | 20.00% |

## ROC-AUC matrix

| Pretraining source | Twitter | Facebook | Cora | PubMed |
|---|---:|---:|---:|---:|
| **Twitter** | 0.801 | **0.819** | **0.815** | 0.843 |
| **Facebook** | 0.705 | 0.762 | 0.674 | 0.765 |
| **Cora** | 0.619 | 0.612 | 0.776 | 0.710 |
| **PubMed** | 0.585 | 0.594 | 0.695 | **0.880** |

## Findings

1. **The pilot rejects a simple social-specificity account.** Twitter pretraining is best on Twitter, Facebook, and Cora, and is second only to PubMed on PubMed. It therefore learns a representation that transfers strongly beyond social-media graphs.

2. **Transfer is sharply asymmetric.** Citation-pretrained models transfer poorly to social targets: Cora and PubMed reach only 6.0–7.6% accuracy, versus 14.4–24.8% for the social-pretrained models. The reverse direction is strong: Twitter reaches 49.3% on Cora and 55.2% on PubMed.

3. **Twitter is the strongest general-purpose source in this pilot.** It wins three of four target columns and has the highest mean normalized accuracy above chance across targets (about 30.6%). PubMed has the strongest in-domain cell but is much less portable.

4. **In-domain advantage is not universal.** PubMed→PubMed is clearly best on PubMed, but Twitter→Cora exceeds Cora→Cora by 3.0 accuracy points. Twitter→Facebook also exceeds Facebook→Facebook by 1.3 points.

5. **Facebook transfers moderately but is not the main driver.** Facebook pretraining is competitive on Facebook and PubMed, but trails Twitter on every target.

## Direction

The actionable hypothesis is not “PRODIGY learns social-media-specific structure.” It is: **the structurally richer Twitter source teaches broadly reusable neighbor-matching structure, whereas these citation sources teach narrower representations that do not extrapolate to social graphs.** Follow-up work should test which source properties explain this asymmetry—scale, degree/radius diversity, feature distribution, or exposure—not spend the next budget merely adding nominal seeds.

The highest-value confirmation is a matched-exposure or size-controlled Twitter-versus-citation comparison. One additional social graph and one additional non-social graph would then test whether the result is domain-level or specific to these datasets.

## Limitations

- One training seed per source: sufficient for a directional pilot, not a variance estimate.
- Evaluation episodes are fixed by split name in the current loaders. Repeating `--seed` does not resample episodes; the accidental duplicate social-target runs reproduced the same accuracies exactly and are not independent replicates.
- Social and citation targets use different `n_way` values because 30-way evaluation is infeasible on Cora's held-out view. Within-column rankings are valid; raw accuracy averages across all four columns are not.
- The pilot changes source graph along with graph size, topology, and feature distribution, so it identifies an asymmetric source effect rather than its mechanism.

## Artifacts

- Setup: `scripts/experiments/setup/social_specificity_pilot/`
- Tucker worktree: `/dataMeR1/phil/gfm/prodigy-socialspec`
- Training checkpoints: `state/social_specificity_pilot/`
- Evaluation logs: `log/social_specificity_pilot/eval/`
- Parsed results: `data/transfer_matrix.csv`
