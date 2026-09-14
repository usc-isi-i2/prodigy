# ogbl-collab top-method audit and sub-1M decision plan

## Decision

Do not spend the next run on another compact-GNN architecture sweep. The released
evidence points to a cheaper, more discriminating question: **can the published
PLNLP/GIDN training recipe retain most of its score after replacing its 256-dimensional
learned node-ID table with deterministic graph-propagated features and a small learned
projection?**

This is the binding compression question. The node tables account for about 60.38M of
GIDN@YITU's reported 60.45M parameters and essentially all of PLNLP+SIGN's reported
34.98M parameters. The ranking loss, recent-year filtering, validation-edge reuse,
random-walk target augmentation, and dot predictor are inexpensive. HyperFusion's
released fusion operation has no learned parameters, but depends on unavailable base
predictions and uses labeled test score partitions to construct its fusion matrix.

No substantial compute was launched for this audit.

## Source snapshot

Inspected 2026-09-14:

- HyperFusion repository revision `e51de11769680b1050443fd0c55d40f1c385053d`:
  <https://github.com/zhangxwww/HyperFusion>
- GIDN@YITU repository revision `3882c6aa3e58a7fc2101db5c494177fd5a7b9301`:
  <https://github.com/wizcox98/ogbl>
- PLNLP+SIGN repository revision `d32fb6f30fd269d028bc5242b64b6cf90e3c9123`:
  <https://github.com/yao8839836/ogb_report>
- PLNLP paper: <https://arxiv.org/abs/2112.02936>
- GIDN paper: <https://arxiv.org/abs/2210.01301>
- OGB leaderboard: <https://ogb.stanford.edu/docs/leader_linkprop/>

None of the three repositories contains the collab prediction arrays, checkpoints, or
logs needed to replay the reported leaderboard result. HyperFusion's README explicitly
requires the user to prepare per-run validation/test positive/negative scores for the
three base models.

## Verified code behavior versus reported result

| Method | Reported test Hits@50 | Reported parameters | Verified released behavior | What is not established |
| --- | ---: | ---: | --- | --- |
| HyperFusion | 71.29 +/- 0.18% | 1,064,446,212 | Loads AGDN, E2N, and PLNLP predictions for ten runs. It computes separate cosine-distance matrices for validation positives, validation negatives, test positives, and test negatives; thresholded similarities populate four incidence columns, and `A = H H^T` is then used to mix all scores. | Base provenance, base scores, their training protocols, and an exact replay are unavailable. The code establishes test-label-dependent post-processing, not that any base model inserted test edges into adjacency. The paper's statement that no test hyperparameter tuning occurred does not remove this dependence. |
| GIDN@YITU | 70.96 +/- 0.55% | 60,449,025 | Released command uses `--model agdn`, one 256-wide layer, `K=1`, no supplied node features, a learned 256-D ID embedding, year >= 2010, train+validation targets and test-time graph, length-5 random-walk augmentation, strict global negatives, WeightedHingeAUC, and dot scoring. Test is evaluated every five epochs. Final logger selects the highest-validation epoch, while an additional live `Best Val/Test` display tracks the best test observed. | There is no released checkpoint/log or matched ablation isolating the claimed GIDN architecture. The command literally invokes the AGDN class; the paper does not explain this discrepancy. Code inspection cannot establish which run or selection produced the submitted scores. |
| PLNLP+SIGN | 70.87 +/- 0.33% | 34,980,864 | README command uses year >= 2010, train+validation graph/targets, length-10 random-walk augmentation, WeightedHingeAUC, one-layer SAGE, dot scoring, and repeated validation/test evaluation. `eval_last_best` selects the latest epoch tied for maximum validation. | The command omits `--use_node_feats=True`; its default is false. Although `[X, AX, A2X]` is computed, the released model then uses only a learned node-ID table. Thus the default code path does not establish that SIGN caused the reported +0.41-point gain. No prediction artifacts or matched SIGN ablation are included. |

The OGB leaderboard's parameter entries agree with the code's dominant learned tables.
For full GIDN, `235,868 * 256 = 60,382,208` ID-embedding scalars before the small
encoder. PLNLP+SIGN reindexes to the post-2010 subgraph and allocates 256 learned
scalars per retained node, explaining the roughly 35M count. Frozen or pretrained
learned embeddings would still count under this project's budget.

## Ingredient evidence ledger

| Ingredient | Enabled in released top recipes? | Evidence it contributes |
| --- | --- | --- |
| Recent-year training (`year >= 2010`) | Yes, GIDN and PLNLP+SIGN | **Not isolated** in these releases. It is a plausible recurrence/recency prior, not a verified causal gain. |
| Train+validation edges as graph and training positives | Yes, GIDN and PLNLP+SIGN | **Not isolated** here. It is benchmark-permitted for final training, but produces inflated validation metrics because validation positives are training targets. It does not imply 2019 test positives are graph inputs. |
| Random-walk augmentation | Yes: length 5 in GIDN; length 10 in PLNLP+SIGN | **Suggestive, not isolated.** PLNLP's headline with augmentation is 70.59%, while its no-random-walk framework ablation reports 68.72%; that comparison also changes other study context and is not a clean RW-only ablation. |
| Ranking loss | Yes, WeightedHingeAUC in both | **Best published causal evidence.** PLNLP reports 68.72% for pairwise learning versus 64.97% for classification with the architecture, negatives, and sample count held fixed and RW disabled. This is a paper result, not independently reproduced here. |
| Learned node-ID table | Yes, dominant input in both released commands | **Verified present and dominant in parameter count; contribution not isolated.** It is the main obstacle to the sub-1M constraint. |
| SIGN propagated features | Computed in PLNLP+SIGN | **Not established by the default command**, because `use_node_feats` remains false. The report claims 70.46 -> 70.87%, but released behavior does not mechanically support attribution. |
| Complementary predictors | Yes, HyperFusion mixes AGDN, E2N, PLNLP | **Outcome reported, mechanism replay unavailable.** Existing local compact-joint evidence independently supports real but modest AA/neural complementarity and shows negative-tail promotion is the limiting issue. |
| Test-dependent adaptation | Yes, HyperFusion | **Directly verified.** Test-positive and test-negative partitions each affect `H`, hence `A`. GIDN and PLNLP repeatedly expose test metrics during training, but their final logger is validation-selected; manual tuning/submission selection remains unknown. |

## Cheapest discriminating campaign

Freeze one seed and the released PLNLP/GIDN data recipe: year >= 2010, train+2018
validation positives as training targets and graph edges, no 2019 positives in the
graph, strict-global negatives, dot prediction, and a fixed evaluation schedule. Use
the same sampled walks and negatives across matched arms.

Stage 1 is two arms only:

1. `ID256`: code-faithful 256-D learned node table, one 256-wide encoder,
   WeightedHingeAUC, random-walk augmentation. This arm is over budget but is the
   reproduction control.
2. `SIGN128`: no node-ID table; deterministic `[X, A X, A^2 X]`, a learned `384 -> 128`
   projection, one 128-wide graph layer, the same loss, walks, targets, and dot scorer.

Run a smoke first, then a short fixed checkpoint trajectory. Stop if the reproduction
control is materially below the released neighborhood (the implementation/protocol is
not reconstructed), or if `SIGN128` trails `ID256` by more than 1.5 Hits@50 points at
two successive mature checkpoints. Continue to full training only if the compact arm
is within 1.5 points. The threshold is decision-relevant: a 70-ish compact backbone
still has a plausible fusion path; a much larger gap does not.

Only after `SIGN128` passes, run two leave-one-out arms on the compact model:

- replace WeightedHingeAUC with CE;
- disable random-walk augmentation.

This staged three-additional-arm design separates compression feasibility first, then
tests the two ingredients with the strongest prior evidence. A full factorial is not
justified initially. Test-informed selection is allowed by the user but must be logged
as such; report validation-selected and test-oracle outcomes separately.

## Concrete sub-1M route

The primary candidate is:

- deterministic weighted graph inputs through 2018 and deterministic raw/SIGN
  features: zero learned parameters;
- `384 -> 128` projection: 49,280 parameters;
- one 128-wide mean-SAGE layer: about 32,896 parameters;
- dot scorer: zero parameters;
- existing compact joint scorer and its calibration: 496,448 parameters;
- AA-DC and small threshold-aware fusion/gating parameters: negligible relative to
  the budget, but counted explicitly.

The nominal learned total before a gate is therefore about **578,624**, leaving more
than 421k headroom. The intended final scorer is not blind scalar max fusion. Train a
small pairwise gate on validation/test-informed development data to rescue AA-zero
positives while penalizing candidates that enter the top-50 negative tail. Inputs may
include AA-DC, compact-joint score, compact PLNLP score, degree/recency, repeat-edge
status, and self-pair status. All components and fitted scalars count toward 1M.

Promotion rule: proceed to the fusion stage only if compact PLNLP reaches at least
about 70% or demonstrates complementary hit coverage whose realizable fused score—not
an OR ceiling—exceeds 71.29 on the disclosed development panel. Require exact OGB
replay, full parameter accounting, pair fingerprints, graph-year audit, and an explicit
check that no 2019 target edge enters adjacency.

## Existing local evidence boundary

The compact-joint work already establishes that a 496,448-parameter AA+neural family
reaches 69.4288% mean and 69.5590% best individual under fixed-grid test tuning, still
1.731 points short. It also shows why naive fusion fails: recovered positives are
partly canceled when extreme neural negatives raise the Hits@50 cutoff. Training on
2018 produces 82.54%, but 99,984/100,000 official test negatives were reused as
training negatives, so that is disclosed test-negative exposure rather than evidence
for recency. The proposed route therefore targets the missing mechanism—a strong
ranking-trained backbone plus explicit negative-tail control—rather than another
variation of the same compact scorer.

Authoritative local evidence:
`/Users/philipp/projects/gfm/prodigy-collab-joint/scripts/experiments/analysis/baselines/ogbl_collab_compact_joint/FINDINGS.md`

## Resource state at audit time

Tucker was inspected read-only on 2026-09-14. Owned GPUs 0-3 were idle; GPUs 4-5
were occupied by unrelated VLLM processes and remain out of scope. The separate
`/dataMeR1/phil/gfm/prodigy-collab-hf-oracle` worktree existed at detached revision
`1bb326c39c7848a48122faf9d0d2d2ca8f31b1c5` and was not touched. No tmux sessions
were listed. Recheck all state with the user before any substantial launch.
