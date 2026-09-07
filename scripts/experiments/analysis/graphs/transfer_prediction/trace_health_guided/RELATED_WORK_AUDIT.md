# Contribution audit against nearby work

Checked 2026-09-07. This is a bounded literature audit, not a systematic review
or a claim that no prior work contains the proposed mechanism. Statements below
distinguish inspected primary sources from our interpretation.

## What cannot be the novelty claim

| Prior work | Verified overlap | Consequence for our claim |
| --- | --- | --- |
| [Better with Less / APT](https://arxiv.org/abs/2311.01038v2), Xu et al., 2023 | Its abstract explicitly reports that more graph pretraining data can hurt, then proposes iterative selection using graph properties and predictive uncertainty. | More graphs is not always better is motivation, not our novel contribution. A new selector must beat appropriate alternatives; our tested agreement allocation did not beat uniform. |
| [Graph Few-Shot Learning via Knowledge Transfer](https://ojs.aaai.org/index.php/AAAI/article/view/6142), Yao et al., AAAI 2020 | Uses transferable node embeddings and graph-specific prototype embeddings. | Separating representation learning from prototype construction is established. We need an explanation of failure, not the observation that both computations exist. |
| [GILT](https://arxiv.org/html/2510.04567v3), Ma et al., version 3, 2026 | Sections 3.2–3.3 use asymmetric support/query tokens, support self-attention followed by query-to-support cross-attention, and a support-derived prototype head. | Role-aware processing is not a new architecture category. A generic asymmetric head would not establish our novelty. |
| [Modality-free Graph In-context Alignment](https://proceedings.iclr.cc/paper_files/paper/2026/hash/23fcc63005ac1a6e460ec4e209d17607-Abstract-Conference.html), Zhuo and Luo, ICLR 2026 | Its abstract describes feature/label alignment, episodic prompt-aware attention, and adaptation from support examples without parameter updates. | Broad claims to introduce support-conditioned graph transfer or distinguish encoding from reasoning are too wide. |
| [TaskNorm](https://proceedings.mlr.press/v119/bronskill20a.html), Bronskill et al., ICML 2020 | Evaluates normalization for meta-learning and reports substantial effects on accuracy/training time across fourteen datasets. | Normalization affecting few-shot performance is established. A normalization explanation must identify a specific graph/episode pathway or yield a useful improvement, rather than rediscover this fact. |
| [When Do Graph Foundation Models Transfer? A Data-Centric Theory](https://arxiv.org/abs/2605.29828), Zhu et al., 2026 | The abstract decomposes output shift into finite-sample and structural-discrepancy terms in a dense-graph limit. | Our computational interventions complement data-distance explanations. They do not refute that theory, nor establish that graph mismatch is irrelevant. |

Access scope: abstract/proceedings pages for APT, GFL, MF-GIA, TaskNorm and the
data-centric theory; method sections 3.2–3.3 for GILT. Do not imply an exhaustive
full-paper comparison. GILT's arXiv version is used here without inferring a
main-conference acceptance from third-party indexing.

## Strongest defensible distinction

The proposed contribution is a predictive computational explanation of a
particular transfer failure, not the invention of supports, prototypes,
asymmetric attention, or context selection. In the one-layer discovery setting,
we can change support values and therefore final class references while leaving
query representations and attention unchanged. A component prediction fixed at
the discovery recipe subsequently holds on the nominated longer-trained recipe.
That evidence goes beyond a correlation between graph similarity and accuracy.

The public replication asks whether the component prediction extends to a
different domain and multiclass task. Its two-layer, training-mode architecture
does not retain the discovery setting's fixed-query guarantee. Crossed decoder
inputs can identify controlled reference-only and query-only effects; they are
not an additive causal mediation decomposition or an on-manifold intervention.

The main intellectual question is therefore not whether a class reference is
used, but why graph context produces an unsuitable reference in a particular
transfer and when that effect survives changes in model recipe and domain.
Without the public outcome and a discriminating explanation of the donor effect,
the current evidence does not yet support the desired broad-impact case.

## Competing explanations and one next decision

Local context content and normalization-mediated episode coupling are distinct
explanations for the public donor. Support-edge removal changes activations and
within-forward batch statistics. The altered support values contain both effects.
Replaying the original buffers and RNG prevents state drift but does not hold
those batch statistics fixed.

Retain the nominated public experiment. Conditional on a positive value/reference
effect, build the donor using native per-layer batch mean/variance while keeping
the transplant recipient native. This is the next high-information test, not a
new target sweep. Survival would strengthen a local-context pathway explanation;
disappearance would redirect the explanation toward normalization coupling.
Neither result alone demonstrates that a new training method improves performance.

If normalization is implicated, TaskNorm is required context and a relevant
adaptation baseline when proposing a repair. If local content is implicated,
GILT and graph-prototype methods are required context for any proposed role-aware
architecture. Do not rename an existing design principle and present it as new.

## Framing lesson from APT

The useful pattern in the user's example is observation -> actionable diagnosis
-> method -> demonstrated benefit. The observation supplies motivation; the
method and its verified benefit carry the contribution. We should apply that
logic without converting our failed allocation experiment into a successful
selection story. At present the mechanism is stronger than TRACE or allocation
as a central contribution, but the demonstrated practical improvement beyond
simple U1 readouts remains an open requirement for a method-led paper.
