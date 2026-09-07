# Temporal crossover: closest prior and the contribution boundary

7 September 2026. Focused primary-source literature audit, not an exhaustive
novelty certificate. The previous turn produced the verified aggregate export,
rendered figure and figure-led argument; it was progress. This audit checks the
intellectual interpretation before any manuscript expansion.

## What is already established

**Functional compatibility via component recombination.** Bansal, Nakkiran
and Barak, *Revisiting Model Stitching to Compare Neural Representations*
(NeurIPS 2021), compare representations by joining frozen lower and upper
networks through a trained low-capacity connector. Their experiments include
training-duration comparisons. Component recombination and consumer-dependent
representation evaluation are therefore not new techniques here. Our crossover
uses no fitted connector and evaluates transfer across graph tasks; failure of
an unaligned swap does not refute their optimized stitching results.
[Primary paper, sections 1–2](https://papers.nips.cc/paper_files/paper/2021/file/01ded4259d101feb739b06c399e9cd9c-Paper.pdf).

**Compatibility and complementary strengths of foundation models.** Mai et al.,
*Revisiting Model Stitching In the Foundation Model Era* (CVPR 2026), study
heterogeneous visual models and train connectors using final-feature matching.
Their stitched models can exceed their constituents. Our best measured temporal
hybrid exceeding both diagonal checkpoints is therefore not, by itself, a new
principle. We have not established a connector or a deployable selection rule.
[Primary paper](https://arxiv.org/html/2603.12433v2).

**Representation/readout dynamics.** Chou et al., *Two Speeds of Learning:
A Representation-Readout Decomposition of Grokking and Double Descent*
(arXiv:2605.27078v1, May 2026), explicitly separates encoder and final-readout
dynamics using probes, geometry and kernel alignment. Its general distinction
between representation progress and end-to-end progress overlaps strongly with
our candidate vocabulary. We cannot claim that decomposition or unequal learning
dynamics as the contribution. Their studied grokking/double-descent mechanisms
are not automatically explanations of our graph-transfer results.
[Primary paper, sections 1–2](https://arxiv.org/html/2605.27078v1).

**Feature distortion and latent generalization.** Kumar et al. (ICLR 2022)
explain how downstream fine-tuning can distort pretrained features and harm OOD
performance relative to probing. Our temporal comparison concerns continued
source pretraining, not downstream parameter adaptation; their mechanism is an
alternative conceptual precedent, not a mechanism we tested.
[Primary abstract](https://arxiv.org/abs/2202.10054).
Ketha and Ramaswamy (2026) study useful latent representations during label-noise
memorization and transfer that utility through a new linear probe. Hidden probe
utility and its training trajectory are not sufficient novelty either.
[Primary abstract](https://arxiv.org/abs/2603.19865).

## What our evidence adds, if the pending replication supports it

The sharper empirical result is **incorrect attribution of transfer decline**:
on a fixed target and fixed episode set, additional source pretraining worsens
the complete graph model even though later inference improves either tested
encoder. The adverse conditional change follows encoder outputs under both
learned inference modules, but not to the same extent under the fixed
support-fitted readout. Normalization-mode sensitivity does not erase the
diagonal divergence. This is a particular causal comparison of deployed
computations, not the invention of representation/readout separation.

The social contribution remains a different, role-specific result: changing
support values can help or harm at fixed query representations and attention.
The public result does not establish that its temporal decline occurs in the
support value path. Do not join them with a sentence asserting one proven
mechanism. Their common diagnostic lesson is useful, but that lesson alone
does not establish the requested high-impact contribution.

## Advisor decision

Retain the figure-led argument, with the above attribution result as the
candidate empirical center. Credit stitching and representation/readout
dynamics explicitly in any revised paper. Do not frame the public result as
a contradiction of a universal 'more training is better' theorem, or rename
generic compatibility as a new mechanism. The pending fixed second-seed test
remains the next decision point; no new experiment follows from this audit.

Access scope: the 2021 primary PDF and 2026 arXiv HTML pages were accessible.
Some OpenReview pages served a browser challenge and the CVF direct open
returned 403; those were not bypassed. Claims above rely on the accessible
primary papers/abstracts, not third-party summaries. No newly added citations
were inserted into the manuscript or compiled PDF in this audit.

Private worktree `.worktrees/role-topology`, branch
`codex/role-topology-interactions`, inspected HEAD `65946f82`. No push.
