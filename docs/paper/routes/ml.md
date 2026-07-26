# ML route: Capability is a property of the objective set

## The paper in one sentence

Self-supervised graph objectives do not compose additively: under a fixed training budget, rotating a complete set of complementary objectives can produce a transferable structural capability that is absent from every constituent objective and every pair.

## Why this is a paper

Most multi-objective pretraining work asks how to balance losses while assuming that capability changes smoothly as objectives are added. Our current result says something sharper and less expected: the objective lattice is non-monotonic. For `{neighbor matching, contrastive learning, feature prediction}`, single objectives and all pairs fail at frozen zero-shot static link prediction, while the three-way rotation succeeds on every evaluated graph. The effect is not a small average gain; it is a qualitative change in behavior.

The paper is not “we found a good recipe for Twitter graphs.” Its subject is **objective-set interactions in graph self-supervision**. Social graphs are the motivating testbed, but the claim must survive outside that one setting.

## Central claim

> A downstream capability can be super-additive in the set of SSL objectives and induced by scheduling rather than loss summation. Consequently, objective selection cannot be reduced to ranking objectives independently or testing only pairwise combinations.

“Emergent” should only be used if the continuous score distributions, seed sweep, and controls below support it. Otherwise the safer wording is “strong three-way interaction.”

## Research questions

1. **Is the three-way effect real?** Does the full objective lattice reproduce across seeds, checkpoints, and evaluation resampling?
2. **Is it structural?** Does the MIX representation encode adjacency information rather than a feature shortcut or an evaluation artifact?
3. **Is rotation load-bearing?** Does per-step rotation outperform simultaneous loss summation and other schedules at matched total and matched per-objective compute?
4. **Why does the complete set work?** Do all proper subsets converge to shortcuts that the third objective destabilizes?
5. **Is the principle portable?** Does a comparable interaction appear on a non-social graph family, another backbone, or a substituted objective menu?

## Existing evidence

The complete seven-arm subset lattice already provides an unusually clean starting point. At matched 40k total episodes, mean frozen static-LP AUC is 0.467/0.332/0.449 for NM/CL/FP, 0.305/0.424/0.227 for the three pairs, and 0.759 for MIX. MIX clears chance on all four graphs, including held-out TwiBot-20, while preserving near-NM classification performance. The best pair is worse than the best single on the joint feature/topology criterion.

This establishes a large one-seed phenomenon, not yet the paper's final claim. Static-LP values are setup-sensitive elsewhere in the project, and the current result has not been causally attributed to topology.

## Minimum viable experimental package

### 1. Replicate the phenomenon

- Re-run all seven lattice arms with at least five seeds through the byte-identical path.
- Evaluate the same checkpoint steps for every arm; report trajectories, not only the selected endpoint.
- Repeat evaluation with multiple negative-edge and probe seeds.
- Report per-dataset effects and hierarchical confidence intervals. The unit of evidence is not the pooled row count.

Pass condition: MIX exceeds every proper subset on static-LP on most seeds and on every dataset in aggregate, while remaining above the feature-task floor. If only one or two seeds show the jump, retire the emergence story.

### 2. Prove what the capability is

Run a crossed test-time surgery on frozen representations:

| intervention | preserves | destroys | prediction if MIX learned topology |
|---|---|---|---|
| feature permutation within graph | feature distribution and graph | node-feature binding | LP mostly survives |
| feature replacement/noise | graph | semantic content | LP substantially survives |
| degree-preserving edge rewiring | feature matrix and degree distribution | adjacency | LP collapses |
| edge deletion / randomized neighborhoods | feature matrix | message-passing structure | LP collapses |

Add a feature-only link predictor, raw-feature probe, random encoder, degree-only predictor, and a trained supervised upper bound. Inspect score distributions and calibration so a reversed or constant classifier cannot masquerade as capability.

### 3. Separate composition from compute and schedule

Use two compute controls:

- **Matched total compute:** all arms receive the same number of updates, as now.
- **Matched per-objective exposure:** MIX receives three times the updates of a single and pairs twice, with an equally long single-objective control to separate duration from exposure.

Compare round-robin rotation with simultaneous weighted-sum training, randomized task sampling, and block schedules. Keep examples, augmentations, optimizer steps, and encoder identical. E4 is useful negative context but is not a clean schedule control because it changes heads, targets, architecture, and loss scale.

### 4. Explain the interaction

The mechanism section should test one modest explanation, not promise a complete theory.

- Track per-objective gradient cosine, gradient norm, and update-to-weight ratio on shared encoder layers.
- Measure representation rank/effective dimension and CKA across lattice arms and checkpoints.
- Train small diagnostic probes for feature reconstruction, degree/count, neighborhood identity, and held-out adjacency.
- Test whether a shortcut becomes linearly decodable in singles/pairs but not MIX.
- Apply one intervention derived from the diagnostics: remove or weaken the suspected third-task constraint and predict the capability change before running it.

Gradient conflict is not itself an explanation. It matters only if its temporal pattern predicts when structural decodability appears and an intervention changes the outcome.

### 5. Establish scope beyond one graph/model recipe

The smallest credible generality package is:

- one non-social benchmark family with node features and enough scale for the same protocol;
- one materially different encoder backbone;
- one objective substitution that preserves the three roles (discrimination, augmentation invariance, reconstruction) without using the exact same losses.

These tests may show a boundary rather than universal replication. A useful conclusion would be that super-additivity occurs only when the objective set blocks both identity/content and structural shortcuts.

## Paper structure

1. **Motivation:** objective mixtures are normally evaluated as recipes or pairwise ablations.
2. **A complete objective lattice:** introduce the controlled seven-arm experiment and the non-monotonic interaction.
3. **Capability audit:** establish that the gain is real, frozen, transferable, and structural.
4. **Schedule versus sum:** isolate rotation as the causal design choice.
5. **Mechanism:** show how representation and optimization dynamics differ at the transition.
6. **Generality and limits:** repeat beyond the original social-graph setup and state where the effect fails.

## Figure plan

1. The seven-node objective subset lattice, colored by static-LP AUC, with confidence intervals.
2. Capability plane: classification versus static-LP, showing seeds and trajectories rather than only means.
3. Causal surgery plot for MIX and strongest proper subset.
4. Schedule/compute-control comparison.
5. Mechanism timeline: structural probe, feature probe, effective rank, and gradient statistic across training.
6. Generality matrix across dataset family, backbone, and objective substitution.

## Claims to avoid

- Do not claim that three objectives are universally necessary.
- Do not call below-0.5 AUC “absence of information” without checking score inversion and degeneracy.
- Do not use E4 as proof that all simultaneous multi-task learning fails.
- Do not claim a general foundation model: regression remains weak and the benchmark is small.
- Do not lead with the 50M model, “best,” or a training recipe. Those are implementation facts, not the contribution.

## Realistic execution order

1. Multi-seed lattice plus robust evaluation.
2. Frozen intervention suite on existing checkpoints.
3. Clean schedule and compute controls.
4. Mechanism diagnostics using saved checkpoints.
5. Only if steps 1–3 survive, pay for the extra backbone and non-social benchmark.

## Kill criteria and fallback

- **Seed failure:** if MIX is unstable, publish the complete lattice as a cautionary reproducibility/benchmark result only if the instability has an identifiable source.
- **Not structural:** if edge rewiring does not selectively destroy the result, reframe around objective-induced feature geometry and drop structural-capability language.
- **Rotation not causal:** if weighted sum or random scheduling matches MIX, the contribution becomes objective-set interaction, not rotation.
- **No external replication:** frame the result as a bounded case study of social attributed graphs; do not stretch it into a universal SSL law.

## Most plausible venue shape

This is an ICLR/NeurIPS-style submission only after replication, causal surgery, clean schedule controls, and one credible external setting. Before those are complete, it is a strong workshop paper with a memorable phenomenon but not yet a defensible general ML result.
