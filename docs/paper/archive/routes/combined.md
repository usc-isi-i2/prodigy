# Combined route: Transfer follows the information channel a pretext learns

## The paper in one sentence

Cross-graph pretraining transfers only along information channels that its self-supervised objective makes usable: content-matching objectives travel across events with similar user-language populations, while structural transfer requires a different composition of objectives.

## Why combine the stories this way

A combined paper should not be the ML paper plus the CSS paper stapled together. It needs one causal spine that connects data, learning, and downstream behavior.

The proposed spine is **channel alignment**:

1. Social graphs expose at least two channels: node/profile content and relational structure.
2. A pretext does not automatically learn both merely because its input is a graph.
3. Neighbor matching learns a permutation-invariant neighborhood-content channel.
4. Its cross-event transfer is therefore governed by content/population coverage, not generic graph similarity.
5. A different objective composition can make topology usable and changes which tasks transfer.

This makes the negative and positive results mutually informative. The feature ablation explains the event-transfer atlas; the objective lattice demonstrates that the channel is learnable rather than absent from the data.

## Central claim

> The transferable unit in graph pretraining is not “the graph” but the information channel selected by the pretext. Measuring shift in that same channel predicts transfer and exposes when a graph foundation model is only a text-population matcher.

The strongest version adds: intervening on channel divergence causally changes transfer, and changing the objective changes both the learned channel and the relevant transfer predictor.

## Research questions

1. **Channel identification:** Which input channel does each pretext actually use—profile content, node-feature binding, local counts, or adjacency?
2. **Transfer geometry:** Does distance in the used channel explain source-to-target transfer better than distance in unused channels?
3. **Causal alignment:** When one channel is changed while the other is held fixed, does transfer move in the predicted direction?
4. **Objective dependence:** Can objective composition switch the learned representation from content-only to content-plus-structure?
5. **Decision value:** Can channel-matched distances select a small pretraining corpus for a new event and task?

## The evidence chain

### Link 1: identify what NM uses

Existing feature surgery is strong: real-but-wrong feature noise collapses NM, whereas within-neighborhood permutation preserves it. With one-hop star episodes, NM acts on a bag of neighborhood biography content. Topology remains intact under both perturbations, so it cannot explain the retained/lost performance in this setup.

### Link 2: show that the same channel organizes transfer

The source-target matrix has stable donor/recipient structure. The pilot similarity analysis finds feature-cloud separability much more aligned with transfer than degree-distribution distance. The inclusion ladder shows why simple source count is insufficient: performance on a blind-spot target stays flat until its own population enters training.

### Link 3: demonstrate that another channel can be learned

The complete `{NM, CL, FP}` rotation is the only arm in the seven-arm objective lattice with strong frozen static-LP, while preserving classification. If edge surgery confirms that this LP signal is adjacency-dependent, the result is the constructive counterpart to the NM diagnosis: structural capability is not automatic, but objective composition can induce it.

### Link 4: show that the predictor must change with the channel

For NM/content transfer, feature/population distance should predict performance. For a topology-capable encoder evaluated on structural tasks, topology or feature–structure coupling should become the stronger predictor. This cross-over is the critical combined-paper experiment. Without it, “channel alignment” remains a plausible synthesis rather than a demonstrated principle.

## Minimum viable experimental package

### A. Build a controlled channel audit

Apply the same intervention grid to three representative encoders:

- NM content specialist;
- MIX candidate generalist;
- a topology-oriented architecture/objective control such as E2.

Evaluate at least one content task and one structural task. Interventions should include feature noise, within-subgraph feature permutation, degree-preserving edge rewiring, edge deletion, and a degree-only preservation control. Include raw-feature, raw-degree, random-encoder, and supervised upper bounds.

The desired result is a signature matrix, not a winner table:

| encoder/task | feature corruption | edge rewiring | interpretation |
|---|---:|---:|---|
| NM on classification/NM | large effect | small effect | content channel |
| MIX on static-LP | small effect | large effect | structural channel |
| E2 on static-LP | to be measured | large effect expected | architectural topology control |

If MIX shows the same feature-sensitive signature as NM, the combined thesis fails and the LP result needs a different explanation.

### B. Construct two transfer matrices under one protocol

Use a manageable but diverse subset of sources and targets rather than multiplying every experiment by eight immediately.

- **Content matrix:** NM encoder evaluated on neighbor matching or classification.
- **Structural matrix:** MIX and/or E2 evaluated on static or preferably temporal link prediction.

Match graph corpus, source sampling, checkpoint budget, seeds, and evaluator. This avoids the current problem that NM→LP values differ by more than 0.1 across experiment families.

### C. Compare channel-specific distances

For each source-target pair, compute:

- content/population: language proportions, feature centroid/Fréchet/MMD, proxy-A-distance;
- topology: directed degree distributions, reciprocity, assortativity, motif or graphlet summaries, spectral summaries where feasible;
- coupling: edge-versus-random feature homophily and directional variants;
- nuisance variables: source size, time span, collection mechanism, and user overlap.

Fit target-stratified models and use held-out-target evaluation. The key prediction is an interaction:

`transfer ~ content_distance × content_sensitive_encoder/task + topology_distance × topology_sensitive_encoder/task`

The paper succeeds if the ranking of predictors changes in the expected direction, not merely if one global correlation is significant.

### D. Intervene on distance

Observational similarity is insufficient because event properties are entangled. Use paired interventions on a fixed graph:

- change content while preserving adjacency: interpolate, replace, or subsample feature populations;
- change adjacency while preserving node features and degree as far as possible: degree-preserving rewiring with increasing doses;
- use natural temporal slices as an external-validity check, not the sole causal design.

For each encoder/task pair, trace a dose-response curve. The content-sensitive model should track content perturbation; the structural model/task should track edge perturbation. Plot natural source-target pairs on the same axes only as context.

### E. Demonstrate decision value

For held-out targets and a declared downstream task, choose 2–3 source graphs by the distance corresponding to the encoder's learned channel. Compare with all-source, largest-source, same-language, topology-only, and random selection at matched compute.

This is a stronger and more realistic payoff than claiming the globally “best” pretraining mix. The output is conditional: given a task and pretext, select sources in the channel that pretext can exploit.

## Paper structure

1. **Motivation:** graph input does not imply structural learning; transfer claims must name the channel.
2. **Framework:** define content, topology, and feature–structure coupling as empirically testable channels.
3. **Diagnosis:** causal surgery identifies NM as neighborhood-content matching.
4. **Across events:** content coverage explains the donor/recipient atlas and target-inclusion staircase.
5. **Changing the channel:** objective rotation yields a candidate structural representation.
6. **Channel-matched prediction:** the relevant graph distance changes with encoder and task.
7. **Interventions and source selection:** causal dose-response plus held-out deployment test.
8. **Implications:** what “foundation” and “cross-graph” should mean for social-network models.

## Figure plan

1. Conceptual diagram: graph input → pretext-selected channel → representation → task-specific transfer.
2. Intervention signature heatmap across encoder/task combinations.
3. Event transfer atlas plus target-inclusion trajectories.
4. Two predictor panels: content transfer versus content distance; structural transfer versus topology distance.
5. Causal two-axis dose-response surface or paired curves.
6. Held-out source-selection regret relative to exhaustive best and all-source training.

## What is novel here

- A **diagnostic principle**: identify the channel by intervention rather than infer it from the model class.
- A **transfer principle**: compare domains in the channel the objective actually preserves.
- An **objective result**: structural capability can appear through non-additive pretext composition.
- A **social-computing consequence**: broad cross-event scores may reflect familiar biography-language populations rather than portable relational behavior.

Each contribution supports the others. Remove the cross-over experiment and the paper becomes two adjacent case studies; remove the event atlas and it becomes the narrower ML route.

## Claims to avoid

- Do not claim a theorem or universal law from two channels and one model family.
- Do not equate profile embeddings with demographics or social identity.
- Do not say topology is unused by all graph models; the statement is pretext-, architecture-, and task-specific.
- Do not claim causal divergence from observational event pairs alone.
- Do not claim corpus selection works until evaluated on held-out targets without using their transfer scores.
- Do not call static edge reconstruction temporal forecasting. If possible, use temporal LP for the operational payoff.

## Realistic execution order

1. Run the intervention signature on existing NM, MIX, and E2 checkpoints.
2. Replicate the objective-lattice headline and the largest atlas effects across seeds.
3. Build the matched content and structural transfer matrices on a reduced graph panel.
4. Test the channel-by-distance interaction.
5. Run the paired divergence interventions.
6. Only then spend compute on prospective source selection and a full eight-graph expansion.

## Kill criteria and fallback

- **MIX is not adjacency-dependent:** drop the structural-switch section and pursue the CSS coverage paper.
- **Distances do not cross over by channel:** pursue the ML objective-composition paper; do not claim channel-aligned prediction.
- **Interventions lack dose-response:** restrict conclusions to diagnostic ablations and observational transfer geometry.
- **Source selection does not beat simple baselines:** retain it as a negative operational result, but remove “actionable predictor” language.

## Most plausible venue shape

This route has the broadest intellectual payoff and the cleanest unifying idea, but it also has the most dependencies. It is realistic only if the first three tests—MIX seed stability, adjacency dependence, and predictor cross-over—succeed. If they do, the result can speak to both graph representation learning and computational social science without reading like two papers forced into one.
