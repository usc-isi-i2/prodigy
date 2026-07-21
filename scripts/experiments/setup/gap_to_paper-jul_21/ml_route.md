# ML Route
## Directions
### ICL cross-graph transfer is governed by divergence measured in the channel the pretext actually uses
#### Headline claim:
> In in-context graph pretraining on text-attributed social graphs, cross-graph transfer is governed by divergence measured in the channel the pretext actually uses — feature divergence predicts and causally controls neighbor-matching transfer, and provably fails on the one capability that lives in the topology channel.

#### RQs:

- **RQ1 (mechanism).** Which information channel does PRODIGY-style episodic pretraining actually exploit? H: feature content, not topology — the noise/permute/rewire surgery signature (noise fatal, permute harmless, rewire harmless), reproduced multi-seed under the fixed harness.
- **RQ2 (prediction).** Is cross-graph transfer predictable before any training from feature-cloud divergence? H: proxy-A-distance + an asymmetric coverage term predicts transfer across all 64 ordered pairs (hierarchical/QAP stats), beating language JS-divergence, user-overlap, size, degree-KS, and LogME/LEEP — and the asymmetric term explains donor/recipient structure no symmetric distance can.
- **RQ3 (causation).** Does feature divergence cause the transfer gap? H: monotone bidirectional dose-response under interventions that move divergence while topology is held fixed — on-manifold subsampling, rotation, interpolation (both directions), natural within-event temporal slices, with degree-preserving rewiring as the null control and natural pairs plotted on the same curve.
- **RQ4 (scope boundary — pre-registered).** Does the predictor fail exactly where the mechanism requires — on topology-borne capability? H: the emergent zero-shot LP from complete {NM, CL, FP} rotation (i) survives seeds, (ii) shows the mirror-image surgery signature (feature-noise harmless, edge-rewire fatal), (iii) requires rotation — a weighted sum of the same three objectives fails — and (iv) is not ranked by feature divergence.
- **RQ5 (payoff).** Can the predictor replace enumeration? H: pre-registered predicted-best 2–3-source mixtures match all-8 merged pretraining per target at matched budget; random/size/language-selected mixtures don't. Plus the standalone negative: source diversity never buys out-of-distribution transfer (rung 7 = rung 8 everywhere).
- **RQ6 (generality — now properly funded).** Does the channel-match principle survive a genuinely different SSL framework (GraphMAE/BGRL-class) and a second feature space (the existing tweet-content embeddings vs bios)?

#### Contributions (as the intro would state them):

- The channel-match principle, established causally: the first interventional, confound-controlled account of what governs cross-graph transfer in graph-FM pretraining (the gap the 2025 transferability survey names explicitly).
- A validated pre-hoc transfer predictor with real statistics at 64-pair scale, an asymmetric donor/recipient term, a competing-predictor table — and a demonstrated, pre-registered failure mode, which is what makes it a scientific claim rather than a tautology.
- Composition, not construction: a non-monotonic set-completeness result (singles .42 / pairs .31 / triple .76) showing zero-shot LP capability emerges only from the complete rotating pretext set, while a weighted multi-head of the same objectives degrades the encoder — certified topological by mirror-image surgery.
- Divergence-guided source selection as the practitioner payoff, plus the negative result that merging never buys OOD transfer.
- A rigor/benchmark artifact: the episodic-GFM evaluation methodology (degenerate 0-shot eval, AUC saturation, the input-scaling artifact with its fully-run overturn as a case study, seeding fix, MDE-gated claims) plus released divergence pipeline, transfer atlas, and synthetic probes — potentially a separate Datasets & Benchmarks submission.

### Emergence from multi-objective rotation
#### NeurIPS
**Thesis:** Structural capability in graph SSL is super-additive in objective diversity and cannot be reproduced by explicit multi-objective optimization — the mechanism is gradient conflict, which the MTL field works to remove.

**Contributions:**
- Emergence. Rotating 3 trivial SSL pretexts on one encoder yields zero-shot link prediction (0.76 AUC); no constituent objective clears chance (.33–.47).
- Super-additivity. Complete 7-arm subset lattice at matched compute: no pair clears chance either — the capability belongs to the full set, not a driver objective.
- Non-monotonicity. Joint feature+topology bar: singles 0.42 → pairs 0.32 → triple 0.76.
- Engineered routes fail. Architectural levers each buy one axis only; a hand-designed joint objective degrades both.
- Mechanism. Pairs admit a joint shortcut, the triple does not; shortcut audit + gradient conflict track emergence across all subsets.
- Inversion. Gradient surgery (PCGrad/CAGrad) destroys the capability — conflict is the source, not a pathology.
- Theory. In a CSBM with a linear encoder, single/pair minimizers stay in the feature subspace; only the full set recovers the adjacency eigenspace.
- Generality. Holds across graph families, architectures, and objective swaps; capability scales with objective-set diversity above a threshold.

Conversations:
- [“What to add for a top ML conference”](~/.claude/projects/-Users-philipp-projects-gfm-prodigy/b46e86df-c920-46fe-96df-571f73913fbf.jsonl)
- 

#### ICLR
**Thesis:** Capability in self-supervised graph pretraining can be super-additive and non-monotone in the objective set, it is produced by scheduling rather than summation, and the mechanism is shortcut suppression.

- A new phenomenon: emergent, super-additive topological transfer. An encoder pretrained by rotating NM+CL+FP reaches static-LP AUC 0.76 from frozen embeddings, while all three single objectives and all three pairs sit at or below chance (0.23–0.47). The capability is present in the full objective set and in no proper subset of it. (Established, 1 seed.)
- A reframe of multi-task SSL: scheduling, not summation. Rotating one objective per step unlocks a capability that a weighted sum of the same objectives does not — identifying objective scheduling as a load-bearing design lever distinct from objective choice. This is the conceptual contribution: the field's default (sum the losses) is the worse option. (Contingent on RQ-A.)
- A mechanism, and an explanation of the non-monotonicity. Adjacency becomes linearly decodable only under rotation; single- and pair-objective encoders collapse dimensionally onto their own shortcut. Decomposing the joint bar shows the feature axis degrades gracefully with per-objective compute while the topological axis has a threshold at three objectives — which explains why capability is non-monotone (singles 0.42 → pairs 0.32 → triple 0.76). (Contingent on RQ-C/RQ-B.)
- The first complete subset-lattice study of SSL objective composition. All 7 non-empty subsets of {NM, CL, FP} at matched compute through a byte-identical code path. Methodologically, this shows that combination effects in SSL are not recoverable from pairwise or additive analyses — the standard ablation grid would have missed this entirely.
- Evidence it is a pretraining principle, not a domain artifact. The effect reproduces off social graphs, across encoder backbones, and under substituted objective menus — establishing whether the driver is number of objectives or diversity of inductive bias. (Contingent on RQ-E/F/G; this is what makes the paper travel.)
- A documented set of negative results on the engineered alternatives. Degree-as-input, count-aware aggregation, and a hand-designed multi-task loss each buy at most one axis, and the multi-task loss degrades both. Simple rotation beats deliberate topology engineering — plus a joint evaluation protocol scoring min(feature, topological) rather than a mean, which prevents a specialist's failure from being averaged away.