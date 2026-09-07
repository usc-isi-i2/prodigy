# Claim audit: class-reference transfer after the nominated pairing

7 September 2026. Private evidence analysis, not manuscript expansion. Worktree
`.worktrees/role-topology`, branch `codex/role-topology-interactions`, code65946f82.
No model forwards, training, source selection or new outcomes were requested.

## New accounting: the pathway contrast depends on the other pathway

Let I, K, V and J be within-episode AUC for intact, keys-only replacement,
values-only replacement and joint replacement. Replacements use the original
support-edge-suppressed donor. The two key effects are K-I and J-V; the two
value effects are V-I and J-K. Interaction is J-K-V+I. These are finite
conditional contrasts, not independent mediation fractions.

| Pair/configuration/stream | Key at native values | Key at donor values | Value at native keys | Value at donor keys | Interaction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hong Kong / political / 2.5k original | +1.230 | +1.584 | +5.664 | +6.019 | +.354 |
| Hong Kong / political / 50k original | +.152 | +1.906 | +6.858 | +8.612 | +1.754 |
| Hong Kong / political / 50k fresh | +.839 | +1.425 | +8.927 | +9.512 | +.586 |
| Ukraine / bots / 2.5k original | +1.188 | +.222 | -6.418 | -7.384 | -.966 |
| Ukraine / bots / 2.5k fresh | +.928 | -.353 | -4.975 | -6.255 | -1.280 |

Units: AUC percentage points. Computed directly from metrics.json in
`data/classrefkv_discovery_20260907`, `data/classrefkv_long_20260907`, and
`data/kv_generality_20260907`. Both algebraic paths reconstruct J-I exactly
up to floating-point arithmetic. This analysis is post-result accounting;
only the previously recorded native-anchor Ukraine ordering was frozen.

The value sign is stable across the two routing endpoints in every measured
cell: suppression-derived values help political ranking and hurt bot ranking.
Value changes exceed the corresponding key changes in absolute magnitude at
both endpoints in all five cells. This is a finite-endpoint observation, not
monotonicity between endpoints or across other sources. The key effect is
small and is not sign-stable in fresh bot episodes. Therefore say **opposing
effects at the native operating point**, not that routing is intrinsically
beneficial or that pathway effects add independently.

## Closest prior work: meaningful overlap, not a contradiction

[How Few-Shot Examples Add Up](https://arxiv.org/html/2605.16591v2), section5,
already factors contextualized QK and V. AppendixL explicitly discusses value
changes that help or harm and cautions that geometric trends do not suffice
to explain these effects. Hence neither the factorial nor bidirectional value
utility is new. Its endpoint is maximum function-vector injection accuracy
over layer/scale choices; ours is a fixed graph classifier's native outputs.
Its contextualization is cross-prompt-component attention; ours changes
within-example graph neighborhoods before support-to-label aggregation.
These distinguish interventions and questions, not establish superior novelty
or refute its routing-dominant finding. [Function Vectors](https://arxiv.org/abs/2310.15213)
already supplies causal task-vector background. [PRODIGY](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf)
already supplies graph-based prompt inference. Credit all three.

## Concrete contribution outline and advisor decision

Working research question: **Which role-specific computation transfers when
a graph in-context classifier is moved to a new graph?**

1. Establish the diagnostic ambiguity using the nine-source role map. Its
   endpoint scores describe source-target behavior, not a property of graph
   similarity alone. Do not claim the map explains the sources' rankings.
2. Localize support effects to class-reference construction with queries
   fixed, then distinguish routing from transported content. Lead with the
   two natural pairings and report both conditional contrasts above, rather
   than presenting the graph architecture as the discovery.
3. Show why this distinction changes interpretation: an attention change
   that improves native bot ranking is bundled with a larger harmful content
   change. Conversely, political ranking gains can arise without changing
   routing. Neither net suppression utility nor an attention map identifies
   which computation failed.
4. Separate ranking from decision quality and scope from universality: retain
   political50k accuracy collapse, the useful public model's role coupling,
   and stronger simple readouts. These bound the diagnostic; they cannot be
   advertised as successful repair or independent-family replication.

This is an application-specific causal diagnosis with a potentially useful
evaluation lesson, not a new general theory of attention. Prior work materially
limits the generic K/V headline. The publication case must rest on the joint
role-specific transfer evidence and its concrete diagnostic consequence.
The nominated test strengthens that case but does not by itself prove the
requested high-impact contribution. Do not restart closed synthetic tests,
launch another source sweep, or expand the manuscript to hide this remaining
judgment. Next writing work should reorganize existing evidence around these
four claims, subject to the one-page argument being judged convincing.
