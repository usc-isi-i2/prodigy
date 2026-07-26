# CCS Route

## Directions

### Cross-event transfer is mappable

#### Headline claim:
> Cross-event transfer of social-graph foundation models is mappable, predictable from event properties — and carried almost entirely by bio text rather than network structure, which changes what moderation and info-ops pipelines can expect from them.

#### RQs:

- RQ1 (atlas). Which events transfer to which, for the tasks practitioners actually run (political leaning, account suspension, bot detection)? H: reproducible donor/recipient structure — large multilingual crisis events (covid, ukraine) are near-universal donors, topically/linguistically isolated events (Hong Kong) are islands, and no pretraining corpus generalizes to an event it doesn't contain.
- RQ2 (what makes a donor). What event-level properties explain donor quality — bio-embedding divergence, language composition, audience/population overlap, event size, time window? H: feature-cloud divergence dominates — but if language or population overlap predicts equally well, that is the finding: "cross-event transfer is language/population matching," reported honestly.
- RQ3 (the warning). Is transfer carried by text or by network structure? H: text — the surgery evidence that these "graph" models are multilingual bio-text matchers, with two consequences tested directly: a bio-paraphrase evasion experiment (does trivial rewriting of profiles defeat transfer-based detection?) and the implication that structure-borne signals aren't inherited.
- RQ4 (the coordination test). Does the topology-capable encoder (pretext rotation) specifically beat the text-matcher on tasks where the signal is structural coordination (suspension, bots)? H: yes on structure-heavy labels, no on ideology labels — the constructive counterpart to RQ3's warning, and the cheapest high-significance experiment in the whole packet.
- RQ5 (rapid response). Facing a new crisis, does a pretrained checkpoint plus the first 5–25% of the event's edges (by timestamp) recover most of full-inclusion performance — and can divergence-guided corpus selection tell a team which past events to pretrain on? Either outcome is decision-relevant; a negative ("foundation checkpoints don't rescue you in a fresh crisis") is published as prominently as a positive.

#### Contributions:

- The first cross-event transfer atlas for social-graph pretraining on real moderation-relevant tasks, with donor/recipient structure that replicates across seeds.
- An event-level predictor of transfer computable before training, adjudicated against the language/population explanations a CSS audience will (rightly) suspect first.
- Mechanism evidence with consequences: current social-graph FMs are bio-text matchers — including an evasion test and the boundary case where structural capability actually matters.
- A rapid-response result for new-event deployment, positive or negative.
- Released artifacts: the atlas, divergence pipeline, and derived matrices (raw graphs can't be shared), plus an ethics/data statement on label provenance (TwiBot-20, suspension-as-policy-artifact, left/right operationalization).