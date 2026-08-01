# Sampling Improvements

## Candidate Remedies

1. **Graceful Low-Degree Handling**  
   Do not skip centers that fail to produce enough unique positives. Keep them eligible by sampling with replacement, padding/masking missing positives, or adapting `n_shot + n_query` downward for sparse centers. This reduces degree/eligibility bias.
2. **Uniform Positive Sampling Over Eligible Neighborhood**  
   Replace repeated random-walk positive sampling with uniform sampling from the center's eligible `l`-hop neighborhood. This prevents high-probability random-walk neighbors from dominating and improves coverage of rare but valid local structure.
3. **Harder Negative Sampling**  
   Replace fully random negative centers with harder negatives. Sample negatives from the same source/community/degree band, or from nodes at controlled graph distances from the query/center. Include multiple distance buckets, e.g. 1-hop/2-hop/4-hop/nonlocal, to avoid trivial far-away negatives.
4. **Partial Cross-Source Episodes** — _under test in [`partial_cross_source/`](partial_cross_source/)_  
   Use mostly within-source episodes, but allow cross-source mixed episodes some fixed fraction of the time. Example: 80-90% within-source, 10-20% cross-source. This preserves source-discrimination learning without letting cross-source shortcuts dominate. Implemented as `neighbor_sampling_cross_source_prob` (p∈[0,1]); a 5-point sweep p∈{0,0.1,0.25,0.5,1} interpolates between the within-source (p=0) and naive (p=1) endpoints on the ukr+covid merge.
5. **Balanced Region Exposure**  
   Sample regions/components/sources with controlled probabilities instead of pure node-uniform sampling. Start with known `graph_id` balancing, then optionally extend to communities or degree buckets. This prevents small or hard regions from being starved.
6. **Loss-Adaptive Region Sampling**  
   Track smoothed loss per source/component/region and modestly upweight high-loss regions. Use this only as a bounded mixture with the base sampler, e.g. 70% balanced + 30% loss-adaptive, to avoid overfitting noisy regions.

## Tested

1. **Naive**: merge graphs disjoint.
2. **Within**: per episode, samples only from one source graph.
3. **Within-Balanced**: do not sample from the source graphs proportionally. Instead, rotate over the source graphs.

## Risks

1. **Eligibility/degree bias**: strict NM skips centers that cannot produce enough unique sampled neighbors. Low-degree, isolated, or sparse-region nodes may be underrepresented.
2. **Random-walk positive bias**: positives are not sampled uniformly from all `l`-hop neighbors; they are sampled through repeated random walks. This can overrepresent high-probability local edges and miss rare but important neighborhood structure. Instead, we can sample uniformly over all eligible neighbors.
3. **Easy random negatives**: random centers are often too easy. Even within one source, many negatives may be structurally/semantically far from the query. Hard negatives from the same community, same degree band, similar features, or nearby non-neighbors may be needed. If we sample two nodes from a graph, they might be far from each other, making the task easy.
4. **Partial cross-source tradeoff**: we do want the model to learn to distinguish between sources, so sample across graphs some controlled fraction of the time.
