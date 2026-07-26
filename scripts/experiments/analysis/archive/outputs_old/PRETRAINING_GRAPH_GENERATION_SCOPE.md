# Pretraining Graph Generation Experiment Scope

## One-line scope

This experiment tests whether self-supervised pretraining on large, temporally evolving social graphs produces representations that use both network structure and content features, transfer to held-out datasets and time windows, and improve with more data or model capacity.

The working goal is:

> Build a reproducible pretraining pipeline whose outputs transfer better than from-scratch training, and define exactly what evidence is enough to stop or continue.

This document scopes the graph generation, pretraining, and evaluation work only. It is meant to align advisor, student, and collaborator expectations around what will be built, what will be measured, and what counts as done.

## What this experiment will do

The experiment will build a reproducible graph pretraining pipeline around the existing PRODIGY-style graph learning codebase. It will generate standardized temporal social graph artifacts with both network structure and actor/content features, pretrain a graph model using self-supervised graph tasks, evaluate pretraining quality, and measure transfer to held-out datasets, event windows, and pre/post-event regimes.

The core artifact is a pretrained graph-model checkpoint plus the graph construction and evaluation code needed to reproduce it.

The experiment has three concrete parts:

1. Generate graph artifacts from multiple socio-technical datasets using one shared graph contract.
2. Pretrain models using self-supervised graph objectives on one or more source graphs.
3. Evaluate transfer on held-out graphs and event windows against from-scratch and standard GNN baselines.

## Research questions

### RQ1: Can one graph pretraining recipe learn useful structure and content representations across multiple social graph datasets?

The model should converge on intrinsic self-supervised objectives that require both topology and content/attribute signals, such as masked node-feature reconstruction, masked edge reconstruction, neighbor matching, and temporal link prediction across a merged multi-event graph. In the current repo, bio/profile embeddings are the first content signal; later versions may include post text, narrative embeddings, account metadata, or interaction-type features.

### RQ2: Does pretraining improve transfer relative to training from scratch?

A pretrained model should outperform an otherwise matched randomly initialized model on held-out datasets, event windows, or downstream tasks.

### RQ3: Does transfer improve with scale?

Increasing pretraining data volume and/or model capacity should produce a monotone or mostly monotone improvement in held-out transfer metrics. The useful output is a scaling curve, not just a single large run.

### RQ4: Are improvements consistent across task families?

The primary transfer tasks are structural and temporal. Label-based classification tasks may be included when labels are already available, but they are secondary to graph reconstruction, content reconstruction, and link prediction.

### RQ5: Do pretrained representations degrade less across major event boundaries?

The model should degrade less than from-scratch and single-event baselines when evaluated across pre-event, during-event, and post-event windows. This is a stress test across known historical event windows, not a claim about forecasting future reactions.

## In scope

### Graph generation

The experiment will generate or reuse graph artifacts with a consistent schema:

- `edge_index`: directed graph edges.
- `edge_attr`: edge features, initially raw retweet count as `n_retweets`.
- `x`: node/content features, initially user bio/profile text embeddings when available.
- `user_ids` and `u2i`: stable user-id mapping.
- `feature_names` and `edge_attr_feature_names`: explicit feature metadata.
- `edge_index_views`: named graph views, including `retweet_all` and `temporal_history`.
- `target_edge_index_views`: target views, including `temporal_new` for future-link prediction.
- `future_edge_index`: compatibility field for temporal link prediction.
- event-window metadata: named pre-event, during-event, and post-event windows when anchor dates are available.
- `.meta.json`: graph counts, time ranges, embedding coverage, cutoff settings, temporal split stats, event-window definitions, and input provenance.

The minimum graph semantics are:

- Nodes represent users or platform-native actor entities.
- A directed edge `A -> B` means actor `A` retweeted, mentioned, replied to, reposted, or otherwise interacted with actor `B`, depending on the dataset.
- Edge weights/counts represent observed interaction frequency over the construction window.
- Node features represent actor/content context, starting with bio/profile embeddings. They are part of the experiment, not just implementation detail.
- Temporal views split historical context from future target edges.
- Event views split the graph around known external events when anchors are available.

### Content and attribute signals

Social network behavior is not only topology. The experiment will explicitly track content and actor attributes as graph signals.

Minimum content feature:

- User bio/profile embeddings, using the existing embedding stores where available.

Optional content features:

- Tweet/post text embeddings aggregated to user, edge, community, or time-window level.
- Narrative/topic embeddings around an event window.
- Account metadata features, if stable and ethically usable.
- Interaction-type features for retweets, replies, mentions, quotes, reposts, or cross-platform equivalents.

Required content-aware analyses:

- Feature availability audit: embedding model, dimensionality, missing-feature rate, and zero-vector rate per dataset.
- Content ablation: structure-only versus content-only versus structure-plus-content where feasible.
- Content transfer check: whether bio/content embeddings improve held-out transfer beyond topology alone.
- Content stability check: whether feature distributions shift across event windows and whether the pretrained model remains useful under that shift.

### Initial dataset set

The first complete experiment should use datasets already represented in the repo:

| Dataset | Role | Primary graph type | Primary use |
|---|---|---|---|
| `ukr_rus_twitter` | Source and held-out target | Twitter retweet graph | Geopolitical conflict event graph |
| `covid19_twitter` | Source and held-out target | Twitter retweet graph | Pandemic event graph |
| `midterm` | Source and held-out target | Twitter retweet graph | Election/political event graph |

These three datasets are enough for the first complete version because they support multi-event pretraining, leave-one-event-out transfer, and temporal link prediction.

Optional stretch datasets should only be added after the three-dataset pipeline is stable:

- `election2020`
- `hate_bots05`
- `hate_bots08`
- `ukr_rus_hate`
- `ukr_rus_suspended`
- `covid_political`
- `social_llm`
- non-Twitter datasets, such as Reddit, BlueSky, Mastodon, TikTok, and Telegram, if they can be converted into the same graph contract.

### Pretraining objectives

The first version should focus on self-supervised graph objectives that define the finish line:

| Objective | Description | Primary metric |
|---|---|---|
| Masked node reconstruction | Hide node features or labels and reconstruct them from graph context. | Reconstruction accuracy or feature reconstruction loss |
| Masked content/attribute reconstruction | Hide bio/profile embeddings or content-derived attributes and reconstruct them from graph context. | Cosine similarity, MSE, retrieval accuracy, or binned reconstruction accuracy |
| Masked edge reconstruction | Hide observed edges and recover missing structure. | Edge reconstruction accuracy |
| Temporal link prediction | Use `temporal_history` to predict `temporal_new` future edges. | AUC, average precision, F1 |
| Neighbor matching | Existing PRODIGY-style task for learning local structural context. | Accuracy, F1, ROC AUC where available |

The primary target metrics are:

- Greater than 90% masked node/edge reconstruction accuracy.
- Greater than 0.85 AUC on link prediction benchmarks.

If masked reconstruction accuracy is not directly implemented yet, the scope includes adding a thin evaluation wrapper that reports the closest equivalent currently supported by the training code, then implementing the missing metric before treating the experiment as complete.

### Model families

The main model is the existing PRODIGY-style pretraining architecture using the repo's configurable layers, such as GraphSAGE layers plus upsample/metagraph layers.

The experiment should test at least three scale points:

| Scale | Intended purpose | Example settings |
|---|---|---|
| Small | Debug and smoke-test baseline | 1 dataset, capped nodes/episodes, low hidden dimension |
| Medium | Feasible repeatable experiment | 2 datasets, moderate cap, default hidden dimension |
| Large | Main result | 3+ datasets or full merged graph, largest 4xH100-feasible model |

Exact hidden dimensions, depth, batch size, and caps should be recorded in each run config. The scope does not require a new architecture unless the existing architecture cannot meet the objectives.

### Evaluation

The evaluation will measure both intrinsic pretraining quality and transfer.

Intrinsic evaluation:

- Pretraining loss convergence.
- Masked node/edge reconstruction accuracy.
- Masked content/attribute reconstruction quality.
- Temporal link prediction AUC and average precision.
- Stability across random seeds.

Transfer evaluation:

- Train on one or more source graphs, evaluate on held-out graph.
- Train on early temporal window, evaluate on later window.
- Train on pre-event windows, evaluate on during-event and post-event windows.
- Compare structure-only, content-only, and structure-plus-content variants.
- Pretrained initialization versus from-scratch initialization.
- Pretrained model versus standard GNN baselines.

Event-window stress test:

- Define event anchors for each dataset, such as pandemic onset, invasion/escalation dates, election day, major protest/riot windows, or campaign exposure windows.
- Construct pre-event, during-event, and post-event graph views using consistent window rules.
- Measure performance degradation from pre-event to during/post-event targets.
- Compare degradation for the pretrained model, from-scratch matched model, single-event pretraining, and standard GNN baselines.
- Report both absolute post-event performance and relative drop from the pre-event setting.

Minimum transfer matrix:

| Pretraining data | Held-out target | Required tasks |
|---|---|---|
| COVID + Midterm | Ukraine/Russia | Temporal LP, neighbor matching |
| Ukraine/Russia + Midterm | COVID | Temporal LP, neighbor matching |
| Ukraine/Russia + COVID | Midterm | Temporal LP, neighbor matching |
| Ukraine/Russia + COVID + Midterm merged | Each individual graph | Temporal LP, neighbor matching |

Minimum event-window matrix:

| Dataset | Event anchor | Train window | Evaluation windows | Required tasks |
|---|---|---|---|---|
| `covid19_twitter` | COVID-19 onset or major policy/news event | Pre-event/history | During-event, post-event | Temporal LP, neighbor matching, content reconstruction if available |
| `ukr_rus_twitter` | Russia-Ukraine invasion/escalation anchor | Pre-event/history | During-event, post-event | Temporal LP, neighbor matching, content reconstruction if available |
| `midterm` | Election-day or campaign-event anchor | Pre-event/history | During-event, post-event | Temporal LP, neighbor matching, content reconstruction if available |

Optional downstream classification tasks may be run when label quality is acceptable:

- Political leaning classification.
- Bot/hate/suspension classification.
- Other dataset-native labels.

These are secondary transfer probes, not required for the main finish line.

### Baselines

The experiment should include three baseline classes:

| Baseline | Purpose |
|---|---|
| From-scratch matched model | Tests whether pretraining itself helps. |
| Single-dataset pretrained model | Tests whether multi-dataset scale helps beyond one source graph. |
| Standard GNN baselines | Tests against common graph architectures. |

Standard GNN baselines should include at least two of:

- GraphSAGE
- GAT
- TGN

Include all three when implementation time allows. The baseline should use the same graph splits, node features, temporal views, and evaluation metrics as the pretrained model whenever possible.

## Out of scope

The following are explicitly outside this experiment:

- IRL reward inference from behavioral trajectories.
- RL fine-tuning using inferred rewards.
- MARL or reward-aligned policy optimization.
- Claims that the model forecasts behavioral response to future disruptions.
- Claims about latent incentives, motivation shifts, or behavioral interpretability through rewards.
- Inference-boundary analysis, partial-observability lower bounds, and reward-stability analysis.
- End-to-end reward-aligned modeling beyond this pretraining experiment.

Event-window degradation is in scope only as a supervised/self-supervised representation stress test across known historical event boundaries. Later work may use the pretrained model, but that later work is not required for declaring this experiment complete.

## Detailed work plan

### Phase 0: Data and graph contract audit

Goal: confirm that each candidate dataset can produce a graph artifact that satisfies the shared contract.

Tasks:

- Verify raw or staged data locations for `ukr_rus_twitter`, `covid19_twitter`, and `midterm`.
- Run graph builders or inspect existing `.pt` graph artifacts.
- Confirm that each graph contains `edge_index`, `edge_attr`, `x`, `user_ids`, `u2i`, and temporal views.
- Generate or update `.meta.json` files with graph counts, time ranges, and embedding coverage.
- Record feature embedding model, dimensionality, and missing-feature rate.
- Define candidate event anchors and pre/during/post windows for each dataset.
- Confirm temporal splits are non-empty and plausible.

Exit criteria:

- Three graph artifacts load successfully through the existing dataset loaders.
- Each graph has a non-empty `temporal_history` view and a non-empty `temporal_new` target view.
- Each graph has either a named event-window split or a documented reason why event-window evaluation is not available.
- Each graph has a metadata file sufficient to reproduce construction.
- One smoke evaluation runs on each graph.

### Phase 1: Single-graph smoke tests

Goal: prove that each graph can train and evaluate independently.

Tasks:

- Run neighbor matching on each graph with a small cap.
- Run temporal link prediction on each graph with a small cap.
- Save model checkpoints, logs, and metrics.
- Fix graph schema or loader issues before scaling.

Exit criteria:

- All three datasets complete a short run without loader, tensor-shape, or empty-split failures.
- Metrics are above random baseline on the smoke setting.
- Evaluation JSON or CSV outputs are written consistently.

### Phase 2: Multi-graph pretraining

Goal: pretrain a model on multiple event graphs using the shared graph contract.

Tasks:

- Merge `ukr_rus_twitter`, `covid19_twitter`, and `midterm` into a disjoint multi-graph artifact.
- Preserve graph/source identifiers for stratified sampling and held-out evaluation.
- Pretrain with neighbor matching and/or masked reconstruction.
- Include content-aware objectives or ablations using bio/profile embeddings.
- Run temporal link prediction evaluation at checkpoints.
- Track convergence, wall-clock time, GPU memory, and throughput.

Exit criteria:

- A pretrained checkpoint exists for the merged graph.
- Intrinsic metrics are logged for validation and test splits.
- The run is reproducible from a config file and a committed command.

### Phase 2b: Content and event-window ablations

Goal: isolate how much transfer comes from topology, content, and stability across event windows.

Tasks:

- Run structure-only, content-only, and structure-plus-content variants where the loader supports them.
- Evaluate pre-event to during/post-event transfer on each graph with a known event anchor.
- Measure feature distribution shift across event windows for bio/content embeddings.
- Report absolute performance and degradation relative to the pre-event evaluation.

Exit criteria:

- At least one content ablation is complete on each of the three core datasets.
- At least two datasets have pre/during/post-event evaluation results.
- Results distinguish content contribution from topology contribution.

### Phase 3: Scaling sweep

Goal: test whether transfer improves with data or model scale.

Minimum sweep:

| Sweep axis | Levels |
|---|---|
| Pretraining data | 1 dataset, 2 datasets, 3 datasets |
| Model capacity | small, medium, large |
| Seeds | at least 3 for main comparison |

If compute is limited, prioritize data scale over model scale because the main question is whether broader pretraining data improves transfer.

Required outputs:

- Scaling table with mean and standard deviation.
- Transfer curve figure.
- Compute table with GPU hours and peak memory.
- Notes on failed or unstable runs.

Exit criteria:

- At least one sweep axis shows a monotone or mostly monotone transfer trend.
- The result is not driven by a single seed.
- The large run fits within the 4xH100 practical compute constraint.

### Phase 4: Transfer evaluation

Goal: show that pretrained representations transfer to held-out event graphs better than alternatives.

Tasks:

- Evaluate pretrained checkpoints on each held-out dataset.
- Compare against from-scratch training with matched architecture and budget.
- Compare against at least two standard GNN baselines.
- Report performance deltas with confidence intervals or seed variance.
- Include both same-task transfer and cross-task transfer where supported.
- Include event-window transfer where event anchors are defined.

Primary metrics:

- Link prediction AUC.
- Link prediction average precision.
- Neighbor matching accuracy/F1/ROC AUC, depending on current evaluator output.
- Reconstruction accuracy for masked objectives.
- Content reconstruction quality for masked content objectives.
- Event-window degradation, measured as relative and absolute metric drop from pre-event to during/post-event windows.

Secondary metrics:

- Few-shot classification accuracy/F1 on labeled tasks.
- Calibration or variance across held-out graphs.
- Degradation under reduced graph observation, if cheap to run.

Exit criteria:

- Pretrained model beats from-scratch matched model on the majority of held-out graph/task pairs.
- Pretrained model beats at least two standard GNN baselines on the primary held-out transfer setting.
- Pretrained model has lower average event-window degradation than from-scratch and single-event baselines.
- Results include variance across seeds or repeated runs.

## Success criteria

### Full success

This experiment is successful if all of the following are true:

- A reproducible graph construction pipeline exists for at least three event graphs.
- A pretrained graph-model checkpoint is produced from multi-graph pretraining.
- Content features, starting with bio/profile embeddings, are included in the main graph artifacts or explicitly ablated.
- Target intrinsic metrics are met or exceeded:
  - greater than 90% masked node/edge reconstruction accuracy;
  - greater than 0.85 link prediction AUC.
- Transfer improves over a matched from-scratch model on held-out datasets/events.
- Transfer improves over at least two standard GNN baselines.
- Event-window degradation is smaller for the pretrained model than for matched from-scratch and single-event baselines on at least two core datasets.
- A scaling curve shows that increasing pretraining data volume or model capacity improves transfer.
- The entire pipeline can be rerun from documented configs and commands.

### Partial success

The experiment is still useful, but does not meet the full finish line, if:

- The graph construction pipeline works and pretraining converges, but transfer gains are mixed.
- Pretraining improves temporal LP but not downstream classification.
- Content features improve some datasets but not others.
- Event-window degradation improves on one dataset but not consistently across event types.
- Scaling improves intrinsic metrics but not held-out transfer.
- Transfer improves over from-scratch but not over strong GNN baselines.

Partial success should produce an internal technical report and a narrowed follow-up plan.

### Failure criteria

The main goal is not supported if:

- Multi-graph pretraining does not outperform from-scratch training on held-out graphs.
- Scaling data/model size does not improve intrinsic or transfer metrics after accounting for variance.
- Structure-plus-content models do not improve over structure-only models and do not explain transfer behavior.
- Event-window degradation is no better than from-scratch or single-event baselines.
- Results are unstable across seeds or depend on one dataset only.
- Graph construction artifacts are not reproducible enough to audit.

## Deliverables

| Deliverable | Contents | Done when |
|---|---|---|
| D1. Graph artifacts | `.pt` graphs and `.meta.json` files for source datasets | Loaders and smoke tests pass |
| D2. Pretraining configs | YAML/SLURM configs for small, medium, and large runs | Runs can be reproduced from repo commands |
| D3. Pretrained checkpoints | Best checkpoints for each scale level | Checkpoints load and evaluate |
| D4. Evaluation outputs | JSON/CSV summaries for intrinsic, transfer, content-ablation, and event-window metrics | Aggregated results include seeds and splits |
| D5. Figures | Scaling curve, transfer heatmap/table, content ablation, event-window degradation, baseline comparison | Figures are generated from saved result files |
| D6. Scope-completion memo | Short report stating success, partial success, or failure | Memo references exact runs and metrics |

## Experimental matrix

Minimum viable matrix:

| Run group | Pretraining graph | Evaluation graph | Tasks | Comparison |
|---|---|---|---|---|
| Smoke | Each individual graph | Same graph | NM, temporal LP | Sanity only |
| Leave-one-event-out | Two-event merged graph | Third event graph | NM, temporal LP | Pretrained vs from-scratch |
| Full merged | Three-event merged graph | Each event graph | NM, temporal LP | Pretrained vs single-source |
| Content ablation | Structure-only/content-only/combined graph variants | Same and held-out graphs | NM, temporal LP, content reconstruction | Combined vs ablated variants |
| Event-window stress test | Pre-event/history graph view | During/post-event views | NM, temporal LP, content reconstruction | Pretrained degradation vs baselines |
| Baseline | Each target graph | Same target graph | LP, NM | Pretrained model vs GraphSAGE/GAT/TGN |
| Scaling | 1, 2, 3 source graphs | Held-out graph | LP, NM | Scale trend |

Recommended reporting table:

| Model | Pretraining data | Target | Window | Feature setting | Task | Metric | Mean | Std | Delta vs scratch | Delta vs best baseline |
|---|---|---|---|---|---|---|---|---|---|---|

## Compute assumptions

The main run should be feasible on a 4xH100 node. The scope should avoid requiring planetary-scale full-corpus pretraining for the first result.

Compute plan:

- Use capped smoke runs locally or on one GPU before any large job.
- Use medium runs to debug the merged graph and checkpointing.
- Reserve 4xH100-scale runs for the final sweep only.
- Log GPU type, number of GPUs, wall-clock time, peak memory, dataset caps, and effective sampled episodes.

## Reproducibility requirements

Every result used in the final analysis must have:

- Git commit hash.
- Dataset artifact path and metadata hash or timestamp.
- Content feature source, embedding model, and feature ablation setting.
- Event anchor and window definition, when applicable.
- Run config.
- Random seed.
- Checkpoint path.
- Evaluation command.
- Raw metrics file.
- Aggregated metrics row.

No figure should be hand-entered. Figures should be regenerated from saved evaluation outputs.

## Open decisions

These choices should be made before the main scaling sweep:

- Whether masked node/edge reconstruction will be implemented as a first-class task or approximated through existing neighbor matching and temporal LP metrics for the first pass.
- Which content objective is primary for bio/profile embeddings: reconstruction, retrieval, contrastive alignment, or ablation-only.
- Which event anchors define pre/during/post-event windows for each dataset.
- Whether model scaling or data scaling is the primary sweep axis.
- Which two standard GNN baselines are mandatory if all three of GraphSAGE, GAT, and TGN cannot be finished.
- Whether classification tasks are included in the main result or kept as supplemental probes.
- Whether non-Twitter datasets are required for the first complete version or deferred.

## Working summary

This experiment is about building and testing a reusable graph pretraining pipeline. It does not try to infer incentives or forecast future reactions. It asks whether a self-supervised graph model trained on multi-event temporal social graphs learns structure-plus-content representations that transfer to held-out datasets, degrade less across known event windows, and improve with scale.

The finish line is a reproducible three-dataset graph pretraining pipeline, target intrinsic metrics, explicit content-feature handling, and a positive held-out transfer delta over from-scratch and standard GNN baselines. The stronger version adds a clean scaling curve, a content ablation, event-window degradation results, and variance across seeds.
