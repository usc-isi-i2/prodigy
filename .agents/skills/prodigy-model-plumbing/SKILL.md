---
name: prodigy-model-plumbing
description: Change or diagnose model layers, samplers, episode construction, datasets, batching, objectives, losses, task routing, or checkpoint state in this repository. Use for cross-layer training or evaluation semantics across PRODIGY and comparison models; do not use for experiment configuration alone.
---

# Model and Episode Plumbing

Protect the scientific semantics of changes that cross sampler, data, model, loss,
trainer, and evaluator boundaries. A forward pass that runs is only a smoke check; it
does not establish episode, objective, or evaluation parity.

The `prodigy-` name is retained for repository-skill consistency. Use the common
workflow for other model families where they share repository plumbing, and respect
model-specific contracts where they do not.

## Before acting

1. Read [model architecture](../../../docs/agent_guides/model_architecture.md).
2. Trace the closest existing implementation through the sampler, dataset/dataloader,
   model, objective, trainer, checkpoint, and evaluator paths that actually apply.
3. If packaging a research experiment around the change, also use
   `prodigy-experiments`. If verification is heavy, also use `prodigy-tucker`.

## Change contract

State before editing:

- the intended behavioral or scientific change;
- the controlled invariants that must remain unchanged;
- affected model and task families;
- expected episode cardinality, source membership, support/query roles, tensor
  shapes, label flow, and loss behavior;
- compatibility expectations for configs, CLI arguments, logs, and checkpoints.

If the requested change necessarily alters an invariant, make that explicit rather
than describing the result as matched or backward compatible.

## Workflow

1. Follow configuration values end to end; unknown task names can fail late, and a
   flag accepted by one layer may be ignored by another.
2. Check episode construction and sampling before model output: source restrictions,
   center/member identity, support/query separation, context expansion, deterministic
   streams, and batch collation are part of the method.
3. Check where labels and task information enter the model, including metagraph edge
   attributes, label embeddings, auxiliary heads, and task-specific losses.
4. Audit every affected task-family branch and train/eval mode. Do not validate only
   the path exercised by the motivating config.
5. Preserve checkpoint loading and resume semantics, or fail clearly with a documented
   migration boundary. New state must not silently disappear on resume.
6. Add focused tests at the lowest responsible layer plus a lightweight integrated
   contract test when behavior crosses layers. Test meaningful invariants rather than
   merely checking that execution completes.
7. Compare resolved configs and deterministic sample, role, or fingerprint artifacts
   before and after when parity is claimed.
8. Label local or short-run execution as smoke validation. Use matched evaluation
   evidence before claiming that performance or research conclusions are preserved.

## Required output

Report the affected pipeline, semantic change, preserved invariants, compatibility
boundary, tests performed, and any heavy validation still required. For research
changes, point to the experiment that isolates the change from confounders.

## Maintenance

Keep the conceptual architecture in `model_architecture.md` and keep this skill
focused on change discipline. Expand its scope when shared repository plumbing gains
new model families or stages; put genuinely model-specific details with that model
rather than turning this entrypoint into an exhaustive architecture manual.
