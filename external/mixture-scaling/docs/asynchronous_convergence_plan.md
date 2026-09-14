# Loss switching after asynchronous convergence

Status: pilot completed, 2026-09-12: three training arms, verified singleton replays, and 40 downstream evaluation cells. [Results and reproducible figures](../results/async_convergence/FINDINGS.md) report improved transfer versus both BCE controls with a remaining deficit versus the stronger constituent singleton. Keep the extended-exposure/stopping control; prioritize loss switching before sampling reweighting, architecture changes, or ForkMerge.

## Primary sources and scope

Lu et al., NAACL 2022 Industry Track, Section 3 and Appendix A: per-task validation patience identifies convergence; the shared model rewinds to that task's best checkpoint, which supplies soft targets replacing true-label training for that task. Joint training ends when all tasks converge. Optimizer/sampler rewind details are unspecified. Section 4.1 monitors test sets for three tasks lacking validation; we must use our separate source validation sets. This is not a universal retention guarantee.
https://aclanthology.org/2022.naacl-industry.18.pdf

Jiang et al., ForkMerge, NeurIPS 2023: branch from common weights, train target-only and target-plus-auxiliary models, and choose a convex parameter mixture using target validation before synchronizing and repeating. The target-only endpoint protects only against that round's candidate on the searched validation metric. It does not guarantee both-source or unseen-graph preservation. Official code retains branch optimizer histories while loading/merging model weights, and merges validation-best branch checkpoints.
https://papers.nips.cc/paper_files/paper/2023/file/60f9118a849e8e9a0c67e2a36ad80ebf-Paper-Conference.pdf
https://github.com/thuml/ForkMerge/blob/develop/auxiliary_domain_image_recognition/forkmerge.py

Mueller et al., TMLR 2024: compare per-task generalization at comparable own-task training loss. Their gradient-conflict evidence concerns optimization and within-task generalization; it does not establish a reliable generalization predictor. Detailed methods were checked in the January 2025 arXiv v2, not assumed identical to an inaccessible accepted PDF.
https://openreview.net/forum?id=QQE5j2OsLW
https://arxiv.org/html/2408.14677v2

Our gradient/update probes are model-specific evidence consistent with this literature, not a new general mechanism. The papers' task generalization and our transfer to graphs excluded from training are different outcomes.

## What changes relative to our ranking pilot

Our completed pilot retained true-label BCE on both graphs and added a ranking KL term for the initial singleton source. Its teacher was the singleton at the start of continuation. That ranking loss ignores a shared score shift, including the decoder bias.

The proposed graph adaptation uses the best JOINT checkpoint for whichever source converges first. The remaining source still uses true-label BCE. The converged source uses only soft-target BCE, computed on its training edges and sampled nonedges with the frozen teacher. The teacher remains in evaluation mode without gradients.

For a converged source, define teacher probability q = sigmoid(z_teacher). Its training term is BCEWithLogits(z_student, q), with coefficient 1 and no extra temperature in the first pilot. Log KL by subtracting the fixed teacher entropy if a zero-at-teacher diagnostic is useful. Do not add true-label BCE to this optimization term. Continue measuring true-label BCE separately under no_grad for comparison with singletons.

Online teacher predictions on freshly sampled training nonedges are our adaptation to the existing dynamic sampler. They are not cached pseudo-labels over a fixed finite classification dataset.

## First comparison and controls

Start with Ukraine + Facebook, which showed disparate exposure requirements and stopping while Ukraine was improving. Do not alter features, decoder, LR, source order, negative sampling, or architecture simultaneously. Use the original fresh seed-0 initialization and LR 0.0005. Keep the current 1:1 alternating schedule; changing to mixed batches would add another factor.

1. Extended exposure control: true-label BCE remains active on both graphs. Disable termination on mean-BCE plateau, retaining its selected checkpoint as a diagnostic. Preserve the existing 100,000-total-update safety cap, allowing 50,000 updates per source; Ukraine's selected singleton used 44,000. Save model/training state and all metrics every 2,000 updates.
2. Loss-switching intervention: track each source separately. Initial proposed detector: validation AUC, absolute improvement 0.0001 (0.01 pp), patience 3 checks, validation every 2,000 total updates. This metric/tolerance is our prespecified adaptation. Keep raw BCE as a secondary diagnostic. If Facebook converges first, rewind to its best joint checkpoint, freeze that teacher, replace Facebook's hard-label objective, and continue Ukraine supervision. Do not force the identity/order of the first converged source if the detector disagrees.
3. Rewind-only sibling: from the same first rewind checkpoint and full training state, continue hard-label BCE on both graphs up to the exposure-control horizon. Compare it with saved loss-switching checkpoints at equal post-rewind source exposure. This separates loss replacement from the benefit of discarding the patience tail. It is a mechanism control in addition to the ordinary extended run.

The KD arm follows per-task convergence and stops when both tasks have converged, rewinding to the final task's best checkpoint. Do not force an already-converged task to remain supervised just to reach the singleton's exposure. The extended hard-label arms answer the separate sufficient-exposure question. Use 44,000 surviving-path supervised Ukraine updates as a reference landmark; report whether each arm reaches it, converges earlier, or hits the safety cap. The current 100,000 physical-update cap remains; a cap is not convergence.

For the sibling comparison, use the same post-rewind sampling sequence and compare saved points only at equal surviving-path exposure within their common range. Report the KD method's source-validation-determined endpoint and physical compute separately. Terminal rewinds can shorten the surviving path without undoing physical work. Do not select any comparison horizon from downstream scores.

Primary KD readout is its all-converged endpoint; report per-source retention as an outcome, not an assumed property. Controls report their exposure landmarks and the saved points matched to that KD endpoint. Retain original mean-BCE-selected checkpoints as historical diagnostic comparisons. A source-validation-only alternative selection may be exploratory, but it must be labeled separately from the method's endpoint. Avoid attributing an extended-control-versus-KD gain solely to distillation without the rewind-only comparison.

## Rewind state and accounting

Restore model weights, decoder bias, AdamW moments/counters, RNGs, per-source order permutations, negative-sampling generator states, and sampler offsets from the rewind checkpoint. This full-state restoration is our explicit implementation choice; it is not documented optimizer behavior in the NAACL paper. Current checkpoints lack exact sampler resume, so implementation must add and validate it before claiming paired replay.

Keep an immutable event record: trigger step, teacher/best step, restored checkpoint, discarded steps, converged-source identities, and chosen metric. Roll back unfinished-task best/patience tracking to the restored trajectory rather than retaining a best point whose weights were discarded. Preserve the newly declared converged task and its teacher across that rewind.

Track both physical work (including discarded/replayed updates and teacher forwards) and the surviving trajectory. Record per-source supervised updates, distillation updates, positive-example counts, and sampled negative counts separately. Distillation exposure is not supervised exposure. Save checkpoints under unique physical-step/event identifiers so rewinds cannot overwrite earlier artifacts.

## Matched-loss diagnostic

Create fixed, independent training probes from each source's supervision edges with the unchanged 1:5 negative distribution. For example, use 20,480 positives per source or all available if fewer, and persist pair identities and seeds. Evaluate them without gradients at initialization and every saved checkpoint using private generators.

Use the identical probe for singleton, extended interleaving, rewind-only, and KD models. Call the quantity fixed-probe training BCE: it estimates the training objective rather than enumerating every possible negative. Retain the same TRUE-label BCE diagnostic after loss switching; never compare KD loss with singleton supervised BCE.

Plot source-validation AUC against both consumed source examples and own-source fixed-probe BCE. Restrict loss matching to overlapping observed ranges, report residual mismatches, and retain chronology for nonmonotone segments. Matching training loss is a diagnostic conditioning comparison, not proof that all optimization effects are controlled. Historical window-average minibatch losses are not interchangeable with these fixed probes.

## Evaluation and claims

Use source validation exclusively for teacher choice, convergence, checkpoint selection, and retention constraints. Teacher pseudo-labels and gradient updates use training pairs only. Do not tune patience, KD weight, training duration, checkpoint, or ForkMerge weights with final test scores.

Report separately:
- Preservation of each source's own selected singleton on that source.
- Preservation of the converged task's best joint teacher.
- Gain on the unfinished source.
- Transfer to graphs excluded from training, after freezing the method and checkpoint rule.

The repeatedly inspected downstream targets are exploratory development evidence. Untouched confirmation requires new reserved evaluation data or graphs; those historical test results cannot become untouched retroactively. No such new confirmation split is claimed here.

ForkMerge remains the fallback if loss switching is insufficient. Its protected target must be declared, merge weights selected on permitted validation data, and optimizer-history/branch-exposure policy stated. A both-source merge constraint would be our modification rather than standard ForkMerge.


## Pilot implementation

Run from the isolated experiment worktree:
bash scripts/run_async_convergence.sh /dataMeR1/phil/gfm/mixture-scaling/state/async_convergence_s0

The implementation exports the three declared endpoints plus both BCE controls at the KD endpoint's exact surviving exposure, when available, before any downstream evaluation. It records absence of a comparable sibling if the KD endpoint precedes that sibling's origin. Endpoint selection is frozen in evaluation_manifest.json.

Exact state rewind plus identical batches makes the rewind-only arm a deterministic replay check of extended BCE, rather than an independent scientific method. The manifest records their parameter difference at matched exposure. The KD-versus-rewind comparison isolates replacement of the training objective.

Singleton reference replays provide dense fixed-probe training BCE curves to complement existing historical best/final checkpoint probes. They end at the original singleton final steps and verify agreement with original checkpoints; a failed agreement check must be reported rather than presented as historical reproduction.

Five synthetic tests cover exact sampler/Adam replay through partial batches and reshuffling, soft-loss replacement and frozen teacher targets, measurement RNG isolation, best-checkpoint versus patience separation, and two successive rewinds with control exports.
