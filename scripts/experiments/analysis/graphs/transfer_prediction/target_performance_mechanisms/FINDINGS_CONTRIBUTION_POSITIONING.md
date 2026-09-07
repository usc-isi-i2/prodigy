# Contribution decision after the public boundary and related-work check

7 September 2026. Advisor decision; private, uncommitted. No new forwards,
manuscript expansion, or experiment launch. Worktree `.worktrees/role-topology`,
branch `codex/role-topology-interactions`, code `30df8cb5`.

## Decision

The defensible center is **transfer failure in support-to-class-reference
construction**, not discovery that context constructs a classifier, a novel
K/V decomposition, or a generally useful suppression method. A candidate
headline is: **When graph context changes the classifier, not the query**.
This is a framing candidate, not a claim of established publication-level
novelty. The scientific delta is a learned failure localized with fixed
queries and attention, plus the distinction between ranking repair and
decision repair.

## Closest prior work changes what we can claim

- [Function Vectors](https://arxiv.org/abs/2310.15213) already establishes
  causal task representations transported by attention heads. Context forming
  a task representation is therefore background, not our discovery.
- [How Few-Shot Examples Add Up](https://arxiv.org/html/2605.16591v2),
  particularly section 5, already uses a contextualized/uncontextualized
  QK-by-V factorial. Its routing effects are more consistent than its value
  effects. Our value-dominant result is not a contradiction: tasks, models,
  interventions and metrics differ. We must credit the shared decomposition
  idea and emphasize transfer failure rather than methodological firstness.
- [Better with Less](https://arxiv.org/html/2311.01038v2) turns nonmonotonic
  pretraining utility into model-dependent graph/sample selection. The useful
  framing lesson is to connect an observation to a consequential decision.
  Its uncertainty criterion and our support-CV loss are different objects;
  our CV failures do not refute its claims.

This is a focused overlap check, not an exhaustive novelty search.

## The argument the existing figure should carry

1. **Learned failure, not a static distance story.** Fixed-input training
   trajectories change support-context utility. Nine-source intervention
   heterogeneity makes Hong Kong an extreme case, not the only affected source.
   Neither observation alone supplies a sign predictor.
2. **Where the failure enters.** In Hong Kong-to-political, changing support
   values improves within-episode ranking with attention and query vectors
   exactly fixed. The value-over-key ordering survives the nominated 50k
   configuration. This is one source-target pair, not nine-source mediation.
3. **What improvement means.** At 50k, suppression raises within-episode AUC
   from about .564 to .659 but lowers accuracy from .492 to .251. Saved
   reference geometry and actual query examples explain this coexistence:
   discrimination changes while a surviving class preference dominates the
   decision boundary. AUC improvement cannot be sold as classifier repair.
4. **Strong counterexample.** The useful original-style public model needs
   context in both roles: native accuracy .738 falls to .399/.390 with
   support/query suppression. Final-state recombination associates the larger
   loss with the corresponding reference/query component; it is endpoint
   accounting, not fixed-query causal mediation or successful generalization
   of the harmful-support claim.

## Community takeaway and remaining acceptance-critical gap

Evaluate transfer of the reference constructor separately from query-feature
quality; distinguish attention routing from transported content; and report
decision behavior beside ranking metrics when diagnosing reference changes.
These are evidence-backed diagnostic lessons, not a deployable method.

The missing high-impact result remains a computation-level condition that
predicts failure outside discovery. The public test cannot be relabeled a
successful replication, and a generic K/V study now has especially weak
novelty. Do not launch another setting search. Before any new experiment,
the researcher must nominate one specific condition, explain why existing
evidence supports it over alternatives, and state a falsifiable prediction
on already-designated non-discovery sources. An outcome-derived correlation
or a geometry statistic mathematically equivalent to target performance does
not qualify. If no such condition survives the evidence audit, frame this as
a bounded diagnostic study rather than assert that wording alone achieves
the requested 8+ contribution.

Authoritative local evidence: `FINDINGS_CLASS_REFERENCE_KV.md`,
`FINDINGS_CLASS_REFERENCE_ACCOUNTING.md`, `FINDINGS_PUBLIC_BOUNDARY.md`, and
`FINDINGS_SUPPORT_WEIGHTING.md`. The existing one-page PDF is unchanged.

## Updated positioning after the matched page prototype trajectory

7 September, code 8ccd3686. Focused primary-source check, not an exhaustive
novelty certification. The earlier introduction to this note predates the
initialization, offset and matched-page results; use this update for current
claims.

- [Luo et al., ICML 2023](https://proceedings.mlr.press/v202/luo23e.html)
  explicitly separates training from adaptation in few-shot classification.
  We cannot claim novelty for evaluating a fixed readout on learned features
  or for disentangling those phases in general.
- [Raghu et al., ICLR 2020](https://openreview.net/pdf?id=rkgMkCEtPB)
  distinguishes feature reuse from rapid adaptation and tests simplified
  alternatives. Our prototype advantage alone is not a new scientific principle.
- [PRODIGY, NeurIPS 2023](https://cs.stanford.edu/~jure/pubs/prodigy-neurips23.pdf)
  explicitly contextualizes both prompt examples and queries, then performs
  learned communication through a task graph. Its claimed transfer setting
  motivates asking which role-specific computation transfers, rather than
  treating prompt-graph construction itself as our discovery.
- [Better with Less, NeurIPS 2023](https://arxiv.org/html/2311.01038v2)
  connects nonmonotonic data utility to adaptive selection and tests the resulting
  method. Borrow the observation-to-decision structure, not its method claim:
  neither our failed selector nor offset subtraction provides that payoff.

### Current central claim, stated positively and narrowly

**End-to-end context utility is not the same as the usefulness of the
information encoded from context. In graph ICL, support-side interventions
can change the class reference while leaving queries fixed, and the measured
benefit depends on how the learned readout uses those representations.**

The matched page example makes this concrete. At step100, suppression improves
prototype AUC by 7.93 points but worsens native AUC by 3.41. At step2500 both
benefit (1.76 and .89 points). The prototype does not change the sign of its
context preference; native inference does. Therefore the native sign reversal
cannot simply be equated with encoded context information changing from useful
to harmful. This is a comparison between two actual readouts, not a claim about
all information-theoretically recoverable signal.

The proposed account that training progressively makes the constructor misuse
still-useful context was contradicted. Both paths improve in absolute ranking.
The sharper interpretation is that the learned readout increasingly uses a
suppressed representation which was already useful to a fixed prototype.
Since the readouts differ in query processing too, this does not isolate one
class-reference projection as the cause. Keep that distinction in the title,
abstract and figure captions.

### Candidate introduction paragraph (private, not inserted into manuscript)

Graph in-context learners use neighborhoods twice: to describe a query and
to construct the class reference against which it is judged. These roles make
transfer performance difficult to interpret. A failure on an unseen graph may
reflect uninformative representations, or a learned inference rule that fails
to use information already present. We investigate this distinction through
matched role interventions, fixed readouts and training trajectories. We find
that the measured usefulness of support context can reverse during training
even when a fixed prototype retains the same preference, and that superficially
similar suppression gains can have very different decision stability. These
results motivate diagnosing the transfer of graph representations and their
use in class-reference construction jointly, rather than reading an end-to-end
score as a property of the graph or encoder alone.

### Decision for subsequent work

Do not market the generic representation/readout distinction as the contribution.
The differentiator must be the role-specific, matched evidence and a consequence
that changes how graph ICL is trained or evaluated. The already-running
encoder-solver isolation study is relevant to the training consequence; it has
not yet supplied transfer evidence here and must not be preemptively counted.
No duplicate training study, new readout sweep or stronger novelty wording is
authorized by this positioning update. The core unresolved issue is why the
native readout cannot use the early available signal, not whether that signal
exists. This remains an active scientific question, not a completed 8+ claim.

## After the completed isolation test: do not substitute a familiar headline

The isolation study now has completed target outcomes (see the current
`FINDINGS_PAPER_SYNTHESIS.md`). It fails its prespecified method gate. Its
positive comparison, that native/joint training can yield better fixed-readout
features while the full solver remains weaker at inference, is supporting
evidence rather than a new paper identity by itself.

Two further primary-source overlap checks reinforce this decision:

- [Tian et al., ECCV 2020](https://arxiv.org/abs/2003.11539) already demonstrates
  that learned representations followed by a linear classifier can outperform
  sophisticated few-shot methods. A strong replacement readout is not new.
- [Hou and Sato, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a559a5a8aa5ae6682ced009ad97cdb16-Abstract-Conference.html)
  studies prototype generalization, normalization and within-/between-class
  variance. Generic geometry explanations also need a specific new predictive
  claim, not just a familiar variance-ratio correlation.

Target-level evidence prevents a universal harmful-solver framing: native
training improves ridge AUC over ridge-only in both streams on three of four
supported blocked targets (TwiBot20 reverses), and all four interleaved targets.
But the native full solver exceeds its ridge readout on blocked political
classification by1.60–2.04 AUC points. Training utility and inference utility
are different empirical comparisons; neither is uniformly positive/negative.
No manuscript expansion or new experiment follows from this literature check.

## Construction-dependent trajectories: closest-prior-work decision (7 September)

This section supersedes the earlier candidate headline, not its observations.
The matched follow evaluation is now complete; see FINDINGS_TRAJECTORY_DECISION.md.

- [Hu et al., ICLR 2020](https://arxiv.org/abs/1905.12265) establishes negative
  transfer from naive graph pretraining. Negative transfer itself is not ours.
- [Mai et al., NeurIPS 2024](https://arxiv.org/html/2409.16223v2), introduction,
  studies improved features alongside degraded full-model
  accuracy, localizes the damage to class-group logit bias, and repairs it by
  calibration. Thus even the combination of better features, worse predictions,
  and component localization is established. Our binary within-episode AUC
  changes concern ordering, not only a shifted decision threshold; this does
  not rule out richer sample-dependent scale effects.
- [Bechler-Speicher et al., ICML 2024](https://arxiv.org/html/2309.04332v2),
  sections 2–3, already compares graph distributions with fixed node features,
  relates failures to degree-dependent computation, and proves extrapolation
  failures for a graph-less teacher. Saying graph construction matters is
  insufficient novelty. Their framework motivates demanding a learned
  computational explanation rather than listing degree correlations.

The candidate delta is narrower and more specific: in graph ICL, fixed-query
role interventions distinguish class-reference use from query representation;
continued source pretraining improves pooled-feature readouts yet loses an
early target-specific ranking advantage; a matched-account relation replacement
tests a nominated consequence of the cue account. These findings currently
form two connected research questions, not one demonstrated causal chain.
Do not silently treat the support-reference localization as explaining the
retweet-to-follow trajectory interaction.

Advisor decision: retain construction-dependent negative transfer as the
current empirical lead, but do not claim an 8+ mechanism on this basis alone.
First perform the equal-early-opportunity transition accounting nominated in
the decision log. This determines whether the relation experiment adds more
than removal of extra early wins. No new training sweep or manuscript expansion.
The literature check is focused overlap assessment, not a novelty certificate.

### Decision updated after the nominated accounting completed

The follow result does not demonstrate better preservation. Pairs with equal
early ranking credit comprise about 54% of comparisons and favor retweet in
the training interaction (−2.35/−1.42 AUC-point contributions). Initial
disagreements supply more than the net positive follow-minus-retweet change.
Consequently construction dependence is corroboration/boundary evidence,
not the missing explanatory bridge. Keep cue-conditioned ranking tradeoffs
as the empirical lead and retain the support-reference localization as a
separate mechanistic result until connected. No broad relation sweep follows.
The missing contribution is still why the learned computation loses that
target-rewarded advantage; renaming it negative transfer does not solve it.

## Contribution decision after independent review of the consolidated brief

7 September, current code6c88c0bc. The synthetic task-switch and support-exception
branches are closed; completed evidence is in DECISION_TASK_SWITCH.md.
The new one-page private argument is
`output/pdf/class_reference_contribution_decision.pdf`, built by
`build_contribution_brief.py`, rendered and visually verified as one letter page.
It preserves the nine-source map, K/V dissociation, longer-trained result,
public boundary and newer capability counterexamples without manuscript expansion.

Independent contribution review changes the research gate: **a universal or
prospective context-sign predictor is not necessary for a strong scoped
mechanism paper.** Earlier notes treating it as the sole acceptance-critical
route overconstrained the problem and risked an indefinite predictor search.
The distinctive supported claim is empirical, not the fact that the architecture
constructs references: support values change discrimination with attention
weights and query representations exactly fixed, while engaged key changes
produce much smaller gains in the studied political transfer.

The critical breadth gap is that this specific pathway explanation still
concerns one source-target pair. The nine-source role map does not establish
nine-source K/V dominance;50k is a second configuration, not a second pairing,
and its accuracy collapse limits practical significance. The useful public
counterexample is boundary evidence, not a positive mechanism replication.

If another experiment is pursued, nominate exactly one natural source-target
pair independently of its unseen K/V outcomes, freeze the pathway ordering,
and test that ordering with original architecture and both complete streams.
This is a more direct route to the missing generality than another synthetic
rule, noise-level sweep or geometry predictor. No new pair or run is selected
in this review turn. The strong contribution goal remains open; the current
evidence is a credible bounded causal diagnostic, not a completed8+ claim.

## After the nominated natural-pair prediction: pathway dissociation is the lead

7 September, code65946f82. This supersedes the preceding breadth-gap assessment.
Ukraine-to-TwiBot20 passes the frozen K/V ordering in both streams. Values-only
changes within-episode AUC by -6.42/-4.98 points; keys-only by +1.19/+.93.
Key attention TV is .071/.072, values-only TV is exactly zero, and queries are
bit-exact throughout. Accuracy follows the same signs. Full evidence and limits:
`DECISION_KV_GENERALITY.md`.

The intellectual center is now **opposing routing and content consequences of
the same support-context intervention**, alongside opposite value effects across
political and bot tasks. This is stronger than a second example that values
matter: a beneficial attention change can coexist with a substantially harmful
content change. End-to-end suppression effects cannot diagnose misrouting.

Advisor decision, independently reviewed: pursue the mechanism paper and stop
this experimental block. No new control sweep or manuscript expansion here.
The updated figure-led brief keeps the nine-source role map, both pathway cases,
50k persistence and public/accuracy boundaries visible. The contribution is a
scoped empirical dissociation in class-reference construction, not a universal
sign predictor. Explaining the content's utility through norms, directions or
operating ranges remains unresolved and must not be implied by the title.

### Full-factorial and closest-prior-work audit

The subsequent `FINDINGS_KV_CLAIM_AUDIT.md` qualifies the generic dissociation
headline. The closest function-vector paper already reports helpful/harmful
value effects, including AppendixL geometry caveats; this is not ours to claim
as a new general phenomenon. Our scientific case is role-specific transfer of
the native graph classifier and the diagnostic consequence of separating its
reference constructor from queries, routing and decision calibration.

New accounting of existing outcomes confirms that the political/bot value
sign survives both key endpoints in all five measured cells. The routing
effect does not: fresh bot key replacement is +.928 points at native values
but -.353 at donor values. Keep the observed native-anchor opposition, but
do not claim universally beneficial routing or additive pathway effects.
No new experiments or manuscript edits follow from this audit.

## Main-text replacement after the one-page decision

7 September2026. Advisor judged the argument coherent enough to write a scoped
main-text replacement, without claiming the high-impact goal is complete.
New private paper artifact outside the repo:
`/Users/philipp/projects/gfm/paper/transfer-prediction/mechanism-draft-2026-09-06/manuscript-class-reference.md`.
Title: **Separating Query and Class-Reference Transfer in Graph In-Context Learning**.
The older repair/trajectory drafts and compiled LaTeX remain unchanged.

Independent review reconciled the pathway numbers and identified two important
omissions, both now repaired in this replacement. The50k training-label-table
restoration shrinks suppression gains to+.34/+1.87 and removes the near-all-
positive collapse; its K/V ordering has not been tested. The public saved-state
hybrids support functional query/reference separation despite the failed
harmful-support generalization. Cross-role coupling is observed; do not causally
attribute it solely to normalization. Both figures are linked and paths verified.

This replacement integrates the nominated pairing, full conditional contrasts,
episode/example accounting, interfaces and prior-work boundaries. It is a main-
text candidate, not a compiled submission. Next integration requirements are
figure/table numbering, verified citations, appendix/protocol reconciliation
and rendering; scientific artifacts remain private and reviewer access TBD.
No experiment was launched and nothing pushed in this drafting turn.

### Compiled replacement, separate from the old submission-shaped artifact

The new private main text now compiles to
`paper/transfer-prediction/mechanism-draft-2026-09-06/output/pdf/class_reference_transfer_draft.pdf`
under the GFM sibling paper directory. Five main-text pages, eight total;
two figures, three result tables, six cited references and a focused appendix.
All eight final rendered pages were visually inspected. The build resolves
citations, reports zero overfull boxes, and preserves the official style,
existing AI/ethics statements, old main.tex and older PDFs. Reviewer access TBD.

The focused appendix covers conditional contrasts, aggregation provenance,
label-interface moderation, public hybrids and inference invariants. It does
not pretend the older exhaustive controls have been fully migrated. That
appendix reconciliation, stronger experiment-setup specification and final
scientific assessment remain incomplete. No claim of goal completion follows
from successful document compilation. No code/data/paper push this turn.

### Recorded-methods reconciliation

The next methods audit read Tucker's saved short-run cache protocol, public
native effective_params/protocol, local50k recorded parameters and the cached
revision97cc7704 configuration. Main text and class-reference appendix now
specify source/target names, episode counts, sampling, label interfaces and
the public runtime's preserved split/mode behavior. The50k case is explicitly
a different historical configuration, not a matched training-length test.
Public training10,010 updates is distinguished from the evaluated8,001-update
checkpoint (filename8000). The short-run K/V member-control checkpoints are
distinguished from the corrected-sampler trajectory models. No new forwards.

The Markdown sources contain these methods additions; the eight-page PDF from
the preceding build predates them and must be rebuilt and visually checked
before claiming they are present in the compiled draft.

The rebuilt PDF now includes those methods: nine total pages, main text still
ending on page5. All nine rendered pages were inspected; no overfull boxes or
undefined citations. QA caught and fixed a builder typography rule that inserted
spaces into revision hashes and code identifiers. Extracted PDF text now retains
75f0853f,97cc7704,107ba572,ignore_label_embeddings=False and effective_params.json
exactly. Current PDF SHA256:
0f4416c742204549ed2262902e09a8d5f3d9a558c9a40a94d31e8903a9b74080.
The main-text source SHA256 is d02574d8b492b0fa2304fdbeca927984649d3e6f7f59b4c48472ae8172944df0.
The historical appendix migration and scientific contribution assessment remain
open; PDF completion does not satisfy the overall publication goal.
