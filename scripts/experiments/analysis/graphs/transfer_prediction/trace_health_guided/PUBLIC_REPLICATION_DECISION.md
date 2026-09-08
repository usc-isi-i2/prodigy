# Public replication: decision before target outcomes

7 September 2026. Native training is live on Tucker in `publickg-train`, PID
4118917, isolated worktree `/dataMeR1/phil/gfm/prodigy-publickg-native`.
The launcher is revision `4c14de12`; its unmodified upstream is pinned to
`107ba57234d3188227cda5b78a2dbcfb84a1c694`.

## What is actually being tested

The implemented public reproduction is Wiki pretraining followed by
FB15K-237 20-way/3-shot relation classification, 500 episodes. It is not the
earlier proposed MAG-to-arXiv experiment. The checkpoint is already nominated
as `state_dict_8000.ckpt`, after 8,001 updates, independent of target outcomes.
Do not choose another checkpoint because it gives a better intervention result.

Native architecture is S2,UX,M2. There are two metagraph layers; query outputs
may therefore change under a support intervention. Native eval-only execution
also retains training mode and BN updates. These properties must be represented
in the scientific claim rather than silently changed for convenience.

## Minimum decisive sequence

1. Complete native evaluation and capture its actual batches, logits, and state
   progression. Establish that a paired unmodified replay reproduces native
   outputs. A single matching aggregate accuracy is insufficient.
2. Retain the existing support-context donor intervention only if its definition
   has a faithful counterpart for relation examples. Verify the head/tail flags,
   center-edge removal, support/query masks, and label relation identities.
   Do not equate a KG relation example with a social account by assumption.
3. Apply the frozen key-versus-value contrast on identical episodes and model
   states. For each pair restore the same pre-forward BN buffers and RNG state;
   retain the native mode. Otherwise donor ordering can become an intervention.
4. Report both the total effect and final class-reference substitution using
   cached native final queries. Label the latter a controlled downstream
   intervention, not proof that upstream support changes leave queries fixed.
5. Primary mechanistic comparison: paired within-episode macro one-vs-rest AUC
   change for value versus key replacement. Report native and changed accuracy,
   macro-F1, and NLL alongside ranking. Include the sign by episode and an
   episode-paired interval, described as conditional on this checkpoint/task.

Do not pool probabilities across episode-local relation columns without mapping
their global identities. Do not substitute pooled binary AUC for the multiclass
endpoint. AUC gains accompanied by a prediction collapse are not a usable repair.

## Decision rule

If native transfer is credible and value dominance survives, the study supports
the class-reference explanation beyond the custom social deployment. If native
transfer is weak, or keys dominate, record that boundary. No threshold or new
donor is to be selected after seeing these outcomes. A single public target does
not establish universality, and no outcome automatically establishes an ICLR 8.

The support-value mechanism is the current paper candidate. TRACE agreement and
the failed source-allocation rule are supporting/limiting evidence. The formula
`c*a + (1-c)*(1-a)` is a heuristic: interpreting it as query correctness requires
conditional readout reliability to equal the estimated support competence.
Correlation alone does not establish that equality.
