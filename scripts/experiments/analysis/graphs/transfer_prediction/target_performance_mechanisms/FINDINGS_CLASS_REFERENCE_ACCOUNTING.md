# Contrast compression with a surviving class preference

**Current status:** public training and the nominated evaluation are complete.
Both support and query suppression harm public accuracy; see
`FINDINGS_PUBLIC_BOUNDARY.md`. The execution update below is historical.

**Execution update, 7 September 2026:** the earlier feasibility-only status
below has been superseded by a running original Wiki pretraining job and a
fixed, not-yet-executed FB15K-237 evaluation. See [the private execution
record](../../../../setup/public_prodigy_kg/README.md). No public-target result
is available yet, and this saved-activation conclusion is unchanged.

7 September 2026. A saved-activation analysis, **zero new model forwards**.
This explains the ranking/accuracy mismatch in the value-path decision; it
does not replace that intervention with a causal claim about BatchNorm.

## Scientific conclusion

At 50k, suppression sharply compresses the support-driven difference between
class references while preserving a positive-class preference. The resulting
scores can order the two classes better while almost every query remains on
the positive side of the decision boundary. This is why an AUC improvement is
not a classifier repair. The value-path intervention supplies the causal
localization; the present accounting shows how its observed scores arise.

The support contribution to the raw class-difference vector has mean norm
**8.533 -> 1.016**, intact to suppressed, and stays closely aligned with the
total difference (mean cosine **.99983 -> .99289**). The normalized total
contrast shrinks .28346 -> .02696. Within-episode residual score SD falls
1.064 -> .056, yet the global mean margin stays positive, **.111 -> .137**.
Class-preference analysis already established 99.85% positive predictions
after suppression, despite 25% positive prevalence.

## An actual pair makes the distinction visible

The entire **first saved original 50k episode** was selected by index, not by
outcome. Its 30 centers, graph features, graph-row identities and CSV labels
were verified against the original graph and user table. The pair below was
then chosen within it to illustrate a corrected ordering but a wrong decision;
it is not a representative prevalence estimate.

| Query (biography paraphrased) | Dataset annotation | Intact margin | Values only | Suppressed margin |
|---|---|---:|---:|---:|
| Describes conservative politics, family and MAGA/KAG cues | conservative | -.27311 | +.18036 | +.22518 |
| Describes being a Texas liberal in a red environment | non-conservative | -.06204 | +.15464 | +.19101 |

Margins are the uncalibrated global conservative-minus-non-conservative logit
contrast. Query vectors and inputs stay fixed. Intact gets the second query
right but orders the pair incorrectly; both interventions correct their order
yet predict conservative for both. The first episode's AUC is .4722 intact,
.5556 value-only, .5370 suppressed; accuracy is .4167, .2500, .2500. The full
256-episode result, not this pair, supports the aggregate claim.

Private identity linkage: first episode original stream, query indices 3 and
21, graph rows 35357 and 60598. Input SHA256:
`052c8fddf27405874916f4d0747278552f7f07a12bb77192026e92c4a0115927`.
The full episode also contains an apparent text/annotation inconsistency: a
Democratic-committee self-description with a conservative dataset annotation.
No labels were changed or treated as independently verified political identity.
The three conservative supports have explicit political self-descriptions;
the other three include film and nonspecific profiles. This is context for an
illustration, not evidence that this support composition causes the failure.

The earlier historical case `covid_political_b00_q30` remains a useful
motivating example of helpful **graph-selected query features** and harmful
support processing. Query-edge removal also repairs that case, so it must
not be presented as helpful query message passing. Its different checkpoint
prevents retrofitting the new K/V explanation to that individual example.

## What the decomposition says

For a frozen single M layer, each unnormalized class reference is an exact
sum of five terms: support values, label self-message, label input residual,
degree-multiplied output bias, and the saved affine BatchNorm offset.

Write `L_c = sum_j L_cj` and retain the **observed** norms `r_c = ||L_c||`.
Then `d_j = L_1j / r_1 - L_0j / r_0` and `d = sum_j d_j`.
The corresponding query margin contributions also add. They are not the
counterfactual outputs of deleting a term: that would change normalization
denominators and potentially upstream computations.

50k mean global-class margin contributions, both streams:

| Observed component | Intact | Values only | Suppressed |
|---|---:|---:|---:|
| Support values | -.32815 | +.04715 | +.06857 |
| Shared BN offset | +.41849 | +.04704 | +.04673 |
| Label self-message | +.02040 | +.01534 | +.02250 |
| Direct label residual | -.00078 | -.00063 | -.00063 |
| Per-edge output bias | +.00108 | +.00012 | +.00012 |
| **Total** | **+.11104** | **+.10903** | **+.13730** |

A common BN offset cancels before normalization, but its normalized contrast
is `b * (1/r_1 - 1/r_0)`. Thus a common vector can contribute to classification.
Here its mean contribution actually **decreases** after suppression. The
observed shift from cancellation to reinforcing positive contributions is
an accounting explanation, not proof that BN is the cause of transfer failure.
No component-removal performance or additive fraction of AUC is inferred.

## A real interface caveat, but not residual dominance

NM training has `ignore_label_embeddings=True`: the initial 768-to-256 label
projection is evaluated and then overwritten, so it has no loss-gradient path.
The replacement embedding table is frozen too. Classification sets the flag
False and activates the saved, untrained projection of class-keyed Gaussian
vectors. The projected labels can affect attention, self-messages, the label
residual, and label-to-query messages; they are not semantic label embeddings.

The earlier complete 540-cell label-interface test used the nine historical
specialists with intact supports. It does not cover the current 2.5k controlled
checkpoint or the nominated 50k model, nor their suppressed/K/V conditions.
It therefore cannot establish interface-independence of the new mechanism.

However, the specific hypothesis that suppression exposes a **dominant direct
untrained label residual** is not supported. At 50k its raw-difference norm is
.02750 in both conditions. Its ratio to the support contribution after
suppression has median .0276, 95th percentile .0378, maximum .0448 across all
256 episodes; it never dominates. For value-only the maximum is .0696.
The same direct-residual dominance hypothesis fails in every discovery episode
(suppressed maximum ratio .2205). This does not rule out indirect label effects
through attention or show how restoring the training interface would behave.

**Advisor decision:** stop this control block. Do not claim BN causation or
interface independence; those stronger claims would require new evidence.
The higher-value next step remains one representative public benchmark, not
another inventory of internal toggles. A read-only feasibility audit of the
original Wiki-to-FB15K-237 route is underway because no official MAG checkpoint
was located; no benchmark download or training has been launched.

## Evidence and reproducibility

- `data/classrefterms_20260907`: 75 summary cells, 9,600 episode-component rows,
  1,440 saved batch-condition suffix reconstructions. All native float32
  reconstructions are bit-exact. Maximum float64 raw-label accounting error
  is 2.70e-6; final native-margin difference is 4.53e-6. These are explicitly
  different checks, not a bit-exact claim for the float64 decomposition.
- Seven pure synthetic tests cover additive accounting, native suffix replay,
  common offsets, observed normalization denominators, unsupported captures,
  immutability and interleaved multi-episode label pairing.
- Code-only commit `4e33b40d`, branch `codex/role-topology-interactions`, local
  worktree `/Users/philipp/projects/gfm/prodigy/.worktrees/role-topology`.
- Tucker worktree `/dataMeR1/phil/gfm/prodigy-classrefterms`, detached at that
  commit; completed output `log/classrefterms_20260907`. Two CPU threads,
  no model forwards, no new training, no GPU use. Existing K/V worktree stays
  pinned to `5ffbfa2b`.
- Run `analyze_class_reference_terms` with explicit `--discovery-root`,
  `--long-root`, `--output` under the prodigy environment. These inputs are
  private Tucker-only captures. `plot_class_reference_terms.py --output ...`
  uses local aggregate/numeric results with Python 3.11 and `MPLBACKEND=Agg`.
  Figure: `figures/class_reference_score_accounting.png`.

New results, findings and figure remain uncommitted/private. The main manuscript
and latest one-page PDF retain their previously verified hashes. No automation
was created or resumed. This is progress toward the publication goal, not a
claim that representative-model relevance or the full paper is established.
