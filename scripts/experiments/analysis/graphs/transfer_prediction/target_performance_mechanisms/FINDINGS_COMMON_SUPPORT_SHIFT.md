# A shared support shift is not sufficient for the 50k ranking gain

7 September 2026. Bounded saved-state decomposition, zero model forwards.
Hypothesis stated before computing new outcomes: the support-message change
may improve references chiefly by shifting every support along a shared
direction rather than changing distinctions among supports.

## Test

Use all 128 original and 128 fresh episodes from the nominated HK50k political
K/V experiment. For each episode let delta z(s) be removed-minus-intact U1
support embeddings. Split it into the unweighted mean over all six supports,
mu, and residuals delta z(s)-mu. This split uses no query labels. No component
or hyperparameter was selected by outcome.

Apply the verified linear transport `sum_s D O A(cs) V delta z(s)` separately
to common and residual components, retaining intact attention, final queries,
and every other reference term. Add each component or both to intact final
references and compute normalized cosine scores. These are saved-state
counterfactual readouts, not newly executed upstream interventions; their
joint reconstructs the existing value-only intervention, not joint K/V.

| Stream | Intact AUC | Common-only | Residual-only | Both = values-only |
|---|---:|---:|---:|---:|
| Original | .566406 | .560764 | .626013 | .634983 |
| Fresh | .560909 | .556568 | .638238 | .650174 |

AUC is the equal-weight mean within binary episodes. Original/fresh accuracy:
intact .486979/.496745; common-only .438802/.446615; residual-only
.370443/.367839; both .259115/.258464. The ranking/decision discrepancy remains.

Across all 256 episodes, reconstructed joint reference changes have maximum
coordinate error 2.73e-6 against saved value-only references. Aggregate AUC
and accuracy match the saved value-only readout in both streams. Float64
reconstruction is not claimed bit-exact with original float32 arithmetic.

## Interpretation

The common shift alone does not reproduce the improvement: it slightly hurts
ranking in both streams. Support-specific changes improve within-episode AUC
by 5.96 and 7.73 points; the full value update improves by 6.86 and 8.93.
Do not convert these ratios into additive mediation fractions. Final cosine
normalization is nonlinear, and the joint need not equal singleton effects.

Reject the sufficient-common-shift account for this configuration. Do not
claim to reject every shared-direction hypothesis: a shared direction with
support-dependent coefficients is not the same as an identical displacement
of every support. Residuals include both between-class and within-class
changes, and this test does not separate them. Nor does it establish why the
learned map makes those changes beneficial, or predict a different source.

Together with the prototype reversal, the sharper candidate concerns how
learned reference construction uses *differences among supports*, rather than
uniformly improved support geometry or a shared offset alone. This remains
a candidate mechanism, not the completed publication claim.

## Follow-up: class means and within-class deviations must change together

Using the same saved tensors, split each displacement exactly into common
mean, class-mean-minus-common, and displacement-minus-class-mean. The latter
two are respectively between-class and within-class components. Recover
support labels from positive support-to-label edges, verifying three supports
per class. No query label defines any component; query labels only score
the counterfactuals. Apply the same fixed-attention transport as above.

| Stream | Intact | Between only | Within only | All class means | Between + within | All components |
|---|---:|---:|---:|---:|---:|---:|
| Original AUC | .566406 | .564019 | .566334 | .570530 | .626013 | .634983 |
| Fresh AUC | .560909 | .563151 | .565683 | .568721 | .638238 | .650174 |
| Original accuracy | .486979 | .490560 | .484049 | .418620 | .370443 | .259115 |
| Fresh accuracy | .496745 | .486003 | .478841 | .429036 | .367839 | .258464 |

All class means means common plus between; between plus within is the
previous residual-only condition. Full reconstruction retains maximum
coordinate error 2.73e-6. Each row uses all 128 episodes in that stream.

Neither moving class means alone nor changing within-class deviations alone
reproduces the ranking benefit. Between plus within improves by 5.96/7.73
AUC points despite near-zero singleton changes. The corresponding AUC
interaction contrast is +6.21/+7.03 points. This is metric nonadditivity,
not evidence of a nonlinear unnormalized reference map: that map is affine
under fixed attention. Cosine normalization and ranking can make jointly
transported changes consequential when singleton changes are not.

Therefore do not rename the effect either class-mean repair or denoising
of individual supports. The evidence supports a joint reference-geometry
change, but does not supply a simple independently predictive component.
Stop this component-splitting branch here. A further partition of these same
256 episodes would not test the acceptance-critical cross-case explanation.

Evidence: Tucker `prodigy-classrefkv/log/classrefkv_long_20260907/`,
`private_activations/{original,fresh}/batch_*.pt` and corresponding
`private_predictions` files; nominated HK50k weights. Input-set hashes are
recorded in `FINDINGS_SUPPORT_PROTOTYPE_CONTRAST.md`. Analysis worktree
`.worktrees/role-topology`, branch `codex/role-topology-interactions`,
HEAD `30df8cb5`. Private and uncommitted; PDF and manuscript unchanged.
