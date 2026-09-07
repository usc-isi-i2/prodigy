# Offset subtraction locates an ablation effect, not a transfer repair

7 September 2026. Completed predeclared paired diagnostic. Private results.

## Design and identity

Code 48907a3b, branch codex/role-topology-interactions, local worktree
`.worktrees/role-topology`; Tucker isolated worktree
`/dataMeR1/phil/gfm/prodigy-offset`, detached at that commit.
Artifacts: `log/offset_paired_20260907/{protocol,metrics,geometry,receipts,DONE}.json`
and private `original.pt`, `fresh.pt`. All eight cells complete, 128 episodes
and 3072 queries per stream. No new checkpoint or data selection.

Intervention subtracts exactly the current message MLP's output on zero input
from support-node pre-BN activations. Same subtraction under intact and zeroed
messages; no subtraction of self bias or tuning. Initial label interface is
unchanged. Both native endpoints reproduce saved logits bit-exactly; final
query vectors remain bit-exact across all four conditions. Input hashes and
checkpoint weights were verified unchanged. Three local hook tests passed.

## Performance

| Stream | Condition | Within-episode AUC | Pooled AUC | Accuracy |
|---|---|---:|---:|---:|
| Original | Intact | .566406 | .566500 | .486979 |
| Original | Intact minus default | .534361 | .534139 | .480794 |
| Original | Suppressed | .654044 | .626033 | .251628 |
| Original | Suppressed minus default | .608941 | .570599 | .393229 |
| Fresh | Intact | .560909 | .563361 | .496745 |
| Fresh | Intact minus default | .519387 | .527755 | .483398 |
| Fresh | Suppressed | .664424 | .630650 | .251302 |
| Fresh | Suppressed minus default | .625868 | .573066 | .387044 |

Intact within-AUC falls 3.20/4.15 points; accuracy also falls. Suppressed
accuracy rises 14.16/13.57 points but remains below native intact accuracy;
its within-AUC falls 4.51/3.86 points. This is not joint ranking/decision repair.

## Internal quantities: restoring one stage does not restore the encoder

| Stream / condition | Pre-BN dispersion | Post-ReLU dispersion | U1 dispersion | Final reference separation |
|---|---:|---:|---:|---:|
| Original intact | .569587 | .277288 | .145405 | .282174 |
| Original intact minus default | .318924 | .096234 | .054476 | .179317 |
| Original suppressed | .000775 | .000737 | .000272 | .027320 |
| Original suppressed minus default | .184817 | .000487 | .000322 | .030105 |
| Fresh intact | .573743 | .280294 | .145407 | .284739 |
| Fresh intact minus default | .323538 | .098069 | .054979 | .182415 |
| Fresh suppressed | .000806 | .000756 | .000285 | .026601 |
| Fresh suppressed minus default | .189963 | .000493 | .000342 | .028558 |

Dispersion is the equal-episode mean squared deviation of unit support-center
vectors (node stages) or unit support embeddings (U1). Final reference
separation is the distance between the two normalized final label vectors.

The predeclared mechanistic prediction passes in both streams: subtracting
MLP(0) restores suppressed pre-BN angular diversity. But post-ReLU and U1
remain nearly collapsed. The present summaries locate the new contraction
between pre-BN and post-ReLU, not separately at BN versus ReLU. Do not claim
the subtraction repairs the full encoder, or that normalization is irrelevant.

## Scientific decision

The learned default offset explains a real upstream effect of suppression,
but is neither a sufficient explanation of final collapsed references nor
evidence of a harmful offset in ordinary inference. Intact results contradict
the proposed simple repair. The zero-message input is evaluated through an
otherwise unchanged learned pipeline; restoring diversity before one operation
does not restore a compatible operating regime downstream.

Close the offset-removal branch. Do not tune subtraction strengths or bypass
additional layers to pursue a favorable metric. Integrate this result into
the figure-led distinction between early robust benefit and late fragile AUC
gain. The paper still needs a compelling explanation of the early benefit;
this late ablation diagnosis must not be substituted for that contribution.
