# Support-only boundary repair: close the tested deployment direction

Completed 2026-09-07, revision1b49e2f9. Tucker run:
`/dataMeR1/phil/gfm/prodigy-boundary-repair/log/boundary_full_20260907`.
DONE confirms128 episodes, six conditions, non-smoke. New offset200009,
3072 query occurrences, HK50k→political target, three supports per class.
No target query labels fitted or hyperparameters revised after the smoke.

| Method | Accuracy | Macro-F1 | Positive F1 | Pooled AUC | Within-episode AUC | NLL |
|---|---:|---:|---:|---:|---:|---:|
| Native | .474609 | .465913 | .397761 | .559512 | .565177 | .812813 |
| Value replacement | .255859 | .209757 | .400629 | .597093 | .622034 | .719427 |
| U1 ridge | .597005 | .556920 | .423650 | .623944 | .655237 | .673977 |
| Native calibrated | .487630 | .470407 | .374901 | .538431 | .531322 | .718718 |
| Value calibrated | .588216 | .522853 | .346253 | .550258 | .561849 | .675924 |
| U1 ridge calibrated | .528971 | .495706 | .366185 | .552325 | .554398 | .682989 |

## Decision fixed before outcomes

Value calibration repairs much of the catastrophic decision collapse and exceeds
calibrated native in accuracy and macro-F1. It does not beat plain U1 on any
listed metric: accuracy -.008789, macro-F1 -.034067, AUC -.073685, NLL +.001947.
Therefore close this particular value-repair deployment direction. Do not tune
the calibrator on this stream or advertise improvement against native while
excluding the stronger U1 baseline.

The support-calibrated U1 variant itself degrades, emphasizing that six
cross-validated support margins do not necessarily yield reliable calibration.
The fixed unconstrained slope can reverse within-episode rankings. These results
do not prove that every possible support-only repair fails, nor that U1 is always
best. They establish the outcome of this nominated operational test.

## Scientific interpretation

Ranking improvement under a mechanistic intervention is not sufficient evidence
of a useful adaptation algorithm. The U1 representation already exposes more
usable discrimination than the tested repair. Across the earlier TRACE panel
and this separate historical checkpoint/protocol, U1 is a meaningful baseline,
not merely a diagnostic helper. A paper claiming improved deployment must clear
that baseline. This observation does not by itself establish novelty or an8+
contribution; it redirects the central question toward which learned inference
operations help versus harm a support-fitted classifier.

All raw private episodes, calibration fits, prediction tensors, metrics, and
hash receipts remain in the run directory. Aggregate metrics are descriptive for
one checkpoint/target; repeated query accounts and episodes are not independent
training seeds or domains.
