# Repeat/novel regime H1 prerequisite: failed

The frozen no-training prerequisite completed on 2026-09-14 and stopped H1 before
training. The deterministic rule used AA-DC for previously observed pairs and the
seed-matched compact joint scorer for novel pairs. Per-expert empirical-CDF
calibration was fit on pooled 2017 candidates and applied unchanged to 2018. The
negative cutoff was recomputed after mixing. No 2019 scores were accessed.

| Panel | AA-DC | Joint mean | Regime mixture mean | Gain vs joint | Gain vs better expert |
|---|---:|---:|---:|---:|---:|
| 2017 calibration/selection | 69.1940% | 78.1208% | 78.0655% | -0.0554 pp | -0.0554 pp |
| 2018 forward assessment | 66.8398% | 63.6820% | 64.8037% | +1.1218 pp | -2.0360 pp |

The rule preserved repeat positives: its 2018 repeat recall exceeded AA-DC by
0.70--0.71 points in every seed. That does not make it competitive. In 2018 the
joint expert's novel recall was only 32.68--35.09%, below AA-DC's 38.43%, so routing
all novel pairs to the joint model discarded useful AA rankings. On 2017 the joint
model was much stronger on novel pairs, but the regime mixture was slightly worse
than the joint model in two of three seeds. The sign and relative expert ordering
therefore changed across years.

Only the repeat-retention condition passed. The all-seed positive-gain condition,
the >=0.5-point mean gain over joint in both years, and the >=0.5-point forward gain
over the better single expert all failed. The independently recomputed decision is
to stop before H1 training.

This rejects the fixed hard rule, not every regime-specific model. It also sharpens
the obstacle: `repeat` is an observable stable stratum, but `novel` does not identify
which expert to trust across years. A future H1 would require an additional
training-observable subdivision or uncertainty signal with a multi-year oracle
showing stable complementarity before any learned gate is justified. Flexible gate
training against these same two panels would risk repeating the already observed
year-specific tree failure.

Evidence: `data/results.json`, frozen `data/protocol.json`, and independently
recomputed `data/independent_audit.json`. Producing revision `b73af00b`; Tucker
worktree `/dataMeR1/phil/gfm/prodigy-collab-regime-h1`; runtime root
`/dataMeR1/phil/gfm/ogbl_collab_compact_joint/regime_h1_prerequisite_v1`.

## Bounded novel-pair selector follow-up: failed

The predeclared follow-up selected among 14 simple observable rules on 2017, then
applied the selected per-seed rule unchanged to 2018. Candidates used AA presence,
L3 evidence, recent endpoint activity, raw-feature cosine, minimum degree, and
calibrated expert disagreement. No learned gate or 2019 score was used.

Every seed selected `joint_all_novel` on 2017. Its selection gain over AA was
9.20, 7.87, and 9.54 points. Forward on 2018, the same rules lost 2.73, 1.45, and
1.93 points to AA. They lost 1,840, 1,068, and 1,355 novel hits respectively even
while repeat recall improved by about 0.70 point. Mean forward Hits@50 was 64.8037%
versus 66.8398% for AA.

The selector fails two of three advancement conditions: no seed beats AA by the
required 0.5 point and every seed loses novel net hits. Only repeat retention passes.
The independent audit reproduces the selected rules, arithmetic, and stop decision.

This is stronger evidence against an immediate learned gate: the broad novel regime
looks overwhelmingly favorable to joint in 2017 and reverses in 2018. Thresholding
the same feature set on these two years would expose a flexible selector to exactly
the temporal-boundary failure already seen in the shallow tree. The next supported
direction is improving the forward novel expert or identifying genuinely historical
replication beyond these two panels, not adding gate flexibility.

Follow-up evidence: `data/selector_results.json`, `data/selector_protocol.json`, and
`data/selector_independent_audit.json`. Producing revision `c8d601df`; runtime root
`/dataMeR1/phil/gfm/ogbl_collab_compact_joint/regime_h1_selector_v1`.
