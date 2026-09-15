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

