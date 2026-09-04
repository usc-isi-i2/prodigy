# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status     |   cells |   expected_cells | endpoint_included_status   |   endpoint_included_delta | endpoint_unseen_status   | endpoint_unseen_delta   | endpoint_all_targets_delta   | all_rung_included_delta   |
|:----------------|:-----------|--------:|-----------------:|:---------------------------|--------------------------:|:-------------------------|:------------------------|:-----------------------------|:--------------------------|
| baseline        | incomplete |      32 |               72 | inconclusive               |               0           | incomplete               |                         |                              |                           |
| exposure        | incomplete |      32 |               72 | degraded                   |              -0.00961119  | incomplete               |                         |                              |                           |
| schedule        | incomplete |      32 |               72 | degraded                   |              -0.00800886  | incomplete               |                         |                              |                           |
| composition     | incomplete |      32 |               72 | improved                   |               0.00184777  | incomplete               |                         |                              |                           |
| centers         | incomplete |      32 |               72 | degraded                   |              -0.00365723  | incomplete               |                         |                              |                           |
| eligibility     | incomplete |      32 |               72 | degraded                   |              -0.0118435   | incomplete               |                         |                              |                           |
| positives       | incomplete |      32 |               72 | inconclusive               |              -0.000930751 | incomplete               |                         |                              |                           |
| negatives       | incomplete |      32 |               72 | degraded                   |              -0.00403161  | incomplete               |                         |                              |                           |
| context         | incomplete |      32 |               72 | degraded                   |              -0.0168256   | incomplete               |                         |                              |                           |
| optimization    | incomplete |      24 |               72 | degraded                   |              -0.00161098  | incomplete               |                         |                              |                           |
| alignment       | incomplete |      24 |               72 | improved                   |               0.00223498  | incomplete               |                         |                              |                           |
| sharing         | incomplete |      24 |               72 | degraded                   |              -0.0156604   | incomplete               |                         |                              |                           |
| capacity        | incomplete |      24 |               72 | inconclusive               |               0.000361966 | incomplete               |                         |                              |                           |
| objective       | incomplete |      24 |               72 | improved                   |               0.00252849  | incomplete               |                         |                              |                           |
| region_adaptive | incomplete |      32 |               72 | degraded                   |              -0.00623185  | incomplete               |                         |                              |                           |
| coverage        | incomplete |      24 |               72 | degraded                   |              -0.00220001  | incomplete               |                         |                              |                           |
| budget          | incomplete |      32 |               72 | inconclusive               |              -0.000964079 | incomplete               |                         |                              |                           |

Training cost and stopping evidence for completed models:

| arm             |   trained_models |   model_parameters |   mean_episodes |   plateau_stops |   cap_stops |   cap_with_last_check_gain |   mean_seconds_per_1000_episodes |   peak_tensor_mib |
|:----------------|-----------------:|-------------------:|----------------:|----------------:|------------:|---------------------------:|---------------------------------:|------------------:|
| baseline        |                8 |            1640514 |        10000    |               1 |           7 |                          5 |                            52.74 |            472.8  |
| exposure        |                7 |            1640514 |         9142.86 |               4 |           3 |                          2 |                            54.46 |            466.01 |
| schedule        |                7 |            1640514 |         9428.57 |               4 |           3 |                          3 |                            55.28 |            472.24 |
| composition     |                7 |            1640514 |        10000    |               0 |           7 |                          0 |                            55.08 |            459.23 |
| centers         |                7 |            1640514 |         8857.14 |               5 |           2 |                          2 |                            55.38 |            481.58 |
| eligibility     |                7 |            1640514 |         8857.14 |               4 |           3 |                          0 |                            52.67 |            468.66 |
| positives       |                7 |            1640514 |         9714.29 |               2 |           5 |                          4 |                            54.07 |            467.93 |
| negatives       |                7 |            1640514 |         9142.86 |               5 |           2 |                          1 |                            52.8  |            483.62 |
| context         |                7 |            1640514 |         9714.29 |               3 |           4 |                          3 |                            57.45 |            488.59 |
| optimization    |                7 |            1640514 |        10000    |               0 |           7 |                          5 |                            55.91 |            476.83 |
| alignment       |                7 |            1640514 |         9428.57 |               2 |           5 |                          4 |                            60.23 |            472.8  |
| sharing         |                7 |            1654338 |         8000    |               6 |           1 |                          0 |                            75.45 |            530.01 |
| capacity        |                7 |            4860034 |         9428.57 |               2 |           5 |                          3 |                            65.03 |            768.18 |
| objective       |                7 |            1640514 |        10000    |               1 |           6 |                          4 |                            55.8  |            531.88 |
| region_adaptive |                7 |            1640514 |         8857.14 |               5 |           2 |                          1 |                            56.8  |            482.05 |
| coverage        |                7 |            1640514 |        10000    |               0 |           7 |                          4 |                            55.41 |            474.87 |
| budget          |                7 |            1640514 |         5178.57 |               1 |           6 |                          3 |                            56.66 |            472.8  |

Parameter counts include the registered frozen label table; resources.csv separately records optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. Effect verdicts apply to this bounded training protocol.


Source exposure audit passed for all 120 collected models: 11121 cumulative curve records plus every terminal record. Inactive sources, including TwiBot-20, have zero exposure; source totals match consumed episodes. Blocked-arm records match the exact 64-episode source cycle. See [per-model checks](data/exposure_audit.json) and [terminal exposures](data/source_exposure.csv).

The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
