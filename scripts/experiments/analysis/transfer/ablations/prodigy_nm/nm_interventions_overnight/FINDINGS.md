# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status     |   cells |   expected_cells | endpoint_included_status   | endpoint_included_delta   | endpoint_unseen_status   | endpoint_unseen_delta   | endpoint_all_targets_delta   | all_rung_included_delta   |
|:----------------|:-----------|--------:|-----------------:|:---------------------------|:--------------------------|:-------------------------|:------------------------|:-----------------------------|:--------------------------|
| baseline        | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| exposure        | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| schedule        | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| composition     | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| centers         | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| eligibility     | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| positives       | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| negatives       | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| context         | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| optimization    | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| alignment       | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| sharing         | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| capacity        | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| objective       | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| region_adaptive | incomplete |       3 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| coverage        | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| budget          | incomplete |       2 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |

Training cost and stopping evidence for completed models:

| arm             |   trained_models |   model_parameters |   mean_episodes |   plateau_stops |   cap_stops |   cap_with_last_check_gain |   mean_seconds_per_1000_episodes |   peak_tensor_mib |
|:----------------|-----------------:|-------------------:|----------------:|----------------:|------------:|---------------------------:|---------------------------------:|------------------:|
| baseline        |                3 |            1640514 |           10000 |               1 |           2 |                          1 |                            52.2  |            468.28 |
| exposure        |                3 |            1640514 |           10000 |               1 |           2 |                          2 |                            52.45 |            465.13 |
| schedule        |                3 |            1640514 |           10000 |               1 |           2 |                          2 |                            51.75 |            468.67 |
| composition     |                2 |            1640514 |           10000 |               0 |           2 |                          0 |                            50.37 |            459.23 |
| centers         |                2 |            1640514 |            9000 |               1 |           1 |                          1 |                            52.01 |            481.58 |
| eligibility     |                2 |            1640514 |            8000 |               1 |           1 |                          0 |                            48.92 |            467.27 |
| positives       |                2 |            1640514 |           10000 |               0 |           2 |                          2 |                            50.56 |            467.93 |
| negatives       |                2 |            1640514 |           10000 |               1 |           1 |                          1 |                            51.88 |            483.62 |
| context         |                2 |            1640514 |           10000 |               0 |           2 |                          1 |                            56.85 |            488.59 |
| optimization    |                2 |            1640514 |           10000 |               0 |           2 |                          2 |                            55.98 |            472.31 |
| alignment       |                2 |            1640514 |           10000 |               0 |           2 |                          2 |                            66.94 |            468.28 |
| sharing         |                2 |            1654338 |            8000 |               1 |           1 |                          0 |                            71.9  |            525.74 |
| capacity        |                2 |            4860034 |           10000 |               0 |           2 |                          1 |                            63.61 |            768.18 |
| objective       |                2 |            1640514 |           10000 |               0 |           2 |                          2 |                            53.29 |            526.99 |
| region_adaptive |                2 |            1640514 |           10000 |               1 |           1 |                          0 |                            52.48 |            482.05 |
| coverage        |                2 |            1640514 |           10000 |               0 |           2 |                          2 |                            62.87 |            468.48 |
| budget          |                2 |            1640514 |            5625 |               0 |           2 |                          1 |                            52.39 |            468.28 |

Parameter counts include the registered frozen label table; resources.csv separately records optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. Effect verdicts apply to this bounded training protocol.


The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
