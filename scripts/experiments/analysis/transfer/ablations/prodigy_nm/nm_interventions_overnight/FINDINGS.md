# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status     |   cells |   expected_cells | endpoint_included_status   |   endpoint_included_delta | endpoint_unseen_status   | endpoint_unseen_delta   | endpoint_all_targets_delta   | all_rung_included_delta   |
|:----------------|:-----------|--------:|-----------------:|:---------------------------|--------------------------:|:-------------------------|:------------------------|:-----------------------------|:--------------------------|
| baseline        | incomplete |      11 |               72 | inconclusive               |               0           | incomplete               |                         |                              |                           |
| exposure        | incomplete |       9 |               72 | degraded                   |              -0.00961119  | incomplete               |                         |                              |                           |
| schedule        | incomplete |      10 |               72 | degraded                   |              -0.00800886  | incomplete               |                         |                              |                           |
| composition     | incomplete |      10 |               72 | improved                   |               0.00184777  | incomplete               |                         |                              |                           |
| centers         | incomplete |      10 |               72 | degraded                   |              -0.00365723  | incomplete               |                         |                              |                           |
| eligibility     | incomplete |      10 |               72 | degraded                   |              -0.0118435   | incomplete               |                         |                              |                           |
| positives       | incomplete |      10 |               72 | inconclusive               |              -0.000930751 | incomplete               |                         |                              |                           |
| negatives       | incomplete |       9 |               72 | degraded                   |              -0.00403161  | incomplete               |                         |                              |                           |
| context         | incomplete |      10 |               72 | degraded                   |              -0.0168256   | incomplete               |                         |                              |                           |
| optimization    | incomplete |      10 |               72 | degraded                   |              -0.00161098  | incomplete               |                         |                              |                           |
| alignment       | incomplete |      10 |               72 | improved                   |               0.00223498  | incomplete               |                         |                              |                           |
| sharing         | incomplete |      10 |               72 | degraded                   |              -0.0156604   | incomplete               |                         |                              |                           |
| capacity        | incomplete |       9 |               72 | inconclusive               |               0.000361966 | incomplete               |                         |                              |                           |
| objective       | incomplete |       9 |               72 | improved                   |               0.00252849  | incomplete               |                         |                              |                           |
| region_adaptive | incomplete |      10 |               72 | degraded                   |              -0.00623185  | incomplete               |                         |                              |                           |
| coverage        | incomplete |      10 |               72 | degraded                   |              -0.00220001  | incomplete               |                         |                              |                           |
| budget          | incomplete |      10 |               72 | inconclusive               |              -0.000964079 | incomplete               |                         |                              |                           |

Training cost and stopping evidence for completed models:

| arm             |   trained_models |   model_parameters |   mean_episodes |   plateau_stops |   cap_stops |   cap_with_last_check_gain |   mean_seconds_per_1000_episodes |   peak_tensor_mib |
|:----------------|-----------------:|-------------------:|----------------:|----------------:|------------:|---------------------------:|---------------------------------:|------------------:|
| baseline        |                5 |            1640514 |           10000 |               1 |           4 |                          3 |                            51.53 |            468.28 |
| exposure        |                5 |            1640514 |            9200 |               3 |           2 |                          2 |                            54.14 |            465.13 |
| schedule        |                5 |            1640514 |           10000 |               2 |           3 |                          3 |                            51.53 |            468.67 |
| composition     |                4 |            1640514 |           10000 |               0 |           4 |                          0 |                            51.84 |            459.23 |
| centers         |                5 |            1640514 |            8400 |               3 |           2 |                          2 |                            52.92 |            481.58 |
| eligibility     |                5 |            1640514 |            8800 |               3 |           2 |                          0 |                            48.74 |            467.27 |
| positives       |                4 |            1640514 |            9500 |               1 |           3 |                          3 |                            51    |            467.93 |
| negatives       |                4 |            1640514 |            9500 |               3 |           1 |                          1 |                            52.02 |            483.62 |
| context         |                4 |            1640514 |           10000 |               1 |           3 |                          2 |                            55.15 |            488.59 |
| optimization    |                4 |            1640514 |           10000 |               0 |           4 |                          3 |                            57.72 |            472.31 |
| alignment       |                4 |            1640514 |           10000 |               0 |           4 |                          3 |                            59.64 |            468.28 |
| sharing         |                4 |            1654338 |            8000 |               3 |           1 |                          0 |                            72.92 |            525.74 |
| capacity        |                4 |            4860034 |            9500 |               1 |           3 |                          2 |                            64.07 |            768.18 |
| objective       |                4 |            1640514 |           10000 |               0 |           4 |                          2 |                            57.48 |            526.99 |
| region_adaptive |                4 |            1640514 |            9000 |               2 |           2 |                          1 |                            57.4  |            482.05 |
| coverage        |                4 |            1640514 |           10000 |               0 |           4 |                          4 |                            56.83 |            468.48 |
| budget          |                4 |            1640514 |            4375 |               0 |           4 |                          2 |                            56.6  |            468.28 |

Parameter counts include the registered frozen label table; resources.csv separately records optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. Effect verdicts apply to this bounded training protocol.


The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
