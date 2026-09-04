# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status     |   cells |   expected_cells | endpoint_included_status   |   endpoint_included_delta | endpoint_unseen_status   | endpoint_unseen_delta   | endpoint_all_targets_delta   | all_rung_included_delta   |
|:----------------|:-----------|--------:|-----------------:|:---------------------------|--------------------------:|:-------------------------|:------------------------|:-----------------------------|:--------------------------|
| baseline        | incomplete |      20 |               72 | inconclusive               |               0           | incomplete               |                         |                              |                           |
| exposure        | incomplete |      20 |               72 | degraded                   |              -0.00961119  | incomplete               |                         |                              |                           |
| schedule        | incomplete |      19 |               72 | degraded                   |              -0.00800886  | incomplete               |                         |                              |                           |
| composition     | incomplete |      21 |               72 | improved                   |               0.00184777  | incomplete               |                         |                              |                           |
| centers         | incomplete |      22 |               72 | degraded                   |              -0.00365723  | incomplete               |                         |                              |                           |
| eligibility     | incomplete |      20 |               72 | degraded                   |              -0.0118435   | incomplete               |                         |                              |                           |
| positives       | incomplete |      20 |               72 | inconclusive               |              -0.000930751 | incomplete               |                         |                              |                           |
| negatives       | incomplete |      20 |               72 | degraded                   |              -0.00403161  | incomplete               |                         |                              |                           |
| context         | incomplete |      19 |               72 | degraded                   |              -0.0168256   | incomplete               |                         |                              |                           |
| optimization    | incomplete |      17 |               72 | degraded                   |              -0.00161098  | incomplete               |                         |                              |                           |
| alignment       | incomplete |      16 |               72 | improved                   |               0.00223498  | incomplete               |                         |                              |                           |
| sharing         | incomplete |      16 |               72 | degraded                   |              -0.0156604   | incomplete               |                         |                              |                           |
| capacity        | incomplete |      16 |               72 | inconclusive               |               0.000361966 | incomplete               |                         |                              |                           |
| objective       | incomplete |      16 |               72 | improved                   |               0.00252849  | incomplete               |                         |                              |                           |
| region_adaptive | incomplete |      20 |               72 | degraded                   |              -0.00623185  | incomplete               |                         |                              |                           |
| coverage        | incomplete |      17 |               72 | degraded                   |              -0.00220001  | incomplete               |                         |                              |                           |
| budget          | incomplete |      20 |               72 | inconclusive               |              -0.000964079 | incomplete               |                         |                              |                           |

Training cost and stopping evidence for completed models:

| arm             |   trained_models |   model_parameters |   mean_episodes |   plateau_stops |   cap_stops |   cap_with_last_check_gain |   mean_seconds_per_1000_episodes |   peak_tensor_mib |
|:----------------|-----------------:|-------------------:|----------------:|----------------:|------------:|---------------------------:|---------------------------------:|------------------:|
| baseline        |                6 |            1640514 |        10000    |               1 |           5 |                          4 |                            52.89 |            469.21 |
| exposure        |                6 |            1640514 |         9333.33 |               3 |           3 |                          2 |                            54.44 |            465.96 |
| schedule        |                6 |            1640514 |         9666.67 |               3 |           3 |                          3 |                            52.65 |            472.24 |
| composition     |                6 |            1640514 |        10000    |               0 |           6 |                          0 |                            55.33 |            459.23 |
| centers         |                6 |            1640514 |         8666.67 |               4 |           2 |                          2 |                            53.35 |            481.58 |
| eligibility     |                6 |            1640514 |         8666.67 |               4 |           2 |                          0 |                            49.92 |            468.66 |
| positives       |                6 |            1640514 |         9666.67 |               2 |           4 |                          4 |                            51.25 |            467.93 |
| negatives       |                6 |            1640514 |         9000    |               5 |           1 |                          1 |                            52.55 |            483.62 |
| context         |                6 |            1640514 |         9666.67 |               3 |           3 |                          2 |                            57.79 |            488.59 |
| optimization    |                5 |            1640514 |        10000    |               0 |           5 |                          3 |                            56.22 |            472.31 |
| alignment       |                5 |            1640514 |         9600    |               1 |           4 |                          3 |                            58.06 |            468.28 |
| sharing         |                5 |            1654338 |         7600    |               4 |           1 |                          0 |                            74.06 |            525.74 |
| capacity        |                5 |            4860034 |         9200    |               2 |           3 |                          2 |                            65.54 |            768.18 |
| objective       |                5 |            1640514 |        10000    |               1 |           4 |                          2 |                            56.56 |            526.99 |
| region_adaptive |                5 |            1640514 |         9200    |               3 |           2 |                          1 |                            56.84 |            482.05 |
| coverage        |                5 |            1640514 |        10000    |               0 |           5 |                          4 |                            55.95 |            468.48 |
| budget          |                5 |            1640514 |         4500    |               0 |           5 |                          3 |                            55.79 |            468.28 |

Parameter counts include the registered frozen label table; resources.csv separately records optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. Effect verdicts apply to this bounded training protocol.


Source exposure audit passed for all 94 collected models: 8678 cumulative curve records plus every terminal record. Inactive sources, including TwiBot-20, have zero exposure; source totals match consumed episodes. Blocked-arm records match the exact 64-episode source cycle. See [per-model checks](data/exposure_audit.json) and [terminal exposures](data/source_exposure.csv).

The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
