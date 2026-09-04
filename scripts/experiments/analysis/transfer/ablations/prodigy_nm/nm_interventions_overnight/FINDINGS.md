# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Frozen training-validation-selected recipe: objective. Only one intervention met the gate. The second-stage runs repeat that seed-zero recipe and provide no test of interactions between interventions. See [frozen selection](data/combined_selection.json) and [input hashes/timestamp](data/combination_freeze_manifest.json).

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status       |   cells |   expected_cells | endpoint_included_status   |   endpoint_included_delta | endpoint_unseen_status   |   endpoint_unseen_delta |   endpoint_all_targets_delta |   all_rung_included_delta |
|:----------------|:-------------|--------:|-----------------:|:---------------------------|--------------------------:|:-------------------------|------------------------:|-----------------------------:|--------------------------:|
| baseline        | inconclusive |      72 |               72 | inconclusive               |               0           | inconclusive             |             0           |                  0           |               0           |
| exposure        | degraded     |      72 |               72 | degraded                   |              -0.00961119  | improved                 |             0.00749238  |                 -0.0077108   |              -0.00898427  |
| schedule        | degraded     |      72 |               72 | degraded                   |              -0.00800886  | degraded                 |            -0.017644    |                 -0.00907943  |              -0.0067718   |
| composition     | improved     |      72 |               72 | improved                   |               0.00184777  | degraded                 |            -0.0022187   |                  0.00139594  |               0.00297616  |
| centers         | degraded     |      72 |               72 | degraded                   |              -0.00365723  | degraded                 |            -0.00613341  |                 -0.00393236  |              -0.00667168  |
| eligibility     | degraded     |      72 |               72 | degraded                   |              -0.0118435   | degraded                 |            -0.00704215  |                 -0.01131     |              -0.00859615  |
| positives       | inconclusive |      72 |               72 | inconclusive               |              -0.000930751 | improved                 |             0.00408403  |                 -0.000373554 |              -0.00270617  |
| negatives       | degraded     |      72 |               72 | degraded                   |              -0.00403161  | improved                 |             0.00323414  |                 -0.00322431  |              -0.00357642  |
| context         | degraded     |      72 |               72 | degraded                   |              -0.0168256   | degraded                 |            -0.0130068   |                 -0.0164012   |              -0.0184483   |
| optimization    | degraded     |      72 |               72 | degraded                   |              -0.00161098  | degraded                 |            -0.00302144  |                 -0.0017677   |              -0.000488837 |
| alignment       | improved     |      72 |               72 | improved                   |               0.00223498  | inconclusive             |             0.000352728 |                  0.00202584  |               0.00074643  |
| sharing         | degraded     |      72 |               72 | degraded                   |              -0.0156604   | degraded                 |            -0.0585976   |                 -0.0204312   |              -0.0157639   |
| capacity        | inconclusive |      72 |               72 | inconclusive               |               0.000361966 | degraded                 |            -0.00307993  |                 -2.04671e-05 |              -2.46401e-05 |
| objective       | improved     |      72 |               72 | improved                   |               0.00252849  | inconclusive             |            -0.000192013 |                  0.00222621  |               0.00159004  |
| region_adaptive | degraded     |      72 |               72 | degraded                   |              -0.00623185  | degraded                 |            -0.00522614  |                 -0.0061201   |              -0.0068783   |
| coverage        | degraded     |      72 |               72 | degraded                   |              -0.00220001  | degraded                 |            -0.00124469  |                 -0.00209386  |              -0.00250602  |
| budget          | inconclusive |      72 |               72 | inconclusive               |              -0.000964079 | degraded                 |            -0.00262548  |                 -0.00114868  |              -0.00180662  |

Training cost and stopping evidence for completed models:

| arm             |   trained_models |   model_parameters |   mean_episodes |   plateau_stops |   cap_stops |   cap_with_last_check_gain |   mean_seconds_per_1000_episodes |   peak_tensor_mib |
|:----------------|-----------------:|-------------------:|----------------:|----------------:|------------:|---------------------------:|---------------------------------:|------------------:|
| baseline        |                8 |            1640514 |           10000 |               1 |           7 |                          5 |                            52.74 |            472.8  |
| exposure        |                8 |            1640514 |            9250 |               4 |           4 |                          2 |                            54.26 |            466.01 |
| schedule        |                8 |            1640514 |            9500 |               4 |           4 |                          4 |                            55.78 |            472.24 |
| composition     |                8 |            1640514 |           10000 |               0 |           8 |                          0 |                            54.54 |            459.23 |
| centers         |                8 |            1640514 |            9000 |               5 |           3 |                          2 |                            55.03 |            482.09 |
| eligibility     |                8 |            1640514 |            9000 |               4 |           4 |                          1 |                            52.22 |            468.66 |
| positives       |                8 |            1640514 |            9750 |               2 |           6 |                          4 |                            53.59 |            467.93 |
| negatives       |                8 |            1640514 |            9250 |               5 |           3 |                          2 |                            53.69 |            483.62 |
| context         |                8 |            1640514 |            9500 |               4 |           4 |                          3 |                            58.26 |            489.77 |
| optimization    |                8 |            1640514 |           10000 |               0 |           8 |                          5 |                            55.39 |            476.83 |
| alignment       |                8 |            1640514 |            9500 |               2 |           6 |                          4 |                            59.37 |            472.8  |
| sharing         |                8 |            1654338 |            8000 |               7 |           1 |                          0 |                            74.43 |            530.01 |
| capacity        |                8 |            4860034 |            9500 |               2 |           6 |                          3 |                            63.13 |            768.18 |
| objective       |                8 |            1640514 |           10000 |               1 |           7 |                          4 |                            55.03 |            531.88 |
| region_adaptive |                8 |            1640514 |            9000 |               6 |           2 |                          1 |                            56.04 |            482.05 |
| coverage        |                8 |            1640514 |            9750 |               1 |           7 |                          4 |                            54.64 |            474.87 |
| budget          |                8 |            1640514 |            5625 |               1 |           7 |                          3 |                            55.49 |            472.8  |

Parameter counts include the registered frozen label table; resources.csv separately records optimizer parameter slots and the auxiliary head. Timing comes from concurrent runs, excludes initial validation-cache construction from the loop timer, and is not an isolated speed benchmark. Peak tensor memory excludes CUDA context overhead. A cap stop is not evidence of convergence; cap_with_last_check_gain counts capped runs whose final validation increment still exceeded 0.001. Effect verdicts apply to this bounded training protocol.


Source exposure audit passed for all 136 collected models: 12664 cumulative curve records plus every terminal record. Inactive sources, including TwiBot-20, have zero exposure; source totals match consumed episodes. Blocked-arm records match the exact 64-episode source cycle. See [per-model checks](data/exposure_audit.json) and [terminal exposures](data/source_exposure.csv).

[Unseen transfer by intervention](figures/unseen_by_arm.png) shows each method against baseline on shared axes. The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
