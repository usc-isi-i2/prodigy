# Source-held-out NM intervention campaign

Seed 0 exploratory results. All checkpoints selected using active training-source validation only; TwiBot-20 excluded from selection.

Overall arm status requires all 8 rungs × 9 targets and paired baselines. Endpoint columns describe only the eight-source endpoint and may be available before the full campaign is complete. Effects use a ±0.001 practical threshold, not statistical significance. Baseline is the reference; its zero delta is not an intervention finding.

| arm             | status     |   cells |   expected_cells | endpoint_included_status   | endpoint_included_delta   | endpoint_unseen_status   | endpoint_unseen_delta   | endpoint_all_targets_delta   | all_rung_included_delta   |
|:----------------|:-----------|--------:|-----------------:|:---------------------------|:--------------------------|:-------------------------|:------------------------|:-----------------------------|:--------------------------|
| baseline        | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| exposure        | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| schedule        | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| composition     | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| centers         | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| eligibility     | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| positives       | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| negatives       | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| context         | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| optimization    | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| alignment       | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| sharing         | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| capacity        | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| objective       | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| region_adaptive | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| coverage        | incomplete |       0 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |
| budget          | incomplete |       1 |               72 | incomplete                 |                           | incomplete               |                         |                              |                           |

The all-target curve uses the same nine graphs at every rung and requires a complete target panel. Included-source and not-yet-included-source averages change graph membership across rungs; use the fixed-panel and unseen-graph curves to avoid that composition confound. All panels remain separate. No CLS or LP runs are included. Plateau/cap metadata and exact configurations are retained in data/model_records.json.
