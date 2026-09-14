# AA-DC reproduction on ogbl-collab

This setup runs the public AA-DC implementation at pinned upstream revision
`b499c2046cfe76448545dfe08fad9effb58dd076`. It uses the authors' full command:
progressive gate with validation calibration, L3 rescue with validation-selected
anchor, time decay 0.95, and train-plus-validation graph input for the test split.

The method is deterministic and has no learned parameters, so there is one result
cell rather than a seed sweep. The upstream stdout is the authoritative raw record;
`track_result.py` extracts the official OGB Hits@K values into `results.json` and
mirrors them to W&B.

Runtime root on Tucker: `/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204`.
