# Official AA-DC reproduction on ogbl-collab

The pinned public AA-DC implementation reproduces its leaderboard result exactly to
the reported precision: validation Hits@50 is `67.3557%` and test Hits@50 is
`68.0243%`. The deterministic CPU run took 67 seconds.

| Method | Test Hits@50 | Difference from AA-DC |
| --- | ---: | ---: |
| OGB-style GraphSAGE (three seeds) | 48.62 +/- 0.99 | -19.40 |
| AA-DC reproduction | **68.02** | - |
| Published AA-DC | 68.02 | 0.00 |
| Published HyperFusion | 71.29 +/- 0.18 | +3.27 |

AA-DC has no learned parameters. It uses time-decayed weighted Adamic-Adar, a
validation-calibrated structural gate, and validation-calibrated length-3-path rescue.
Validation scoring uses the training graph; test scoring uses train plus validation
edges. It is therefore comparable to leaderboard methods using the optional
validation-as-input test setting, not to the stricter no-validation-edge SAGE arm.

## Evidence contract

- Producing revision: `4c1066a97ce7c81fae9a54b18d17515d244ee577`.
- Pinned upstream revision: `b499c2046cfe76448545dfe08fad9effb58dd076`.
- Upstream repository: <https://github.com/anmo-fish/aa-dc>.
- Raw log SHA-256: `30bb1868698fb86009884619c92b23ce7253d59ff74a3154727c188f2c693543`.
- Runtime artifacts: `/dataMeR1/phil/gfm/ogbl_collab_aadc/official_b499c204/` on Tucker.
- W&B: <https://wandb.ai/eibl-usc/ogbl-collab-aadc/runs/po2d02y3>.

