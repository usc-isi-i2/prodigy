# Social-checkpoint to FB15K-237 adapter

## Evidence contract

Four frozen seed-0, step-2,500 social specialists were evaluated on native
FB15K-237 20-way, 3-shot relation-classification episodes with 500 fixed test
episodes. The adapter deliberately omits the native KG endpoint and KG-specific
modules. Results were produced at `26afd599be` in Tucker worktree
`prodigy-social-fb15k`. The eight JSON files under `data/` preserve the four
text-only runs and one canonical endpoint-flag run per source. Smoke and duplicate
endpoint launches are excluded.

## Results

| source specialist | text-only accuracy | text-only AUC | endpoint-flag accuracy | accuracy delta |
|---|---:|---:|---:|---:|
| COVID-19 | 0.51260 | 0.92303 | 0.51343 | +0.00083 |
| Ukraine/Russia | 0.41338 | 0.88145 | 0.41305 | -0.00033 |
| TwiBot-20 | 0.41640 | 0.88766 | 0.37643 | -0.03998 |
| Midterm | 0.36053 | 0.84091 | 0.35015 | -0.01038 |

All four specialists transfer above the 5% chance accuracy of a 20-way episode,
with COVID strongest. Supplying endpoint flags is not consistently beneficial and
materially lowers accuracy for TwiBot-20 and Midterm.

These are valid results for the explicit social-to-KG adapter only. They must not
be presented as native PRODIGY paper-architecture comparisons or as a causal
source-only contrast.
