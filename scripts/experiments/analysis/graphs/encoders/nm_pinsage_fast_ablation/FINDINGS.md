# PinSAGE fixed-test result

The seed-0 screen provides a strong negative signal: PinSAGE underperforms its matched
GraphSAGE final-core checkpoint in all 16 held-out NM cells. Evaluation uses the exact
published 512-episode static-test streams; both raw-plan and observed-stream hashes match
cell by cell.

| Training source set | Mean PinSAGE accuracy | Mean GraphSAGE accuracy | Mean delta |
|---|---:|---:|---:|
| COVID specialist | 17.83% | 19.51% | -1.69 pp |
| Election specialist | 14.32% | 15.38% | -1.06 pp |
| Leave-TwiBot-out mixture | 21.01% | 23.67% | -2.67 pp |
| TwiBot specialist | 16.61% | 19.58% | -2.97 pp |

The closest cell is COVID specialist to COVID Political (-0.28 pp); the largest loss is
the leave-TwiBot-out mixture to Election 2020 (-4.01 pp). Even the two in-domain cells
decline: Election specialist to Election 2020 is -0.37 pp and TwiBot specialist to
TwiBot-20 is -1.30 pp.

These are one-seed screening results, so they do not estimate seed variance. Because the
direction agrees in every paired cell and the mixture loses on every target, the result
does not justify completing the remaining PinSAGE training sweep under this sampler.
