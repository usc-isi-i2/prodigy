# topology_feature_ssl — matched-40k results (B0/B1/E1/E2/E2b vs trivial floor)

_All arms at a matched **40k**-episode budget (true `state_dict_40000.ckpt`; E2/E2b
use `epochs:5` per the trainer off-by-one). Anchored against `raw_feat` (bio features,
no encoder) and `raw_degree` (leakage), both 10-shot. An arm only 'improves performance'
if it beats these. **Headline: E1 wins the feature tasks, E2 wins the topological task
(static-LP), and neither clears the joint bar; E2b (drop-BN) trades LP for probe scores.**_

### Regression Spearman (test, mean over datasets) — full panel
```
target      followers_count  friends_count  statuses_count  favourites_count  listed_count  account_age_days
raw_degree            0.162          0.069           0.156             0.093         0.096             0.010
raw_feat              0.203          0.139           0.095             0.073         0.152             0.024
B0_40k                0.033         -0.072           0.021            -0.002        -0.001            -0.062
B1_40k               -0.127         -0.154          -0.107            -0.110        -0.109            -0.133
E1_40k                0.191          0.151           0.095             0.044         0.141             0.118
E2_40k               -0.068         -0.105          -0.095            -0.078        -0.045            -0.069
E2b_40k              -0.041             —           -0.009               —             —              0.047
```

_An arm 'learned structure' only if it beats raw_degree on followers/statuses; 'beats
features' if it tops raw_feat. **Only E1 clears the floor** (followers 0.191 > raw_degree
0.162; account_age 0.118 >> raw_feat 0.024). E2/E2b regression is at/below zero.
(E2b was run on the 3-target eval sweep, not the 6-target panel — hence the blanks.)_

### Classification ROC-AUC (test)
```
dataset   election2020  twibot20
raw_feat         0.848     0.560
B0_40k           0.981     0.605
B1_40k           0.984     0.613
E1_40k           0.953     0.604
E2_40k           0.972     0.589
E2b_40k          0.969     0.599
```
_Flat across arms — classification does not discriminate the encoder/objective here._

### Static-LP ROC-AUC (test) — the direct topological task
```
dataset  covid19_twitter  midterm  twibot20  ukr_rus_twitter    MEAN
B0_40k             0.657    0.658     0.635            0.753    0.676
B1_40k             0.341    0.378     0.339            0.306    0.341
E1_40k             0.628    0.635     0.714            0.650    0.657
E2_40k             0.780    0.708     0.735            0.823    0.762
E2b_40k            0.402    0.361     0.517            0.323    0.401
```
_**E2 is best on static-LP (0.762)** — the count-aware encoder genuinely helps the
topological task. **E2b (drop-BN) collapses it to 0.401** — removing the conv BatchNorm
destroys LP usefulness even as it raises the count probes (below)._

### T2 — 2x2 retained (feature tasks; unstable — near-zero intact denominators)
```
condition  random_feat  rewired_edge   both
B0_40k           3.220        -1.028  0.565
B1_40k           0.213         0.878  0.200
E1_40k           1.002         0.810  0.255
E2_40k           0.777         1.336  1.540
E2b_40k          0.320         1.020  0.410
```
_Uninformative for these arms: reg/pl feature-task metrics sit near zero, so the retained
FRACTION (metric/intact) explodes/inverts. The topology signal lives in static-LP + probes._

### T3 — capability probes (AUC, chance 0.50) — PRIMARY
```
rule    count_threshold  in_degree  out_degree  existence  conjunction
B0_40k            0.478      0.515       0.524      0.515        0.513
B1_40k            0.527      0.509       0.523      0.525        0.526
E1_40k            0.672      0.627       0.515      0.535        0.534
E2_40k            0.589      0.513       0.583      0.623        0.626
E2b_40k           0.660      0.559       0.710      0.550        0.570
```
_E1 leads count/in-degree (from its injected directed-degree inputs, passthrough). **E2b's
drop-BN lifts count 0.59→0.66, in-deg 0.51→0.56, out-deg 0.58→0.71 vs E2** — confirming
BatchNorm was washing out sum-aggregation's count magnitude at the linear-probe level. But
that gain did NOT transfer to the tasks (LP crashed, regression stayed ~0): representable ≠ used._
