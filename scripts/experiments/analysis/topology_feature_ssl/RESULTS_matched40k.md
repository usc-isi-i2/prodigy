# topology_feature_ssl — matched-40k results (B0/B1/E1/E2/E2b vs trivial floor)

_All arms at a matched 40k-episode budget (true state_dict_40000). Anchored against `raw_feat` (bio features, no encoder) and `raw_degree` (leakage), both 10-shot. An arm only 'improves performance' if it beats these._

### Regression Spearman (test, mean over datasets) — full panel
```
target      followers_count  friends_count  statuses_count  favourites_count  listed_count  account_age_days
raw_degree            0.162          0.069           0.156             0.093         0.096             0.010
raw_feat              0.203          0.139           0.095             0.073         0.152             0.024
B0_40k                0.033         -0.072           0.021            -0.002        -0.001            -0.062
B1_40k               -0.127         -0.154          -0.107            -0.110        -0.109            -0.133
E1_40k                0.191          0.151           0.095             0.044         0.141             0.118
E2_40k               -0.068         -0.105          -0.095            -0.078        -0.045            -0.069
E2b_40k              -0.041            NaN          -0.009               NaN           NaN             0.047
E4_40k               -0.181            NaN          -0.139               NaN           NaN            -0.079
E4r_40k              -0.233            NaN          -0.133               NaN           NaN            -0.005
```

_An arm 'learned structure' only if it beats raw_degree on followers/statuses; 'beats features' if it tops raw_feat._

### Classification ROC-AUC (test)
```
dataset   election2020  twibot20
model                           
raw_feat         0.848     0.560
B0_40k           0.981     0.605
B1_40k           0.984     0.613
E1_40k           0.953     0.604
E2_40k           0.972     0.589
E2b_40k          0.969     0.599
E4_40k           0.511     0.378
E4r_40k          0.651     0.636
```

### Static-LP ROC-AUC (test)
```
dataset  covid19_twitter  midterm  twibot20  ukr_rus_twitter
model                                                       
B0_40k             0.657    0.658     0.635            0.753
B1_40k             0.341    0.378     0.339            0.306
E1_40k             0.628    0.635     0.714            0.650
E2_40k             0.780    0.708     0.735            0.823
E2b_40k            0.402    0.361     0.517            0.323
E4_40k             0.646    0.608     0.732            0.664
E4r_40k            0.212    0.281     0.276            0.168
```

### T2 — 2x2 retained (feature tasks)
```
condition  random_feat  rewired_edge   both
arm                                        
B0_40k           3.220        -1.028  0.565
B1_40k           0.213         0.878  0.200
E1_40k           1.002         0.810  0.255
E2_40k           0.777         1.336  1.540
E2b_40k          0.323         1.022  0.409
```

### T3 — capability probes (AUC, chance 0.50)
```
rule     count_threshold  in_degree  out_degree  existence  conjunction
arm                                                                    
B0_40k             0.478      0.515       0.524      0.515        0.513
B1_40k             0.527      0.509       0.523      0.525        0.526
E1_40k             0.672      0.627       0.515      0.535        0.534
E2_40k             0.589      0.513       0.583      0.623        0.626
E2b_40k            0.659      0.558       0.710      0.548        0.574
E4_40k             0.245      0.291       0.359      0.508        0.468
E4r_40k            0.178      0.148       0.369      0.449        0.377
```
