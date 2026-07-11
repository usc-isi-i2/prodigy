# topology_feature_ssl — RESULTS

> **Final matched-40k results (B0/B1/E1/E2/E2b) are in [`RESULTS_matched40k.md`](RESULTS_matched40k.md)** and summarized in `FINDINGS.md`. The tables below are the earlier B0/B1/E1 render, kept for reference. Headline: E1 wins features, E2 wins static-LP (0.76), E2b's drop-BN lifts count probes but crashes LP — no encoder arm clears the joint bar → E4.

_Auto-rendered from the parsed CSVs (see the notebook for the interactive version). Primary evidence: T2 (2×2) + T3 (probes); T1 is confirmatory. Headline is `min(feature, topological)`, never the mean._

### Free preview — NM vs FP (regression, test Spearman)

```
model          dataset            target     fp     nm  fp-nm
0      covid19_twitter  account_age_days -0.045  0.010 -0.055
1      covid19_twitter  favourites_count  0.021 -0.022  0.043
2      covid19_twitter   followers_count -0.024 -0.062  0.038
3      covid19_twitter     friends_count -0.025 -0.055  0.029
4      covid19_twitter      listed_count -0.090 -0.043 -0.047
5      covid19_twitter    statuses_count  0.033 -0.100  0.133
6              midterm  account_age_days -0.071  0.157 -0.228
7              midterm  favourites_count  0.003 -0.004  0.007
8              midterm   followers_count -0.013 -0.071  0.058
9              midterm     friends_count -0.014 -0.043  0.029
10             midterm      listed_count -0.042  0.013 -0.055
11             midterm    statuses_count  0.007 -0.058  0.065
12            twibot20  account_age_days -0.169  0.010 -0.179
13            twibot20   followers_count -0.383 -0.211 -0.172
14            twibot20     friends_count -0.182 -0.141 -0.042
15            twibot20      listed_count -0.421 -0.100 -0.321
16            twibot20    statuses_count -0.320 -0.195 -0.125
17     ukr_rus_twitter  account_age_days -0.042  0.049 -0.091
18     ukr_rus_twitter  favourites_count  0.052 -0.058  0.109
19     ukr_rus_twitter   followers_count  0.039 -0.060  0.099
20     ukr_rus_twitter     friends_count  0.045 -0.055  0.100
21     ukr_rus_twitter      listed_count  0.008 -0.062  0.069
22     ukr_rus_twitter    statuses_count  0.054 -0.109  0.163
```


mean(fp-nm) = -0.016 (fp wins 13/23) -> fp does NOT beat nm

### T1 — Benchmark (test)

```
     cls_AUC  reg_content_age  reg_struct  reg_struct_Δ_vs_leak  staticLP_AUC
arm                                                                          
B0     0.791            0.030      -0.011                -0.170         0.722
B1     0.782           -0.113      -0.102                -0.261         0.359
E1     0.774            0.131       0.144                -0.015         0.758
```


leakage baseline (raw-structural -> followers/statuses) = 0.159

### T2 — 2×2 ablation (fraction of real/real retained; feature tasks)

```
condition  real·real  random_feat  rewired_edge  both
arm                                                  
B0               1.0         0.39          1.11  0.19
B1               1.0         0.24          0.93  0.47
E1               1.0         0.21          0.94 -0.00
```

### T3 — capability probes (linear-probe AUC, chance = 0.50)

```
rule  count_threshold  in_degree  out_degree  existence  conjunction
arm                                                                 
B0               0.51       0.50        0.51       0.51         0.51
B1               0.52       0.51        0.52       0.50         0.52
E1               0.64       0.59        0.53       0.56         0.52
```
