# NM single-source transfer matrix — findings

**Setup.** Train a neighbor-matching (NM) model on **each of the 8 graphs alone**
(plain arch, no aug, matched-40k, `state_dict_40000`), then evaluate every model on
**all 8** graphs — NM 30-way / 3-shot, ROC-AUC. Rows = train source, cols = test
graph. **Diagonal = in-domain specialist; off-diagonal = zero-overlap transfer.**
This is the single-source counterpart to the merged NM "interpolation ladder"
(`scripts/experiments/analysis/transfer/ladders/prodigy_nm/canonical/nm_ladder/RESULTS.md`).
**1 seed.** Data: `nm_single_source_matrix.csv` (wide), `_long.csv` (acc/f1 too).

## The matrix (ROC-AUC, `*` = in-domain diagonal)

```
train\test    ukr     covid   midterm cov_pol elec20  ukr_sus twibot  cp_hk
ukr          .947*    .973    .881    .839    .826    .789    .922    .714
covid        .926     .981*   .884    .850    .835    .786    .926    .720
midterm      .797     .879    .925*   .835    .805    .644    .861    .626
cov_pol      .630     .699    .688    .915*   .783    .551    .737    .544
elec20       .602     .655    .680    .787    .952*   .563    .710    .548
ukr_susp     .770     .831    .725    .733    .728    .964*   .765    .623
twibot20     .869     .947    .860    .843    .802    .711    .949*   .689
cp_hk        .681     .758    .763    .683    .641    .602    .738    .906*
```

**Sanity check passed:** `ukr→ukr` = **.947** reproduces ladder rung-1 `.948` and
`nm_transfer_matrix`'s single-ukr `.9497` — the protocol is consistent, so these
numbers slot directly into the ladder table.

## Findings

### 1. Every graph has a strong specialist, and the specialist beats the all-8 merged model in-domain — on all 8 columns.
Diagonals span **.906–.981** (weakest is cp_hk, strongest covid/ukr_susp). Against
the merged all-8 model (ladder rung 4) the in-domain specialist wins **every** column:

| col | specialist | all-8 merged | Δ (spec − merged) |
|---|---|---|---|
| ukr | .947 | .934 | +.013 |
| covid | .981 | .975 | +.006 |
| midterm | .925 | .908 | +.017 |
| cov_pol | .915 | .906 | +.009 |
| elec20 | .952 | .920 | **+.032** |
| ukr_susp | .964 | .931 | **+.033** |
| twibot20 | .949 | .937 | +.012 |
| cp_hk | .906 | .867 | **+.039** |

The gap is small on the big twitter graphs (covid +.006, ukr +.013) and **largest on
the small/topical graphs** (cp_hk +.039, ukr_susp +.033, elec20 +.032): merging dilutes
a niche graph's signal more than a large one's. So merging carries a **consistent but
modest in-domain tax**, paid mostly by the small graphs — while buying one model that
covers all 8 instead of eight separate ones.

### 2. covid and ukr are near-universal donors; the small/topical graphs are not.
Ranking sources by **mean off-diagonal transfer OUT**:

`ukr .849 ≈ covid .847 > twibot20 .817 > midterm .778 > ukr_susp .739 > cp_hk .695 > cov_pol .662 > elec20 .649`

**covid is the single best cross-source donor to 7 of 8 targets** (the exception is
`ukr_susp`, where its Ukraine-family sibling `ukr` edges it out .789 vs .786). Notably
`ukr→covid` = **.973**, nearly covid's own .981 ceiling — the two big twitter graphs
transfer to each other almost for free. In contrast the topical specialists
(`election2020`, `covid_political`, `cp_hk`) transfer **poorly** everywhere off their
own diagonal (rows in the .55–.79 range): a strong in-domain model on a narrow graph
does **not** generalize.

### 3. Merging's payoff is exactly on the columns no single source covers.
Compare the all-8 merged model to the **best available single source** for each target:
- On graphs a big source already covers, merging barely helps (covid col: best single
  `ukr→covid` .973 ≈ merged .975).
- On the **isolated** graphs, merging leaps past any single source. all-8 vs the best
  foreign donor: **cp_hk .867 vs .720** (+.147), **ukr_susp .931 vs .789** (+.142),
  **elec20 .920 vs .835** (+.085). And vs a single ukr specialist (ladder rung 1) the
  merged model gains **+.14–.16** on cp_hk / ukr_susp.

So the picture is a clean trade: **merging gives up ~.006–.04 of in-domain peak to buy
+.09–.16 of robustness on the graphs that have no strong donor.** That is precisely the
"rung-4 jump" seen in the ladder, now bracketed by the single-source ceilings.

### 4. Transfer is asymmetric, and cp_hk is an island.
Big→small is moderate (covid→cp_hk .720), small→big is weak (cp_hk→covid .758,
cp_hk→ukr .681). By **receptivity** (mean transfer IN), the hardest targets are
`cp_hk` (.638) and `ukr_susp` (.664). **cp_hk is the most isolated graph** — weak donor
*and* hardest target — consistent with it being topically/linguistically distinct
(HK/China political). ukr_susp is a hard target but its own specialist is the highest
diagonal (.964), i.e. easy to fit in-domain, hard to reach from outside.

## Caveats
- **1 seed** — sub-.01 AUC gaps (e.g. covid col specialist vs merged +.006, or the
  covid/ukr donor tie) are within run-to-run noise; the ≥.03 effects (small-graph
  in-domain tax, isolation of cp_hk) are the trustworthy ones. Configs take `--seed N`.
- **Small-graph off-diagonals are noisier** — `election2020` / `ukr_rus_suspended` are
  sparse (eval `nm_n_query=1`); they trained fine at 30-way but their transfer cells
  rest on fewer episodes.
- All cells at **matched-40k**, plain arch — same protocol as the ladder, so the two
  tables are directly comparable. See also `../_cross/NM_MERGED_VS_SINGLE_SUMMARY.md` (the
  earlier ukr/covid + covid/midterm pairwise version this generalizes).
