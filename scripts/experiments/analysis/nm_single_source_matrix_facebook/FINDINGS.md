# NM single-source matrix with Facebook — findings

**Complete (2026-08-06).** This extends the historical 8×8 single-source
neighbor-matching matrix to **9×9** by adding `facebook-page-reference` as both a
training source and an evaluation target. All 17 new cells use the historical
protocol: plain architecture, one hop, 30-way/3-shot, `n_query=4`, 500 fixed test
episodes, seed 0, and the matched `state_dict_40000.ckpt` checkpoint. The Facebook
run uses the 119,228-node structural graph, excluding attributed pages with no
reference edges.

Rows are training sources and columns are test graphs. The metric is ROC-AUC;
`*` marks an in-domain specialist.

```
train\test  ukr    covid  mid    covpol elec   ukrsus twibot cp_hk  facebook
ukr         .947*  .973   .881   .839   .826   .789   .922   .714   .975
covid       .926   .981*  .884   .850   .835   .786   .926   .720   .974
midterm     .797   .879   .925*  .835   .805   .644   .861   .626   .901
covpol      .630   .699   .688   .915*  .783   .551   .737   .544   .742
elec20      .602   .655   .680   .787   .952*  .563   .710   .548   .686
ukrsus      .770   .831   .725   .733   .728   .964*  .765   .623   .856
twibot      .869   .947   .860   .843   .803   .712   .949*  .690   .954
cp_hk       .681   .758   .763   .683   .641   .602   .738   .906*  .804
facebook    .787   .871   .809   .805   .775   .652   .835   .627   .999*
```

## On-Facebook results

| training source | ROC-AUC | accuracy |
|---|---:|---:|
| Facebook | **.9995** | **.9336** |
| Ukraine/Russia Twitter | .9753 | .6395 |
| COVID-19 Twitter | .9742 | .6532 |
| TwiBot-20 | .9541 | .5632 |
| Midterm | .9006 | .4063 |
| Ukraine/Russia suspended | .8559 | .3096 |
| CP/HK Twitter | .8040 | .2266 |
| COVID political | .7420 | .1751 |
| Election 2020 | .6858 | .1356 |

## Findings

### 1. Facebook is the matrix's most receptive target.

The eight non-Facebook specialists average **.8615 AUC on Facebook**, higher than
the foreign-source mean for every Twitter target (the next easiest is COVID-19 at
.8265). Three Twitter-trained encoders transfer especially well:
Ukraine/Russia **.9753**, COVID-19 **.9742**, and TwiBot-20 **.9541**. The Facebook
specialist reaches **.9995**, only **+.0242** over the best foreign encoder.

This is strong evidence that the learned NM representation crosses platforms in
the Twitter→Facebook direction; Facebook-specific training improves the ceiling,
but is not necessary for high ranking performance on this target.

### 2. The Facebook specialist is a useful but mid-tier donor.

Its mean transfer to the eight Twitter graphs is **.7700**, fifth among the nine
sources:

`ukr .8650 > covid .8626 > twibot .8346 > midterm .7934 > facebook .7700 > ukr_sus .7535 > cp_hk .7087 > covpol .6718 > elec20 .6538`

Facebook is never the best foreign donor for a Twitter target. It ranks fourth on
six targets, fifth on COVID-political, and sixth on Election 2020. Its gap to the
best Twitter foreign donor ranges from **-.0450** (COVID-political) to **-.1398**
(Ukraine/Russia). Its strongest outgoing cell is Facebook→COVID-19 (**.8708**);
its weakest is Facebook→CP/HK (**.6266**).

### 3. Cross-platform transfer is directional.

For the eight paired directions, Twitter→Facebook is **+.0915 AUC higher on
average** than Facebook→Twitter. The largest asymmetries are Ukraine/Russia
suspended (**+.2041**), Ukraine/Russia Twitter (**+.1886**), and CP/HK
(**+.1774**). COVID-political and Election 2020 reverse the direction, but both
Facebook→Twitter cells remain below .81.

The resulting picture is not “Facebook does not transfer.” It is more specific:
the Facebook-only encoder is a respectable cross-source model, but the broad
Twitter specialists remain materially more portable, while Facebook itself is an
unusually easy target for those encoders.

## Evidence and caveats

- `data/facebook_extension_metrics.csv` preserves the exact 17 new results, Tucker
  metric paths, and checkpoint paths.
- `data/nm_single_source_matrix_9x9.csv` is the complete AUC matrix;
  `data/nm_single_source_matrix_9x9_long.csv` also contains accuracy and F1.
  `assemble_matrix.py` deterministically rebuilds both from the historical matrix
  plus the Facebook extension.
- This is **one training seed**. The evaluation episodes are paired and fixed by
  split, so rerunning with a different CLI seed would not constitute an independent
  episode sample. Treat small differences cautiously; the .09 directional mean and
  .05–.14 donor gaps are the robust-scale effects.
- The Facebook relation is page-reference, whereas the Twitter graphs use their
  own interaction semantics. The result demonstrates representation transfer, not
  that the graph-generating processes are identical.
