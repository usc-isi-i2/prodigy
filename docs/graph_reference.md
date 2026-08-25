# Graph reference

This is the human-readable overview of every graph currently registered in the
project. The machine-readable source of truth remains
[`graph_catalog.json`](graph_catalog.json); update that file first whenever a graph
or artifact changes, then refresh this document. Catalog inventory was last verified
on Tucker on **2026-08-04**. Structural diagnostics were generated on Tucker on
**2026-08-07** at commit `263e08e`.

## How to read this document

- **Canonical name** is the preferred name in prose.
- **Dataset key** is the compatibility name used by configs, command-line arguments,
  loaders, logs, and historical artifacts.
- All artifact paths are relative to `/dataMeR1/phil/data` on Tucker.
- **WCC** means weakly connected component: edge direction is ignored.
- An **isolate** is a node with no incoming or outgoing edges. It is also a singleton
  component, i.e. a connected component of size one.
- `—` means the fact is not yet recorded locally; it does not mean zero.

## Source graphs

| Canonical name | Dataset key | Nodes | Edges | Artifact | Features | Labels | Supported tasks |
|---|---|---:|---:|---:|---:|---|---|
| Ukraine | `ukr_rus_twitter` | 10,400,775 | 76,851,913 | 37.006 GB | 768 | none | NM, temporal LP, static LP, regression |
| COVID | `covid19_twitter` | 23,012,850 | 107,222,182 | 78.267 GB | 768 | none | NM, temporal LP, static LP, regression |
| Midterm | `midterm` | 341,908 | 899,979 | 1.123 GB | 768 | none | NM, temporal LP, static LP, regression |
| COVID Political | `covid_political` | 78,672 | 180,928 | 0.247 GB | 768 | binary node class | NM, classification |
| Ukraine Suspended | `ukr_rus_suspended` | 72,295 | 354,209 | 0.232 GB | 768 | binary node class | NM, classification |
| Election2020 Political | `election2020` | 78,932 | 2,818,603 | 0.312 GB | 768 | binary node class | NM, classification |
| TwiBot-20 | `twibot20` | 162,990 | 2,010,925 | 0.592 GB | 768 | binary node class | NM, static LP, classification, regression |
| Hong Kong | `cp_hk_twitter` | 333,800 | 1,184,379 | 1.108 GB | 768 | none | NM, temporal LP, static LP |
| Facebook Page Reference | `facebook_page_reference` | 150,000 | 167,622 | 0.487 GB | 768 | multiple classification and regression targets | NM, temporal LP, static LP, classification, regression |
| Cora | `cora` | 2,708 | — | — | 768 | 7-class node class | NM, static LP, classification |
| PubMed | `pubmed` | 19,717 | — | — | 768 | 3-class node class | NM, static LP, classification |

Task abbreviations: **NM** = neighbor matching and **LP** = link prediction.

### Source artifacts and construction

| Graph | Graph artifact | Node features | Edge features | Construction script |
|---|---|---|---|---|
| Ukraine | `ukr_rus_twitter/graphs/retweet_graph_parquet.pt` | GTE multilingual bio embeddings; missing users zero-filled | `n_retweets` | `scripts/graph_construction/generate_ukr_rus_retweet_graph_from_parquet.py` |
| COVID | `covid19_twitter/graphs/retweet_graph_parquet.pt` | GTE multilingual bio embeddings; missing users zero-filled | `n_retweets` | `scripts/graph_construction/generate_covid19_twitter_retweet_graph_from_parquet.py` |
| Midterm | `midterm/graphs/retweet_graph_parquet.pt` | GTE multilingual bio embeddings; missing users zero-filled | `n_retweets` | `scripts/graph_construction/generate_midterm_retweet_graph_from_parquet.py` |
| COVID Political | `covid_political/graphs/retweet_graph.pt` | mean-pooled GTE multilingual bio embeddings | `rt_weight`, `mn_weight` | `data/data/covid_political/scripts/generate_graph.py` |
| Ukraine Suspended | `ukr_rus_suspended/graphs/retweet_graph.pt` | GTE multilingual bio embeddings | `rt_weight`, `mn_weight` | `scripts/social_llm/generate_graph.py` |
| Election2020 Political | `election2020/graphs/retweet_graph.pt` | mean-pooled GTE multilingual bio embeddings | `rt_weight`, `mn_weight` | `scripts/social_llm/generate_graph.py` |
| TwiBot-20 | `twibot20/graphs/retweet_graph.pt` | GTE multilingual bio embeddings; missing users zero-filled | `n_retweets` | `scripts/graph_construction/generate_twibot20_retweet_graph.py` |
| Hong Kong | `cp_hk_twitter/graphs/retweet_graph.pt` | GTE multilingual bio embeddings | `n_retweets` | `scripts/graph_construction/generate_cp_hk_retweet_graph_from_parquet.py` |
| Facebook Page Reference | `facebook_page_reference/graphs/page_reference_graph.pt` | GTE multilingual page-description embeddings; 486 missing descriptions zero-filled | `n_reference_posts`, `n_content_reference_posts` | `scripts/graph_construction/generate_facebook_page_reference_graph.py` |
| Cora | `cora/graphs/citation_graph.pt` | normalized GTE embeddings of titles and abstracts | none recorded | `scripts/graph_construction/generate_tag_citation_graph.py` |
| PubMed | `pubmed/graphs/citation_graph.pt` | normalized GTE embeddings of titles and abstracts | none recorded | `scripts/graph_construction/generate_tag_citation_graph.py` |

## Connectivity profile

These counts use WCCs. The largest-component and isolate percentages use **nodes** as
their denominator; the final column uses **components** as its denominator. This
distinction matters: Hong Kong's isolates are 9.86% of nodes but 94.11% of all
components.

| Graph | Nodes | WCCs | Largest WCC nodes | Largest WCC % of nodes | Isolated nodes | Isolates % of nodes | Singleton % of WCCs | Other non-singleton WCCs | Nodes in those WCCs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| COVID | 23,012,850 | 313,316 | 22,298,924 | 96.90% | 0 | 0.00% | 0.00% | 313,315 | 713,926 |
| Ukraine | 10,400,775 | 85,464 | 10,198,843 | 98.06% | 0 | 0.00% | 0.00% | 85,463 | 201,932 |
| Midterm | 341,908 | 7,344 | 316,505 | 92.57% | 0 | 0.00% | 0.00% | 7,343 | 25,403 |
| Hong Kong | 333,800 | 34,965 | 294,595 | 88.25% | 32,904 | 9.86% | 94.11% | 2,060 | 6,301 |
| TwiBot-20 | 162,990 | 1,995 | 160,561 | 98.51% | 1,617 | 0.99% | 81.05% | 377 | 812 |
| Election2020 Political | 78,932 | 3 | 78,928 | 99.99% | 0 | 0.00% | 0.00% | 2 | 4 |
| COVID Political | 78,672 | 25,702 | 51,964 | 66.05% | 24,930 | 31.69% | 97.00% | 771 | 1,778 |
| Ukraine Suspended | 72,295 | 16,470 | 55,023 | 76.11% | 15,855 | 21.93% | 96.27% | 614 | 1,417 |
| Facebook Page Reference | 150,000 | — | — | — | — | — | — | — | — |
| Cora | 2,708 | — | — | — | — | — | — | — | — |
| PubMed | 19,717 | — | — | — | — | — | — | — | — |

The complete component-size arrays are not stored locally. Consequently, this table
cannot yet report the second-largest component, top-ten component sizes, or the size
distribution of the remaining components. The eight available rows come from
`scripts/experiments/analysis/graphs/structure_features/path_feature_coupling/data/dimension_diagnostics.json`.

## Merged training graphs

Merged artifacts are disjoint unions: source graphs remain separate components and no
cross-source edges are introduced. Their component counts therefore equal the sums of
their source component counts once every source has a measured connectivity profile.

| Canonical name | Dataset key | Sources | Nodes | Edges | Artifact |
|---|---|---|---:|---:|---:|
| Merged Ukraine–COVID | `merged_ukr_rus_covid` | Ukraine, COVID | 33,413,625 | 184,074,095 | 115.519 GB |
| Merged COVID–Midterm | `merged_covid_midterm` | COVID, Midterm | 23,354,758 | 108,122,161 | 79.535 GB |
| Merged Ukraine–COVID–Midterm | `merged_ukr_rus_covid_midterm` | Ukraine, COVID, Midterm | 33,755,533 | 184,974,074 | 116.645 GB |
| 4-source ladder | `merged_ukr_rus_covid_midterm_4src` | + COVID Political | 33,834,205 | 185,155,002 | 109.239 GB |
| 5-source ladder | `merged_ukr_rus_covid_midterm_5src` | + Election2020 Political | 33,913,137 | 187,973,605 | 109.531 GB |
| 6-source ladder | `merged_ukr_rus_covid_midterm_6src` | + Ukraine Suspended | 33,985,432 | 188,327,814 | 109.763 GB |
| 7-source ladder | `merged_ukr_rus_covid_midterm_7src` | + TwiBot-20 | 34,148,422 | 190,338,739 | 110.309 GB |
| Merged all eight | `merged_ukr_rus_covid_midterm_all8` | + Hong Kong | 34,482,222 | 191,523,118 | 111.386 GB |
| Merged all eight, static split | `merged_ukr_rus_covid_midterm_all8_static_split` | same eight sources | 34,482,222 | 191,523,118 | 114.450 GB |

All merged graphs support neighbor matching. The full source list and graph artifact
paths are authoritative in `graph_catalog.json`.

## Synthetic probe graphs

| Canonical name | Dataset key | Nodes | Edges | Artifact | Target |
|---|---|---:|---:|---:|---|
| Count threshold | `probe_count_threshold` | 4,000 | 20,158 | 0.013 GB | binary synthetic rule |
| In-degree | `probe_in_degree` | 4,000 | 20,025 | 0.013 GB | binary synthetic rule |
| Out-degree | `probe_out_degree` | 4,000 | 19,797 | 0.013 GB | binary synthetic rule |
| Existence | `probe_existence` | 4,000 | 19,782 | 0.013 GB | binary synthetic rule |
| Conjunction | `probe_conjunction` | 4,000 | 19,953 | 0.013 GB | binary synthetic rule |

All probe graphs have 768-dimensional node features and support node classification.
Their artifacts live beneath `synthetic_probes/graphs/`.

## Structural facts not yet inventoried

Component sizes describe fragmentation but not topology inside a component. A complete
structural fingerprint should eventually add, for each source graph and preferably for
its largest WCC:

- second-largest and top-ten component sizes;
- complete component-size histogram or rank-size distribution;
- minimum, maximum, mean, median, and quantiles of degree;
- degree-zero and degree-one node counts, plus degree concentration or Gini;
- density, global transitivity, and approximate mean local clustering;
- k-core distribution and maximum core number;
- bridge and articulation-point counts or scalable estimates;
- approximate distance distribution and effective diameter;
- community-size distribution and modularity, with the algorithm and seed recorded.

Exact diameter, exact betweenness, and some community algorithms are impractical on the
largest graphs. Any future inventory should record whether a result is exact, sampled,
or approximate, along with the library version, parameters, graph direction convention,
timestamp, and git commit.
