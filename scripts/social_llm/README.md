# Social LLM Graph Builds

The Tucker build script creates graph artifacts from:

```text
/dataMeR1/phil/data/social_llm_data
```

It writes outputs under:

```text
/dataMeR1/phil/data/<dataset>/
```

`covid` is mapped to the repo dataset/output name `covid_political`.

## GTE Bio Embeddings

Profile bios are embedded with the same model used by the bio pretraining pipeline:

```text
Alibaba-NLP/gte-multilingual-base
revision=9bbca17d9273fd0d03d5725c7a4b0f6b45142062
bio-text-v001 normalization
768-dimensional, L2-normalized vectors
```

The row-aligned embedding artifact is:

```text
/dataMeR1/phil/data/<dataset>/embeddings/user_bio_embeddings_gte_multilingual_base.pt
```

## Graph Outputs

The graph builder emits one graph per `label_*` column:

```text
/dataMeR1/phil/data/<dataset>/graphs/retweet_graph_<label>.pt
```

Binary `0/1` labels are classification graphs. Continuous labels stay continuous and are emitted as regression graphs.

For compatibility with existing scripts, the first label graph is also copied to:

```text
/dataMeR1/phil/data/<dataset>/graphs/retweet_graph.pt
```

The Tucker GTE build uses `--embeddings-only` and
`--embedding_feature_prefix bio_emb`, so `data.x` is expected to be
768-dimensional GTE bio features (`bio_emb_0` ... `bio_emb_767`) without
numeric `user_data.csv` statistics. This keeps the social_llm graphs
feature-compatible with the parquet bio-embedding graphs.

Run on Tucker from the repo root:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/social_llm/build_gte_graphs_tucker.sh
```

Use whichever GPU is free. The script defaults to `covid_political`,
`election2020`, and `ukr_rus_suspended`, writes under
`/dataMeR1/phil/data/<dataset>/`, and moves any existing graph directory to
`graphs.backup_before_gte_<timestamp>` before writing new graph artifacts.
