# rapids

Shared data-preprocessing library for the RAPIDS project. All code that was previously copy-pasted across the three dataset pipelines (`midterm`, `ukr_rus_twitter`, `covid19_twitter`) lives here. Dataset scripts in `data/data/<dataset>/scripts/` are thin wrappers that supply dataset-specific loading and label logic, then delegate to this package for everything else.

## Package layout

```
rapids/
  utils.py                  # Small normalisation helpers (user IDs, handles)
  loaders/
    base.py                 # RecordLoader protocol / base class
    csv_loader.py           # StandardCsvLoader, UkrRusLoader
    json_loader.py          # JsonLoader
  graph/
    build.py                # All graph-building primitives
    validate.py             # Artifact validation
    inspect.py              # Human-readable artifact inspection
    eval.py                 # Node splitting, label downsampling, feature subsetting
  embeddings/
    pipeline.py             # run_embedding_pipeline, finalize_embeddings
```

---

## `rapids.utils`

Small, stateless helpers used throughout the pipeline.

| Function | Purpose |
|---|---|
| `normalize_user_id(val)` | Cast any value to `int`; returns `None` on failure |
| `normalize_handle(h)` | Lowercase, strip, return `None` for blanks / `"nan"` / `"<na>"` |

---

## `rapids.loaders` — the loader protocol

### `RecordLoader` base class (`loaders/base.py`)

Any new dataset format implements one method:

```python
from rapids.loaders.base import RecordLoader
import pandas as pd

class MyLoader(RecordLoader):
    def load_dataframe(self, path: str) -> pd.DataFrame:
        # Return a DataFrame following the column contract below.
        # Return pd.DataFrame() on unrecoverable error.
        ...
```

The base class provides two methods for free:

- `load_records(path)` — returns `load_dataframe(path).to_dict("records")`. Override for formats that are more naturally iterated before DataFrame conversion (see `JsonLoader`).
- `iter_files(paths)` — yields `(path, dataframe)` pairs, skipping empty or failed files with a warning.

### Column contract

| Column | Type | Required | Notes |
|---|---|---|---|
| `userid` | int64 | yes | Source node |
| `rt_userid` | int64 | yes | Destination node |
| `date` | str | for temporal views | Raw timestamp string |
| `screen_name` | str | no | Used for label attachment and embedding fallback |
| `rt_screen` | str | no | |
| `followers_count`, `verified`, `statuses_count` | numeric | no | Node features; zero-filled when absent |
| `rt_fav_count`, `rt_reply_count`, `sent_vader` | numeric | no | Node features |
| `hashtag`, `mentionsn`, `media_urls` | list-like | no | Node features |
| `description` | str | no | Used for weak-label inference |
| `urls`, `urls_list` | list-like | no | Used for weak-label inference |

If your dataset uses different column names for the edge endpoints, pass `src_col` / `dst_col` to `prepare_retweet_rows()` — they are renamed to the canonical names before any further processing.

### Concrete loaders

| Class | File | Dataset |
|---|---|---|
| `StandardCsvLoader` | `csv_loader.py` | midterm (plain CSV) |
| `UkrRusLoader` | `csv_loader.py` | ukr_rus_twitter (interleaved CSV, 66-col + 11-col sub-rows) |
| `JsonLoader` | `json_loader.py` | covid19_twitter (JSON array, wrapper dict, or NDJSON) |

All three are exported from `rapids.loaders`:

```python
from rapids.loaders import StandardCsvLoader, UkrRusLoader, JsonLoader
```

---

## `rapids.graph.build` — graph-building primitives

The core of the pipeline. Operates on a pandas DataFrame of retweet rows.

### Constants

```python
NODE_FEATURE_NAMES  # 11 account + graph-degree features
EDGE_FEATURE_NAMES  # 4 edge features
```

### Functions in pipeline order

```
prepare_retweet_rows(df, strict_dates, timestamp_format, src_col, dst_col)
    → rt DataFrame (only retweet rows, typed, timestamped)

trim_rt_to_max_nodes(rt, max_nodes)
    → rt DataFrame capped to top-N users by degree

build_user_index(rt)
    → (user_ids: List[int], u2i: Dict[int, int])

build_user_metadata(rt, user_ids)
    → handles: List[Optional[str]]

aggregate_edge_features(rt)
    → edge_df with log1p-transformed features

to_edge_tensors(edge_df, u2i, edge_feature_names)
    → (edge_index: LongTensor[2, E], edge_attr: FloatTensor[E, F])

build_node_features(rt, u2i, edge_df, feature_spec=None)
    → (x: FloatTensor[N, F], feature_names: List[str])

maybe_attach_embeddings(x, feature_names, user_ids, handles, embeddings_path, embedding_pool)
    → (x, feature_names, stats_dict)

drop_isolates_from_graph(x, edge_index, edge_attr, user_ids, y=None, handles=None)
    → (x, edge_index, edge_attr, user_ids, y, handles, u2i, n_dropped)

build_temporal_views(rt, u2i, history_fraction, future_target_mode, full_edge_index, full_edge_attr)
    → (graph_entries: Dict, stats: Dict)

save_graph(out_path, graph_obj, meta)
    → writes .pt artifact + .meta.json sidecar
```

### Feature spec

`build_node_features` accepts an optional `feature_spec` list. When omitted it uses `NODE_FEATURE_NAMES`. Columns not in the standard set are aggregated by mean from the raw retweet DataFrame and appended — useful for dataset-specific signals without touching shared code.

---

## `rapids.graph.validate`

```python
from rapids.graph.validate import validate_graph
import torch

raw = torch.load("retweet_graph.pt", map_location="cpu")
validate_graph(raw)   # raises AssertionError with a descriptive message on failure
```

Checks: required keys present, tensor shapes consistent, no NaN/Inf in node features, edge index in bounds, temporal view shapes match.

---

## `rapids.graph.inspect`

```python
from rapids.graph.inspect import inspect_graph
import torch

raw = torch.load("retweet_graph.pt", map_location="cpu")
inspect_graph(raw, topk=10)
```

Prints: node/edge counts, degree distribution, label distribution, feature names and stats, temporal view sizes.

---

## `rapids.graph.eval` — evaluation utilities

Dataset-agnostic helpers used by baseline scripts and the experiment framework.

```python
from rapids.graph.eval import (
    build_stratified_node_splits,
    apply_label_downsample,
    apply_feature_subset,
)
```

| Function | Purpose |
|---|---|
| `build_stratified_node_splits(labels, seed, train_frac, val_frac)` | Stratified train/val/test index arrays; unlabelled nodes (`label < 0`) excluded |
| `apply_label_downsample(labels, label_names, ratio_spec, seed)` | Mask nodes to a target class ratio, e.g. `"50:50"` |
| `apply_feature_subset(graph, subset_spec)` | Select / transform node features on a PyG `Data` object |

`apply_feature_subset` spec strings: `all`, `constant1`, `stats_only`, `emb_only`, `emb_only_plus_label`, `label_only`, `keep:<f1>,<f2>`, `drop:<f1>,<f2>`.

---

## `rapids.embeddings.pipeline`

Callback-based accumulation loop for building per-user mean-pooled text embeddings.

```python
from rapids.embeddings.pipeline import run_embedding_pipeline, finalize_embeddings

uid_to_row, sum_mat, cnt_arr, stats = run_embedding_pipeline(
    files=files,
    model=model,                     # SentenceTransformer
    get_records=lambda path: [...],  # returns iterable of records for one file
    get_uid=lambda rec: ...,         # extracts a hashable user id from a record
    get_text=lambda rec: "...",      # extracts text to embed from a record
    batch_size=1024,
    max_nodes=0,                     # 0 = no limit
    stop_after_max_nodes=True,
)

user_ids, embeddings, counts = finalize_embeddings(uid_to_row, sum_mat, cnt_arr, max_nodes=0)
# embeddings: FloatTensor[N, D], L2-normalised mean-pooled per user
```

The caller is responsible only for the three callables — everything else (accumulation, batching, mean-pooling, normalisation, progress logging) is handled internally.

---

## Adding a new dataset

1. Create a `RecordLoader` subclass (or standalone functions) in `rapids/loaders/` that returns a DataFrame following the column contract.
2. Write a `data/data/<dataset>/scripts/generate_retweet_graph.py` that handles raw loading and any dataset-specific label logic, then calls the shared `rapids.graph.build` functions.
3. Write a `build_user_embeddings.py` that provides `get_records`, `get_uid`, `get_text` callables and calls `run_embedding_pipeline`.
4. The `validate_graph.py` and `inspect_graph.py` scripts need no changes — they call `rapids.graph.validate` and `rapids.graph.inspect` directly.
5. Add a `scripts/config/<dataset>.sh` to wire the new dataset into the training and evaluation pipeline.
