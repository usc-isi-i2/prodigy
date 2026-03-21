# Midterm Dataset — `retweet_graph.pt`

Graph of US politicians and their retweet interactions, split into historical and future edges for temporal link prediction.

## File Structure

```python
{
    'x':                              tensor([308983, 395]),   # node features
    'edge_index':                     tensor([2, 807036]),     # all retweets (leaks future!)
    'edge_attr':                      tensor([807036, 4]),     # edge features for edge_index
    'edge_attr_feature_names':        ['first_retweet_time', 'n_retweets', 'avg_rt_fav', 'avg_rt_reply'],
    'user_ids':                       array([308983]),         # original Twitter IDs
    'feature_names':                  [...],                   # 395 column names
    'y':                              tensor([308983]),        # state labels, -1=unlabeled
    'label_names':                    ['AK', 'AL', ...],      # 51 US states
    'edge_index_views': {
        'retweet_all':       tensor([2, 807036]),
        'temporal_history':  tensor([2, 657932]),
    },
    'edge_attr_views': {
        'retweet_all':       tensor([807036, 4]),
        'temporal_history':  tensor([657932, 4]),
    },
    'edge_attr_feature_names_views': {
        'retweet_all':       ['first_retweet_time', 'n_retweets', 'avg_rt_fav', 'avg_rt_reply'],
        'temporal_history':  ['first_retweet_time', 'n_retweets', 'avg_rt_fav', 'avg_rt_reply'],
    },
    'target_edge_index_views': {
        'temporal_new':      tensor([2, 149104]),
    },
    'future_edge_index':              tensor([2, 149104]),     # same as temporal_new
}
```

## Nodes

| Field | Shape | dtype | Description |
|---|---|---|---|
| `x` | `[308983, 395]` | float32 | Node feature matrix (see Feature Views); values in [-2.83, 27.21] |
| `y` | `[308983]` | int64 | State label index (0–50); -1 = unlabeled |
| `user_ids` | `[308983]` | int64 | Original Twitter user IDs |
| `label_names` | 51 strings | — | US state abbreviations: AK, AL, AR, AZ, CA, ... WY |
| `feature_names` | 395 strings | — | Column names for `x` (see Feature Views below) |

## Edge Views

All views share the same 4-dim edge features: `first_retweet_time`, `n_retweets`, `avg_rt_fav`, `avg_rt_reply` (values in [0, 531]).

| Key | Shape | Description |
|---|---|---|
| `edge_index` (default) | `[2, 807036]` | All retweets — same as `retweet_all`. **Leaks future edges, do not use for temporal LP.** |
| `edge_index_views["retweet_all"]` | `[2, 807036]` | All retweets across full time period |
| `edge_index_views["temporal_history"]` | `[2, 657932]` | Retweets up to the time cutoff — use as background graph for temporal LP |
| `target_edge_index_views["temporal_new"]` | `[2, 149104]` | Retweets after the time cutoff — use as prediction target for temporal LP |
| `future_edge_index` | `[2, 149104]` | Same as `temporal_new`, legacy top-level key |

Note: `temporal_history` + `temporal_new` = `retweet_all` (657,932 + 149,104 = 807,036).

## Node Features

395 total dimensions, split into two groups:

**Statistics (11 dims):** `subscriber_count`, `verified`, `avg_favorites`, `avg_comments`, `avg_score`, `avg_n_hashtags`, `avg_n_mentions`, `avg_has_media`, `post_count`, `in_degree`, `out_degree`

**Text embeddings (384 dims):** `emb_0` ... `emb_383` — sentence embeddings of user profile/content (384-dim model, e.g. `all-MiniLM-L6-v2`)

Use `--midterm_feature_subset` to select a subset:

| Value | Dims | Description |
|---|---|---|
| `all` | 395 | All features |
| `emb_only` | 384 | Text embedding dims only |
| `stats_only` | 11 | Statistics dims only |
| `keep:<f1,f2,...>` | varies | Named columns only |
| `drop:<f1,f2,...>` | varies | All except named columns |
| `constant1` | 1 | Single constant feature (ablation) |

## Recommended Command for Temporal Link Prediction

```bash
python experiments/run_single_experiment.py \
  --dataset midterm \
  --root /scratch1/eibl/data/midterm/graphs \
  --graph_filename retweet_graph.pt \
  --task temporal_link_prediction \
  --midterm_edge_view temporal_history \
  --midterm_target_edge_view temporal_new \
  --midterm_edge_feature_subset none \
  --midterm_feature_subset emb_only \
  --input_dim 384 \
  --n_way 1 \
  --midterm_binary_lp True \
  --midterm_lp_neg_ratio 1
```

Key flags:
- `--midterm_edge_view temporal_history` — prevents future edge leakage into background graph
- `--midterm_target_edge_view temporal_new` — sets the future edges as the prediction target
- `--midterm_binary_lp True` — binary (positive/negative) framing instead of multiway
- `--input_dim 384` — must match feature dim when using `emb_only`
