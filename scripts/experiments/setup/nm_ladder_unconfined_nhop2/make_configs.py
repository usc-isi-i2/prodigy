#!/usr/bin/env python3
"""Generate the compute-matched two-hop unconfined NM ladder configs."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
GRAPHS = {
    1: ("/dataMeR1/phil/data/ukr_rus_twitter/graphs", "retweet_graph_parquet.pt"),
    2: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_retweet_graph.pt"),
    3: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_retweet_graph.pt"),
    4: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_4src_retweet_graph.pt"),
    5: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_5src_retweet_graph.pt"),
    6: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_6src_retweet_graph.pt"),
    7: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_7src_retweet_graph.pt"),
    8: ("/dataMeR1/phil/data/merged/graphs", "ukr_rus_covid_midterm_all8_retweet_graph.pt"),
}


def render(rung: int, root: str, graph: str) -> str:
    return f"""# Compute-matched 2-hop UNCONFINED NM ladder, canonical order, rung {rung}/8.
# Episodes are sampled naively from the merged component union: there is deliberately
# no graph_id confinement, source balancing, subset, or source sequence.
dataset: covid19_twitter
root: {root}
graph_filename: {graph}
task_name: neighbor_matching

edge_view: default
feature_subset: all
original_features: true

emb_dim: 256
layers: S,U,M
gnn_type: sage
n_layer: 1
dropout: 0
n_hop: 2
neighbor_sampling_hop_sizes: "9,9"
neighbor_sampling_node_limit: 101
neighbor_matching_walk_hops: 1

n_way: 30
n_shots: 3
n_query: 4
batch_size: 1
dataset_len_cap: 10000
val_len_cap: 500
test_len_cap: 500

epochs: 4
eval_step: 100000
checkpoint_step: 10000
workers: 2
device: 0
seed: 0
prefix: nm_ladder_unconf_h2m_r{rung}
"""


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for rung, (root, graph) in GRAPHS.items():
        (CONFIGS / f"train_r{rung}.yaml").write_text(render(rung, root, graph))
    print("wrote 8 unconfined two-hop ladder configs")


if __name__ == "__main__":
    main()
