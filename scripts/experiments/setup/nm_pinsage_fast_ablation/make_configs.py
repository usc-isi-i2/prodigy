#!/usr/bin/env python3
"""Generate paired GraphSAGE/PinSAGE configs for the fast shared-graph protocol."""

from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGS = HERE / "configs"
ARMS = {
    "covid": "covid",
    "election": "election2020",
    "twibot20": "twibot20",
    "loo_twibot20": "ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,cp_hk,facebook_page_reference",
}

TEMPLATE = """\
# Paired fast-protocol sampler ablation: {arm}, {encoder}.
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt
task_name: neighbor_matching
edge_view: static_train
target_edge_view: static_test
neighbor_matching_edge_split: true
feature_subset: all
original_features: true

emb_dim: 256
layers: S,U,M
gnn_type: {gnn_type}
n_layer: 1
dropout: 0
n_hop: 2
neighbor_sampling_hop_sizes: "9,9"
neighbor_sampling_node_limit: 101
neighbor_matching_walk_hops: 1
neighbor_sampling_method: {method}
pinsage_num_walks: 64
pinsage_walk_length: 2
pinsage_restart_prob: 0.0
pinsage_topk: 100

n_way: 30
n_shots: 3
n_query: 4
batch_size: 4
learning_rate: 0.002
dataset_len_cap: 2500
val_len_cap: 500
test_len_cap: 500
neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: {sources}

epochs: 1
eval_step: 100000
checkpoint_step: 100000
checkpoint_steps: "100,300,900,2500"
workers: 2
device: 0
seed: {seed}
prefix: nm_{arm}_{encoder}_s{seed}_finalcore2500
"""


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    for arm, sources in ARMS.items():
        for encoder in ("graphsage", "pinsage"):
            for seed in (0, 1, 2):
                text = TEMPLATE.format(
                    arm=arm,
                    encoder=encoder,
                    seed=seed,
                    sources=sources,
                    gnn_type="sage" if encoder == "graphsage" else "pinsage",
                    method="uniform" if encoder == "graphsage" else "pinsage",
                )
                (CONFIGS / f"train_{arm}_{encoder}_s{seed}.yaml").write_text(text)


if __name__ == "__main__":
    main()
