#!/usr/bin/env python3
"""Generate the canonical naive-global two-hop NM ladder configs."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
MANIFEST = HERE / "manifest.tsv"

RUNGS = [
    (2, "covid19_twitter", "covid", "ukr_rus_covid_retweet_graph.pt"),
    (3, "midterm", "midterm", "ukr_rus_covid_midterm_retweet_graph.pt"),
    (4, "covid_political", "covid_political", "ukr_rus_covid_midterm_4src_retweet_graph.pt"),
    (5, "election2020", "election2020", "ukr_rus_covid_midterm_5src_retweet_graph.pt"),
    (6, "ukr_rus_suspended", "ukr_rus_suspended", "ukr_rus_covid_midterm_6src_retweet_graph.pt"),
    (7, "twibot20", "twibot20", "ukr_rus_covid_midterm_7src_retweet_graph.pt"),
    (8, "cp_hk_twitter", "cp_hk", "ukr_rus_covid_midterm_all8_retweet_graph.pt"),
]

CONFIG_TEMPLATE = """\
# Naive-global 2-hop NM ladder -- canonical Order A, rung {rung}/8.
# Adds {newcomer}; the entire disjoint {rung}-source merge is sampled as one graph.
# Intentionally omit every graph_id/source-balancing option. Mixed-source episodes and
# node-mass weighting are the intervention; all other settings match nm_ladder_nhop2.
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: {graph_filename}
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
prefix: nm_ladder_global_h2m_r{rung}
"""

SMOKE_CONFIG = CONFIG_TEMPLATE.format(
    rung=2,
    newcomer="covid",
    graph_filename="ukr_rus_covid_retweet_graph.pt",
).replace(
    "dataset_len_cap: 10000\nval_len_cap: 500\ntest_len_cap: 500",
    "dataset_len_cap: 20\nval_len_cap: 20\ntest_len_cap: 20",
).replace(
    "epochs: 4\neval_step: 100000\ncheckpoint_step: 10000",
    "epochs: 1\neval_step: 100000\ncheckpoint_step: 20\ncheckpoint_steps: \"0,20\"",
).replace("workers: 2", "workers: 1").replace(
    "prefix: nm_ladder_global_h2m_r2",
    "prefix: nm_ladder_global_h2m_smoke",
)


def expected_files() -> dict[Path, str]:
    files = {
        CONFIG_DIR / f"train_r{rung}.yaml": CONFIG_TEMPLATE.format(
            rung=rung,
            newcomer=newcomer,
            graph_filename=graph_filename,
        )
        for rung, _dataset, newcomer, graph_filename in RUNGS
    }
    files[CONFIG_DIR / "smoke.yaml"] = SMOKE_CONFIG
    lines = ["rung\tprefix\tnewcomer_dataset\tnewcomer_graph_id\tgraph_filename"]
    lines.extend(
        f"{rung}\tnm_ladder_global_h2m_r{rung}\t{dataset}\t{newcomer}\t{graph_filename}"
        for rung, dataset, newcomer, graph_filename in RUNGS
    )
    files[MANIFEST] = "\n".join(lines) + "\n"
    return files


def check(files: dict[Path, str]) -> int:
    failed = False
    expected_configs = {path for path in files if path.suffix == ".yaml"}
    actual_configs = set(CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else set()
    for path in sorted(actual_configs - expected_configs):
        print(f"unexpected generated config: {path}")
        failed = True
    for path, expected in files.items():
        if not path.is_file():
            print(f"missing generated file: {path}")
            failed = True
            continue
        actual = path.read_text(encoding="utf-8")
        if actual != expected:
            failed = True
            print(f"generated file drift: {path}")
            print("\n".join(difflib.unified_diff(
                actual.splitlines(), expected.splitlines(),
                fromfile=str(path), tofile=f"{path} (expected)", lineterm="",
            )))
    if not failed:
        print("OK: seven global-merge rung configs, smoke config, and manifest are current")
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()
    if args.list_configs:
        for rung, *_rest in RUNGS:
            print(f"configs/train_r{rung}.yaml")
        return 0
    files = expected_files()
    if args.check:
        return check(files)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    print("wrote seven global-merge rung configs, smoke config, and manifest.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
