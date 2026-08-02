#!/usr/bin/env python3
"""Generate the canonical fair-2-hop NM ladder with a real edge split."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
MANIFEST = HERE / "manifest.tsv"

SOURCES = [
    ("ukr_rus", "ukraine"),
    ("covid", "covid"),
    ("midterm", "midterm"),
    ("covid_political", "covid-political"),
    ("election2020", "election2020-political"),
    ("ukr_rus_suspended", "ukraine-suspended"),
    ("twibot20", "twibot20"),
    ("cp_hk", "hongkong"),
]


def plan() -> list[dict[str, object]]:
    rows = []
    for rung in range(1, 9):
        active = [key for key, _ in SOURCES[:rung]]
        rows.append(
            {
                "rung": rung,
                "added": SOURCES[rung - 1][0],
                "sources": active,
                "prefix": f"nm_ladder_tts_h2m_r{rung}",
                "config": f"train_r{rung}.yaml",
            }
        )
    return rows


CONFIG_TEMPLATE = """\
# Leakage-free NM ladder, canonical order, rung {rung}/8.
# Active sources: {source_labels}.
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: ukr_rus_covid_midterm_all8_static_split_retweet_graph.pt
task_name: neighbor_matching

# A real edge split: held-out positive edges never enter message passing or training.
edge_view: static_background
target_edge_view: static_holdout
neighbor_matching_edge_split: true

feature_subset: all
original_features: true
emb_dim: 256
layers: S,U,M
gnn_type: sage
n_layer: 1
dropout: 0

# Fair compute-matched two-hop protocol from pretrain_saturation_nhop2.
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

neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: {subset}

epochs: 4
eval_step: 100000
checkpoint_step: 10000
workers: 2
device: 0
seed: 0
prefix: {prefix}
"""


SMOKE_CONFIG = """\
# Twenty-step split-aware resource/protocol smoke on election2020.
dataset: election2020
root: /dataMeR1/phil/data/election2020/graphs
graph_filename: retweet_graph.pt
task_name: neighbor_matching
edge_view: static_background
target_edge_view: static_holdout
neighbor_matching_edge_split: true
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
dataset_len_cap: 20
val_len_cap: 20
test_len_cap: 20
epochs: 1
eval_step: 100000
checkpoint_step: 20
checkpoint_steps: "0,20"
workers: 1
device: 0
seed: 0
prefix: nm_ladder_tts_h2m_smoke_election
"""


def render_config(row: dict[str, object]) -> str:
    active = list(row["sources"])
    labels = dict(SOURCES)
    return CONFIG_TEMPLATE.format(
        rung=row["rung"],
        source_labels=", ".join(labels[key] for key in active),
        subset=",".join(active),
        prefix=row["prefix"],
    )


def render_manifest(rows: list[dict[str, object]]) -> str:
    lines = ["rung\tadded\tn_sources\tsources\tmodel_prefix\tconfig"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["rung"]),
                    str(row["added"]),
                    str(len(row["sources"])),
                    ",".join(row["sources"]),
                    str(row["prefix"]),
                    str(row["config"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def expected_files() -> dict[Path, str]:
    rows = plan()
    files = {CONFIG_DIR / row["config"]: render_config(row) for row in rows}
    files[CONFIG_DIR / "smoke_election.yaml"] = SMOKE_CONFIG
    files[MANIFEST] = render_manifest(rows)
    return files


def check(files: dict[Path, str]) -> int:
    failed = False
    for path, expected in files.items():
        actual = path.read_text() if path.exists() else ""
        if actual != expected:
            failed = True
            print("".join(difflib.unified_diff(
                actual.splitlines(True), expected.splitlines(True),
                fromfile=str(path), tofile=f"expected:{path}",
            )), end="")
    extras = set(CONFIG_DIR.glob("*.yaml")) - {
        path for path in files if path.suffix == ".yaml"
    }
    if extras:
        failed = True
        print("unexpected generated configs:", *sorted(extras), sep="\n  ")
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()
    if args.list_configs:
        for row in plan():
            print(f"configs/{row['config']}")
        return 0
    files = expected_files()
    if args.check:
        return check(files)
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
