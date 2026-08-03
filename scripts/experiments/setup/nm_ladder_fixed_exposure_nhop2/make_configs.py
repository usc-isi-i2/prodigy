#!/usr/bin/env python3
"""Generate fixed-exposure, compute-matched 2-hop NM ladder configs.

Each rung receives 10,000 optimizer steps per active source: rung 1 trains for
10k steps, rung 2 for 20k, ..., and rung 8 for 80k. The three orders contain
24 rows but only 21 unique source sets, so duplicate sets share one checkpoint.
"""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
MANIFEST = HERE / "manifest.tsv"
STEPS_PER_SOURCE = 10_000

# key, dataset_key, canonical name
SOURCES = [
    ("ukr_rus", "ukr_rus_twitter", "ukraine"),
    ("covid", "covid19_twitter", "covid"),
    ("midterm", "midterm", "midterm"),
    ("covid_political", "covid_political", "covid-political"),
    ("election2020", "election2020", "election2020-political"),
    ("ukr_rus_suspended", "ukr_rus_suspended", "ukraine-suspended"),
    ("twibot20", "twibot20", "twibot20"),
    ("cp_hk", "cp_hk_twitter", "hongkong"),
]
SOURCE_BY_KEY = {row[0]: row for row in SOURCES}

ORDERS = {
    "A": [
        "ukr_rus", "covid", "midterm", "covid_political",
        "election2020", "ukr_rus_suspended", "twibot20", "cp_hk",
    ],
    "B": [
        "covid", "ukr_rus", "twibot20", "midterm",
        "ukr_rus_suspended", "cp_hk", "covid_political", "election2020",
    ],
}
ORDERS["C"] = list(reversed(ORDERS["B"]))

ORDER_DESCRIPTIONS = {
    "A": "published topical order",
    "B": "donor strength descending",
    "C": "donor strength ascending (reverse of B)",
}


def canonical(key: str) -> str:
    return SOURCE_BY_KEY[key][2]


def target_step(row: dict[str, object]) -> int:
    return len(row["sources"]) * STEPS_PER_SOURCE


def plan() -> list[dict[str, object]]:
    """Return all 24 order/rung rows mapped to 21 unique model runs."""
    known: dict[frozenset[str], dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for order in ("A", "B", "C"):
        sequence = ORDERS[order]
        for rung in range(1, 9):
            sources = sequence[:rung]
            source_set = frozenset(sources)
            primary = known.get(source_set)
            if primary is None:
                prefix = f"nm_ladder_fx10k_h2m_ord{order}_r{rung}"
                config = f"train_ord{order}_r{rung}.yaml"
                primary = {
                    "primary_order": order,
                    "primary_rung": rung,
                    "prefix": prefix,
                    "config": config,
                }
                known[source_set] = primary
                status = "train"
            else:
                status = "reuse_fx10k_h2m"
            row: dict[str, object] = {
                "order": order,
                "rung": rung,
                "added": sequence[rung - 1],
                "sources": sources,
                "status": status,
                **primary,
            }
            row["target_step"] = target_step(row)
            rows.append(row)
    return rows


def unique_rows(rows: list[dict[str, object]] | None = None) -> list[dict[str, object]]:
    return [row for row in (rows or plan()) if row["status"] == "train"]


def phase_rows(phase: str) -> list[dict[str, object]]:
    rows = unique_rows()
    if phase == "A":
        return [row for row in rows if row["primary_order"] == "A"]
    if phase == "robustness":
        return [row for row in rows if row["primary_order"] in {"B", "C"}]
    if phase == "all":
        return rows
    raise ValueError(f"unknown phase: {phase}")


CONFIG_TEMPLATE = """\
# Fixed-exposure 2-hop NM graph ladder -- order {order}, rung {rung}/8.
# Order: {order_description}.
# Adds: {added} ({added_canonical}).
# Active sources: {sources_pretty}.
# Budget: {rung} sources x 10,000 episodes/source = {target_step:,} total steps.
#
# All rungs read the disjoint all8 merge and restrict eligible graph_ids. Since
# source components have no cross-source edges, sampled neighborhoods cannot leave
# their source component. The sampler matches pretrain_saturation_nhop2.
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: ukr_rus_covid_midterm_all8_retweet_graph.pt
task_name: neighbor_matching

edge_view: default
feature_subset: all
original_features: true

# Fair 2-hop sampler: change context radius without increasing subgraph size.
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

# Uniform source choice makes the expected exposure 10,000 episodes/source.
neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: {subset}

# Total steps = epochs x dataset_len_cap = rung x 10,000.
epochs: {rung}
eval_step: 100000
checkpoint_step: 10000
workers: 2
device: 0
seed: 0
prefix: {prefix}
"""


SMOKE_CONFIG = """\
# Fixed-exposure 2-hop resource smoke: election2020 only (high average degree).
# This distinct 20-step run is never used in ladder analysis.
dataset: election2020
root: /dataMeR1/phil/data/election2020/graphs
graph_filename: retweet_graph.pt
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
prefix: nm_ladder_fx10k_h2m_smoke_election
"""


def render_config(row: dict[str, object]) -> str:
    sources = list(row["sources"])
    return CONFIG_TEMPLATE.format(
        order=row["primary_order"],
        rung=row["primary_rung"],
        order_description=ORDER_DESCRIPTIONS[str(row["primary_order"])],
        added=row["added"],
        added_canonical=canonical(str(row["added"])),
        sources_pretty=", ".join(canonical(str(key)) for key in sources),
        target_step=target_step(row),
        subset=",".join(str(key) for key in sources),
        prefix=row["prefix"],
    )


def render_manifest(rows: list[dict[str, object]]) -> str:
    fields = [
        "order", "rung", "added", "n_sources", "target_step", "sources", "status",
        "primary_order", "primary_rung", "model_prefix", "config",
    ]
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["order"]),
                    str(row["rung"]),
                    str(row["added"]),
                    str(len(row["sources"])),
                    str(row["target_step"]),
                    ",".join(str(key) for key in row["sources"]),
                    str(row["status"]),
                    str(row["primary_order"]),
                    str(row["primary_rung"]),
                    str(row["prefix"]),
                    str(row["config"]),
                ]
            )
        )
    return "\n".join(lines) + "\n"


def expected_files() -> dict[Path, str]:
    rows = plan()
    files = {
        CONFIG_DIR / str(row["config"]): render_config(row)
        for row in unique_rows(rows)
    }
    files[CONFIG_DIR / "smoke_election.yaml"] = SMOKE_CONFIG
    files[MANIFEST] = render_manifest(rows)
    return files


def check_files(files: dict[Path, str]) -> int:
    failed = False
    expected_paths = set(files)
    actual_configs = set(CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.exists() else set()
    expected_configs = {path for path in expected_paths if path.suffix == ".yaml"}
    for path in sorted(actual_configs - expected_configs):
        print(f"unexpected generated config: {path}")
        failed = True
    for path, expected in files.items():
        if not path.is_file():
            print(f"missing generated file: {path}")
            failed = True
            continue
        actual = path.read_text(encoding="utf-8")
        if actual == expected:
            continue
        failed = True
        print(f"generated file drift: {path}")
        diff = difflib.unified_diff(
            actual.splitlines(), expected.splitlines(),
            fromfile=str(path), tofile=f"{path} (expected)", lineterm="",
        )
        for line in diff:
            print(line)
    if not failed:
        print("OK: 21 train configs + smoke config + manifest are current")
    return int(failed)


def write_files(files: dict[Path, str]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for path, contents in files.items():
        path.write_text(contents, encoding="utf-8")
    print("wrote 21 fixed-exposure configs, one smoke config, and manifest.tsv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if generated files drift")
    parser.add_argument(
        "--list-configs", choices=["A", "robustness", "all"],
        help="print generated config paths for one launch phase",
    )
    args = parser.parse_args()

    if args.list_configs:
        for row in phase_rows(args.list_configs):
            print(f"configs/{row['config']}")
        return 0
    files = expected_files()
    if args.check:
        return check_files(files)
    write_files(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
