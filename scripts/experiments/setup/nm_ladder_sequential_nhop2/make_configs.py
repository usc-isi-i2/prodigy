#!/usr/bin/env python3
"""Generate the compute-matched two-hop sequential NM ladder configs."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_DIR = HERE / "configs"
MANIFEST = HERE / "manifest.tsv"
TOTAL_STEPS = 40_000

# graph_id order in ukr_rus_covid_midterm_all8_retweet_graph.pt
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
DATASET_KEYS = [source[1] for source in SOURCES]


def allocate_steps(rung: int, total: int = TOTAL_STEPS) -> list[int]:
    """Equal exposure with the at-most-seven remainder episodes assigned left-to-right."""
    base, remainder = divmod(total, rung)
    return [base + int(index < remainder) for index in range(rung)]


def cumulative(values: list[int]) -> list[int]:
    result = []
    running = 0
    for value in values:
        running += value
        result.append(running)
    return result


def plan() -> list[dict[str, object]]:
    rows = []
    for rung in range(1, len(SOURCES) + 1):
        sources = SOURCES[:rung]
        steps = allocate_steps(rung)
        rows.append(
            {
                "rung": rung,
                "added": sources[-1][0],
                "sources": [source[0] for source in sources],
                "steps": steps,
                "boundaries": cumulative(steps),
                "prefix": f"nm_ladder_seq_h2m_r{rung}",
                "config": f"train_r{rung}.yaml",
            }
        )
    return rows


CONFIG_TEMPLATE = """\
# Compute-matched 2-hop SEQUENTIAL NM ladder -- canonical order, rung {rung}/8.
# Adds {added}; active source order: {sources_pretty}.
# One continuous optimizer sees one contiguous block per source: {schedule_pretty}.
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: ukr_rus_covid_midterm_all8_retweet_graph.pt
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

neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: {source_csv}
neighbor_sampling_source_sequence: {source_csv}
neighbor_sampling_source_sequence_steps: "{step_csv}"
neighbor_sampling_cross_source_prob: 0.0

# 4 x 10k is an honest 40k budget. Explicit checkpoints include every source boundary.
epochs: 4
eval_step: 100000
checkpoint_step: 10000
checkpoint_steps: "{checkpoint_csv}"
workers: 2
device: 0
seed: 0
prefix: {prefix}
"""


SMOKE_CONFIG = """\
# Twenty-step merged-graph smoke for the blocked source scheduler (10 ukr, then 10 covid).
dataset: covid19_twitter
root: /dataMeR1/phil/data/merged/graphs
graph_filename: ukr_rus_covid_midterm_all8_retweet_graph.pt
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
neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: balanced
neighbor_sampling_source_subset: ukr_rus,covid
neighbor_sampling_source_sequence: ukr_rus,covid
neighbor_sampling_source_sequence_steps: "10,10"
neighbor_sampling_cross_source_prob: 0.0
epochs: 1
eval_step: 100000
checkpoint_step: 20
checkpoint_steps: "0,10,20"
workers: 1
device: 0
seed: 0
prefix: nm_ladder_seq_h2m_smoke
"""


def render_config(row: dict[str, object]) -> str:
    source_keys = list(row["sources"])
    steps = list(row["steps"])
    names = {key: canonical for key, _, canonical in SOURCES}
    return CONFIG_TEMPLATE.format(
        rung=row["rung"],
        added=row["added"],
        sources_pretty=" -> ".join(names[key] for key in source_keys),
        schedule_pretty="; ".join(
            f"{names[key]}={step}" for key, step in zip(source_keys, steps)
        ),
        source_csv=",".join(source_keys),
        step_csv=",".join(str(step) for step in steps),
        checkpoint_csv=",".join(
            str(step) for step in [0, *row["boundaries"]]
        ),
        prefix=row["prefix"],
    )


def render_manifest(rows: list[dict[str, object]]) -> str:
    lines = ["rung\tadded\tn_sources\tsources\tblock_steps\tboundaries\tmodel_prefix\tconfig"]
    for row in rows:
        lines.append(
            "\t".join(
                [
                    str(row["rung"]),
                    str(row["added"]),
                    str(len(row["sources"])),
                    ",".join(row["sources"]),
                    ",".join(str(step) for step in row["steps"]),
                    ",".join(str(step) for step in row["boundaries"]),
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
        for row in rows
    }
    files[CONFIG_DIR / "smoke.yaml"] = SMOKE_CONFIG
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
        for line in difflib.unified_diff(
            actual.splitlines(), expected.splitlines(),
            fromfile=str(path), tofile=f"{path} (expected)", lineterm="",
        ):
            print(line)
    if not failed:
        print("OK: 8 sequential rung configs, smoke config, and manifest are current")
    return int(failed)


def write_files(files: dict[Path, str]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    print("wrote 8 sequential rung configs, smoke config, and manifest.tsv")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-configs", action="store_true")
    args = parser.parse_args()
    files = expected_files()
    if args.list_configs:
        for row in plan():
            print(CONFIG_DIR / str(row["config"]))
        return 0
    if args.check:
        return check_files(files)
    write_files(files)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
