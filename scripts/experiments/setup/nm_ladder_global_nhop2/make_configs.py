#!/usr/bin/env python3
"""Generate the source-unaware Order-A ladder on the final-core contract."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
FINAL_CORE = HERE.parent / "final_core"
sys.path.insert(0, str(FINAL_CORE))

from fixed_test_plan import model_for_ladder, physical_jobs  # noqa: E402


CONFIG_DIR = HERE / "configs"
MANIFEST = HERE / "manifest.tsv"
ORDER_A = (
    "ukr_rus", "covid", "midterm", "covid_political", "election2020",
    "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference",
)


def rows() -> list[dict[str, object]]:
    jobs = physical_jobs()
    result = []
    for rung in range(2, 10):
        model = model_for_ladder("A", rung)
        job_index = next(
            index for index, job in enumerate(jobs)
            if job.seed == 0 and job.model.model_id == model.model_id
        )
        result.append({
            "rung": rung,
            "model_id": model.model_id,
            "sources": model.sources,
            "newcomer": ORDER_A[rung - 1],
            "job_index": job_index,
        })
    return result


CONFIG_TEMPLATE = """\
# Final-core source-unaware Order-A ladder, rung {rung}/9.
# Active source union: {sources}.
# p=1 makes every episode node-uniform over this union and permits mixed-source labels.
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
batch_size: 4
learning_rate: 0.002
weight_decay: 0.001
dataset_len_cap: 2500
val_len_cap: 500
test_len_cap: 500

# Resolve the active rung on the immutable all-nine artifact, then take the p=1
# node-uniform mixed-source branch instead of the interleaved within-source branch.
neighbor_sampling_episode_source: graph_id
neighbor_sampling_episode_source_weighting: proportional
neighbor_sampling_source_subset: {sources}
neighbor_sampling_batch_source_mode: independent
neighbor_sampling_cross_source_prob: 1.0

epochs: 1
eval_step: 100000
checkpoint_step: 100000
checkpoint_steps: "100,300,900,2500"
print_step: 100
workers: 2
device: 0
seed: 0
tags: [final_core, global_merge_ladder, order_A, seed0, rung_{rung}]
prefix: finalcore_global_ordA_r{rung}
"""


def render_config(row: dict[str, object]) -> str:
    return CONFIG_TEMPLATE.format(
        rung=row["rung"], sources=",".join(str(value) for value in row["sources"])
    )


def expected_files() -> dict[Path, str]:
    planned = rows()
    files = {
        CONFIG_DIR / f"train_r{row['rung']}.yaml": render_config(row)
        for row in planned
    }
    smoke = render_config(planned[0]).replace(
        "dataset_len_cap: 2500", "dataset_len_cap: 5"
    ).replace(
        'checkpoint_steps: "100,300,900,2500"', 'checkpoint_steps: "5"'
    ).replace("workers: 2", "workers: 0").replace(
        "prefix: finalcore_global_ordA_r2", "prefix: finalcore_global_smoke_r2"
    )
    files[CONFIG_DIR / "smoke.yaml"] = smoke
    lines = ["rung\tmodel_id\tjob_index\tnewcomer\tsources\tconfig"]
    lines.extend(
        "\t".join([
            str(row["rung"]), str(row["model_id"]), str(row["job_index"]),
            str(row["newcomer"]), ",".join(str(value) for value in row["sources"]),
            f"configs/train_r{row['rung']}.yaml",
        ])
        for row in planned
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
        print("OK: eight final-core global-merge rungs, smoke, and manifest are current")
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = expected_files()
    if args.check:
        return check(files)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.write_text(content, encoding="utf-8")
    print("wrote eight final-core global-merge rung configs, smoke, and manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
