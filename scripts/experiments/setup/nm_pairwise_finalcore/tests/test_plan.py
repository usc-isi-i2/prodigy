from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
SETUP = HERE.parent
sys.path.insert(0, str(SETUP))

from pair_plan import CHECKPOINT_STEP, SOURCES, build_models, checkpoint_path, physical_jobs


def simple_yaml(path: Path) -> dict[str, str]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip('"')
    return values


def test_plan_is_every_unordered_pair_at_seed_zero():
    models = build_models()
    assert len(models) == 36
    assert len({frozenset(model.sources) for model in models}) == 36
    appearances = Counter(source for model in models for source in model.sources)
    assert appearances == Counter({source: 8 for source in SOURCES})
    jobs = physical_jobs()
    assert len(jobs) == 36
    assert {job.seed for job in jobs} == {0}


def test_generated_configs_preserve_final_core_contract():
    configs = sorted((SETUP / "configs").glob("*.yaml"))
    assert len(configs) == 36
    expected = {model.model_id: model.sources for model in build_models()}
    seen = set()
    for path in configs:
        values = simple_yaml(path)
        model_id = values["prefix"]
        assert model_id in expected
        assert values["neighbor_sampling_source_subset"] == ",".join(expected[model_id])
        assert values["seed"] == "0"
        assert values["batch_size"] == "4"
        assert values["dataset_len_cap"] == "2500"
        assert values["neighbor_sampling_episode_source"] == "graph_id"
        assert values["neighbor_sampling_episode_source_weighting"] == "balanced"
        assert values["edge_view"] == "static_train"
        assert values["target_edge_view"] == "static_test"
        seen.add(model_id)
    assert seen == set(expected)


def test_shared_manifest_resolves_every_checkpoint(tmp_path):
    jobs = physical_jobs()
    manifest_jobs = []
    for index, job in enumerate(jobs):
        exp_name = f"{job.model.model_id}_run_{index:03d}"
        manifest_jobs.append({"prefix": job.model.model_id, "exp_name": exp_name})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"jobs": manifest_jobs}), encoding="utf-8"
    )
    for index, job in enumerate(jobs):
        assert checkpoint_path(tmp_path, job, "ignored") == (
            tmp_path / "state" / f"{job.model.model_id}_run_{index:03d}"
            / "checkpoint" / f"state_dict_{CHECKPOINT_STEP}.ckpt"
        )
