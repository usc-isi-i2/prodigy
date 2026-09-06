"""Independent effective-config and declared episode-source audit; no outcomes."""
import argparse
import json
from pathlib import Path

import yaml


REVISION = "75f0853f96272120a1e295dfbd38c04f2b71fe62"
VARIED = {"config", "prefix", "exp_name", "timestamp", "seed", "neighbor_sampling_source_subset",
          "neighbor_matching_member_policy", "neighbor_matching_member_seed"}


def reconstruct(base, delta):
    result = {k: v for k, v in base.items() if k not in delta["removed"]}
    return {**result, **delta["changes"]}


def check_protocol(inventory, declared, source_protocol):
    manifest = inventory["manifest"]
    if manifest["revision"] != REVISION or manifest["mode"] != "training" or manifest["device"] != "cpu":
        raise ValueError("wrong substantive launch revision/mode/device")
    base = inventory["base_effective_config"]
    actual = [reconstruct(base, d) for d in inventory["effective_configs"]]
    planned = [reconstruct(base, d) for d in inventory["planned_jobs"]]
    if len(actual) != 24 or len(planned) != 24 or len(declared) != 24:
        raise ValueError("incomplete declared/effective configuration inventory")
    if {p["prefix"] for p in actual} != set(declared):
        raise ValueError("declared model inventory differs")
    common = {k: v for k, v in actual[0].items() if k not in VARIED}
    for p, launch in zip(actual, planned):
        expected = {**declared[p["prefix"]], "device": "cpu"}
        if any(p.get(k) != v for k, v in expected.items()):
            raise ValueError("effective configuration differs from declared CPU recipe")
        if {k: v for k, v in p.items() if k not in VARIED} != common:
            raise ValueError("non-treatment training parameters differ between models")
        if {k: v for k, v in p.items() if k not in {"exp_name", "timestamp"}} != {
                k: v for k, v in launch.items() if k not in {"exp_name", "timestamp"}}:
            raise ValueError("effective configuration differs from launch manifest")
        if str(Path(p["root"]) / p["graph_filename"]) != source_protocol["graph_path"]:
            raise ValueError("source-name metadata describes a different graph")
        if p["pretrained_model_run"] or p["resume_training_checkpoint"]:
            raise ValueError("unexpected pretraining or resume")
    return actual


def check_episode_sources(arms, actual, source_protocol):
    settings = {p["prefix"]: p for p in actual}
    names = source_protocol["source_names"]
    if len(names) != len(set(names)) or len(arms) != 24 or {r["model_id"] for r in arms} != set(settings):
        raise ValueError("source map or completed arm inventory differs")
    for row in arms:
        p = settings[row["model_id"]]
        source = p["neighbor_sampling_source_subset"]
        source_id = names.index(source)
        summary = row["summary"]
        count = p["epochs"] * p["dataset_len_cap"] * p["batch_size"]
        expected = {str(source_id): count}
        if row["source"] != source or {str(k): v for k, v in summary["source_ids"].items()} != expected:
            raise ValueError("consumed episode source labels differ from intended source")
        if summary["episodes"] != count or row["seed"] != p["seed"] or row["policy"] != p["neighbor_matching_member_policy"]:
            raise ValueError("consumed arm metadata differs from effective configuration")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--config-only", action="store_true", help="Pre-completion check; does not publish a final receipt")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[6]
    folder = repo / "scripts/experiments/setup/target_performance_mechanisms/member_configs"
    declared = {}
    for path in sorted(folder.glob("*.yaml")):
        p = yaml.safe_load(path.read_text())
        if p["prefix"] in declared:
            raise ValueError("duplicate declared configuration")
        declared[p["prefix"]] = p
    inventory = json.loads((args.data / "member_training_config_inventory.json").read_text())
    source_protocol = json.loads((args.data / "source_sampler_protocol.json").read_text())
    actual = check_protocol(inventory, declared, source_protocol)
    if args.config_only:
        print("All 24 effective configurations match their declared CPU recipe and common non-treatment settings.")
        return
    arms = json.loads((args.data / "member_training_verified/arms.json").read_text())
    check_episode_sources(arms, actual, source_protocol)
    result = {"models": 24, "revision": REVISION, "declared_cpu_recipe_matches": True,
              "common_non_treatment_settings": True, "consumed_episode_source_labels_match": True,
              "source_map_from_exact_graph_audit": True,
              "independent_graph_id_lookup_of_each_member": False}
    (args.data / "member_training_contract_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
