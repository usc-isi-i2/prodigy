import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

import aggregate_center_disjoint as aggregate  # noqa: E402
from protocol import INDUCED_PROTOCOL, PROTOCOL, TARGETS  # noqa: E402


def write_grid(clean_root: Path, original_root: Path) -> None:
    for seed in (0, 1, 2):
        for source_index, source in enumerate(TARGETS):
            model_id = f"ss_{source}"
            for target_index, target in enumerate(TARGETS):
                if source == target:
                    continue
                plan = f"{target_index + 1:064x}"
                observed = f"{target_index + 101:064x}"
                clean_path = clean_root / f"seed_{seed}" / model_id / f"{target}.json"
                original_path = original_root / f"seed_{seed}" / model_id / f"{target}.json"
                clean_path.parent.mkdir(parents=True, exist_ok=True)
                original_path.parent.mkdir(parents=True, exist_ok=True)
                original_score = 0.2 + 0.01 * source_index + 0.001 * seed
                original_path.write_text(json.dumps({
                    "episode_plan_fingerprint": plan,
                    "score": original_score,
                }))
                clean_path.write_text(json.dumps({
                    "protocol": PROTOCOL,
                    "seed": seed,
                    "model_id": model_id,
                    "target": target,
                    "episode_count": 512,
                    "exclusion_level": "episode_centers",
                    "unfiltered_prefix_plan_fingerprint": plan,
                    "episode_plan_fingerprint": f"{target_index + 11:064x}",
                    "observed_episode_fingerprint": observed,
                    "score": original_score + 0.02,
                    "score_std": 0.01,
                    "loss": 1.0,
                    "aux_loss": 0.0,
                    "excluded_node_count": 10,
                    "target_graph_nodes": 100,
                    "sampled_context_node_occurrences": 1000,
                    "sampled_context_overlap_occurrences": 20,
                    "sampled_context_unique_overlap_nodes": 5,
                }))


def test_aggregate_validates_and_emits_18_cells(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    original = tmp_path / "original"
    output = tmp_path / "output"
    write_grid(clean, original)
    monkeypatch.setattr(sys, "argv", [
        "aggregate_center_disjoint.py",
        "--results-root", str(clean),
        "--original-results-root", str(original),
        "--output-root", str(output),
    ])
    assert aggregate.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["cells"] == 18
    assert summary["directions"] == 6
    assert abs(summary["delta_mean"] - 0.02) < 1e-12
    assert len((output / "paired_cells.tsv").read_text().splitlines()) == 19


def test_induced_aggregate_requires_zero_context_overlap(tmp_path, monkeypatch):
    clean = tmp_path / "clean"
    original = tmp_path / "original"
    output = tmp_path / "output"
    write_grid(clean, original)
    for path in clean.glob("seed_*/*/*.json"):
        payload = json.loads(path.read_text())
        payload.update({
            "protocol": INDUCED_PROTOCOL,
            "exclusion_level": "induced_subgraph",
            "sampled_context_overlap_occurrences": 0,
            "sampled_context_unique_overlap_nodes": 0,
        })
        path.write_text(json.dumps(payload))
    monkeypatch.setattr(sys, "argv", [
        "aggregate_center_disjoint.py",
        "--variant", "induced",
        "--results-root", str(clean),
        "--original-results-root", str(original),
        "--output-root", str(output),
    ])
    assert aggregate.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["variant"] == "induced"
    assert abs(summary["induced_mean"] - summary["original_mean"] - 0.02) < 1e-12
    assert all(
        item["overlap_occurrences"] == 0
        for item in summary["residual_sampled_context_overlap"].values()
    )
