#!/usr/bin/env python3
"""Build a unified inventory of repository experiments and W&B runs."""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
SETUP_ROOT = ROOT / "scripts" / "experiments" / "setup"
ANALYSIS_ROOT = ROOT / "scripts" / "experiments" / "analysis"
WANDB_INPUT = ROOT / "wandb_exports" / "graph_clip_run_metadata.csv"
OUTPUT = Path(__file__).resolve().parent / "EXPERIMENT_INVENTORY.md"


def category(name: str) -> str:
    """Assign the agreed top-level experiment category."""
    value = name.lower()
    if any(token in value for token in ["transfer_matrix", "covid_midterm", "task_transfer", "final_core"]):
        return "Ablations"
    if "saturation" in value or "radius" in value or "exposure" in value:
        return "Saturation"
    if any(token in value for token in ["ablation", "shortcut", "disjoint", "nhop", "sequential", "gatv2", "source_complete", "train_test", "sampling"]):
        return "Ablations"
    if "ladder" in value or "weak_to_strong" in value:
        return "Ladders"
    if any(token in value for token in ["graph_divergence", "identity_overlap", "bio_embedding", "path_feature", "tag_citation", "similarity_vs_transfer"]):
        return "Graph/data diagnostics"
    if any(token in value for token in ["slp_evaluator", "error_audit", "regression_probe_repair", "static_link_prediction", "node_classification", "node_regression"]):
        return "Methods/validation"
    if any(token in value for token in ["multitask_ssl", "pretrain", "topology_feature", "covid_only"]):
        return "Objectives/pretext"
    if any(token in value for token in ["single_source", "icl_arch_matrix"]):
        return "Matrices"
    return "Other/legacy"


def family(name: str) -> str:
    rules = [
        (r"^nm_", "PRODIGY / neighbor matching"),
        (r"^samgpt_", "SAMGPT / GraphCL"),
        (r"^multitask_ssl", "multitask SSL"),
        (r"^pretrain_", "pretraining / probes"),
        (r"^topology_", "topology and features"),
        (r"^similarity_", "transfer prediction"),
        (r"^covid", "COVID studies"),
        (r"^train[123]$", "legacy training batches"),
        (r"^icl_", "architecture comparison"),
        (r"^.*(slp|link_prediction).*", "link prediction"),
        (r"^.*regression.*", "regression"),
    ]
    for pattern, label in rules:
        if re.search(pattern, name, re.IGNORECASE):
            return label
    return "other"


def setup_rows():
    rows = []
    for path in sorted(p for p in SETUP_ROOT.iterdir() if p.is_dir()):
        files = [p for p in path.rglob("*") if p.is_file()]
        configs = sum(p.suffix in {".yaml", ".yml", ".json"} for p in files)
        runners = sum(p.suffix in {".sh", ".sbatch", ".py"} for p in files)
        readme = (path / "README.md").exists()
        matching_analysis = []
        for candidate in ANALYSIS_ROOT.rglob("*"):
            if candidate.is_dir() and candidate.name == path.name:
                matching_analysis.append(str(candidate.relative_to(ROOT)))
        rows.append({
            "name": path.name,
            "category": category(path.name),
            "family": family(path.name),
            "path": str(path.relative_to(ROOT)),
            "readme": "yes" if readme else "no",
            "configs": str(configs),
            "runners": str(runners),
            "analysis": " || ".join(matching_analysis),
        })
    return rows


def analysis_only_rows(setups):
    setup_names = {row["name"] for row in setups}
    rows = []
    pattern = re.compile(r"\[`([^`]+)`\]\(([^)]+)\)")
    readme = (ANALYSIS_ROOT / "README.md").read_text(encoding="utf-8")
    for name, path in pattern.findall(readme):
        leaf = Path(path).name
        if leaf in setup_names or leaf in {"archive", "cross_experiment"}:
            continue
        rows.append({"name": leaf, "category": category(leaf), "path": str((ANALYSIS_ROOT / path).resolve().relative_to(ROOT))})
    return rows


def wandb_rows():
    if not WANDB_INPUT.exists():
        return []
    with WANDB_INPUT.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    setups = setup_rows()
    analysis_only = analysis_only_rows(setups)
    runs = wandb_rows()
    lines = [
        "# Unified experiment inventory",
        "",
        "This is the broad experiment archive, combining repository-defined experiment setups, analysis-only studies, and all exported W&B runs.",
        "It is intentionally broader than the evaluation ledger, which contains metric rows rather than experiment records.",
        "",
        "## Inventory counts",
        "",
        f"- Repository setup experiments: **{len(setups):,}**",
        f"- Analysis-only or differently named studies: **{len(analysis_only):,}**",
        f"- W&B runs: **{len(runs):,}**",
        f"- Total inventory records: **{len(setups) + len(analysis_only) + len(runs):,}**",
        "- Repository analysis index: `scripts/experiments/analysis/README.md`",
        "",
        "## Repository experiment setups",
        "",
        "Each row is a setup directory. The analysis column points to a same-named analysis directory when one exists.",
        "",
        "| Experiment | Category | Family | Setup path | README | Config files | Runner/code files | Matching analysis |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in setups:
        analysis = row["analysis"] or ""
        lines.append(
            f"| `{row['name']}` | {row['category']} | {row['family']} | `{row['path']}` | {row['readme']} | "
            f"{row['configs']} | {row['runners']} | `{analysis}` |"
        )

    lines.extend([
        "",
        "## Analysis-only or differently named studies",
        "",
        "These are listed in the analysis index but do not have a same-named current setup directory.",
        "",
        "| Study | Category | Analysis path |",
        "|---|---|---|",
    ])
    for row in sorted(analysis_only, key=lambda item: (item["category"], item["name"])):
        lines.append(f"| `{row['name']}` | {row['category']} | `{row['path']}` |")

    lines.extend([
        "",
        "## W&B run archive",
        "",
        "Each row is one W&B run from `eibl-usc/graph-clip`. Timestamps are UTC.",
        "",
        "| Created (UTC) | Category | Status | Run name | W&B ID | Link |",
        "|---|---|---|---|---|---|",
    ])
    runs.sort(key=lambda row: (row.get("created_at", ""), row.get("run_id", "")), reverse=True)
    for row in runs:
        name = (row.get("run_name", "") or "(unnamed)").replace("|", "\\|")
        run_id = row.get("run_id", "")
        url = row.get("url", "") or f"https://wandb.ai/eibl-usc/graph-clip/runs/{run_id}"
        lines.append(f"| {row.get('created_at','')} | {category(name)} | {row.get('state','')} | {name} | `{run_id}` | [open]({url}) |")

    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"setup_experiments={len(setups)} wandb_runs={len(runs)} output={OUTPUT}")


if __name__ == "__main__":
    main()
