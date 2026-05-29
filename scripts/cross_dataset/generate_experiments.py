#!/usr/bin/env python3
"""
Generate cross-dataset leave-k-out experiment scripts.

For each combination of:
  - C(n, k) held-out dataset sets
  - P(n-k, n-k) orderings of training datasets
  - m^(n-k) task assignments (one per training step, drawn from --train-tasks)

Creates a numbered experiment directory with SLURM sbatch scripts and submit wrappers.

Usage:
    python generate_experiments.py \\
        --config  scripts/cross_dataset/experiment_config.yaml \\
        --datasets midterm covid ukr_rus \\
        --train-tasks NM LP \\
        --eval-tasks NM LP PL \\
        --k 1 \\
        --shots 1 5 10 \\
        --output-dir scripts/cross_dataset \\
        --start-id 16 \\
        [--dry-run]

Datasets and tasks must appear in the config file.
"""

import argparse
import itertools
import stat
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    sys.exit("PyYAML is required: pip install pyyaml")


# Fields that appear at fixed positions in the python command; everything
# else in a task config is appended as extra args after the standard block.
_STANDARD_TRAIN_KEYS = frozenset({
    "task_name", "n_way", "n_shots", "n_query", "zero_shot",
    "val_len_cap", "test_len_cap", "epochs", "eval_step", "checkpoint_step",
})
_STANDARD_EVAL_KEYS = frozenset({
    "task_name", "n_way", "n_query", "zero_shot",
    "val_len_cap", "test_len_cap", "epochs", "eval_step", "checkpoint_step",
    "dataset_len_cap",
})

TASK_TAGS = {"NM": "nm", "LP": "lp", "PL": "pl"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def task_tag(task: str) -> str:
    return TASK_TAGS.get(task, task.lower())


def fmt_args(pairs: list[tuple[str, Any]]) -> str:
    """Format a list of (key, value) pairs as aligned --key value \\ lines."""
    lines = []
    for i, (k, v) in enumerate(pairs):
        suffix = " \\" if i < len(pairs) - 1 else ""
        lines.append(f"  --{k:<28} {v}{suffix}")
    return "\n".join(lines)


def set_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def write(path: Path, content: str, executable: bool = False, dry_run: bool = False) -> None:
    if dry_run:
        print(f"    {path.relative_to(path.parent.parent)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    if executable:
        set_executable(path)


# ---------------------------------------------------------------------------
# Script generators
# ---------------------------------------------------------------------------

def _slurm_header(job_name: str, mem: str, time: str, proj: str) -> str:
    return (
        f"#!/bin/bash\n"
        f"#SBATCH --job-name={job_name}\n"
        f"#SBATCH --output={proj}/logs/%x_%j.out\n"
        f"#SBATCH --error={proj}/logs/%x_%j.err\n"
        f"#SBATCH --partition=gpu\n"
        f"#SBATCH --gres=gpu:1\n"
        f"#SBATCH --cpus-per-task=8\n"
        f"#SBATCH --mem={mem}\n"
        f"#SBATCH --time={time}"
    )


def _env_setup(proj: str) -> str:
    return (
        "set -euo pipefail\n\n"
        "module purge || true\n"
        "module load conda\n\n"
        f"cd {proj}\n"
        "source scripts/cross_dataset/env.sh\n"
        'mkdir -p "$LOGS_DIR"'
    )


def _build_train_args(
    ds_cfg: dict,
    task_cfg: dict,
    global_cfg: dict,
    prefix: str,
    is_finetune: bool,
) -> list[tuple[str, Any]]:
    args: list[tuple[str, Any]] = [
        ("dataset",               ds_cfg["dataset_arg"]),
        ("root",                  f'"${ds_cfg["root_var"]}"'),
        ("graph_filename",        f'"${ds_cfg["graph_var"]}"'),
        ("task_name",             task_cfg["task_name"]),
        ("midterm_feature_subset", global_cfg["midterm_feature_subset"]),
    ]
    for k, v in ds_cfg.get("common_args", {}).items():
        args.append((k, v))
    args += [
        ("input_dim",        global_cfg["input_dim"]),
        ("original_features", global_cfg["original_features"]),
        ("n_way",            task_cfg["n_way"]),
        ("n_shots",          task_cfg["n_shots"]),
        ("n_query",          task_cfg["n_query"]),
    ]
    if "zero_shot" in task_cfg:
        args.append(("zero_shot", task_cfg["zero_shot"]))
    args += [
        ("val_len_cap",      task_cfg["val_len_cap"]),
        ("test_len_cap",     task_cfg["test_len_cap"]),
        ("epochs",           task_cfg["epochs"]),
        ("eval_step",        task_cfg["eval_step"]),
        ("checkpoint_step",  task_cfg["checkpoint_step"]),
    ]
    for k, v in task_cfg.items():
        if k not in _STANDARD_TRAIN_KEYS:
            args.append((k, v))
    args += [
        ("workers", ds_cfg["workers_train"]),
        ("device",  global_cfg["device"]),
        ("seed",    global_cfg["seed"]),
    ]
    if is_finetune:
        args.append(("pretrained_model_run", '"$CKPT_PATH"'))
    args.append(("prefix", f'"{prefix}"'))
    return args


def gen_train_sbatch(
    exp_id: int,
    step: int,
    ds: str,
    task: str,
    ds_cfg: dict,
    task_cfg: dict,
    global_cfg: dict,
    prefix: str,
    is_finetune: bool,
) -> str:
    proj = global_cfg["project_root"]
    job  = f"exp{exp_id}_s{step}_{ds}_{task_tag(task)}"
    verb = "Fine-tuning" if is_finetune else "Pretraining"

    header  = _slurm_header(job, ds_cfg["mem_train"], ds_cfg["time_train"], proj)
    setup   = _env_setup(proj)
    ckpt_guard = ': "${CKPT_PATH:?CKPT_PATH is required}"\n\n' if is_finetune else ""
    py_args = _build_train_args(ds_cfg, task_cfg, global_cfg, prefix, is_finetune)
    cmd = (
        "conda run -p \"$CONDA_ENV\" --no-capture-output "
        "python3 experiments/run_single_experiment.py \\\n"
        + fmt_args(py_args)
    )

    return (
        f"{header}\n\n"
        f"{setup}\n\n"
        f"{ckpt_guard}"
        f'echo "=== [Exp{exp_id}] {verb} on {ds} {task} ==="\n'
        f"{cmd}\n"
    )


def gen_train_submit(
    step: int,
    ds: str,
    task: str,
    sbatch_file: str,
    is_finetune: bool,
) -> str:
    verb = "finetune" if is_finetune else "pretrain"
    usage_line = (
        f"# Usage: bash {Path(sbatch_file).stem.replace('sbatch', 'submit')}.sh <ckpt>\n"
        if is_finetune else
        f"# Usage: bash {Path(sbatch_file).stem.replace('sbatch', 'submit')}.sh\n"
    )

    if is_finetune:
        return (
            "#!/bin/bash\n"
            f"{usage_line}"
            "set -euo pipefail\n\n"
            'if [[ $# -ne 1 ]]; then\n'
            f'  echo "Usage: $0 <ckpt>" >&2\n'
            '  exit 1\n'
            'fi\n\n'
            'CKPT_PATH="$1"\n'
            'if [[ ! -f "$CKPT_PATH" ]]; then\n'
            '  echo "Checkpoint not found: $CKPT_PATH" >&2\n'
            '  exit 1\n'
            'fi\n\n'
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n'
            f'echo "Submitting step {step}: {verb} on {ds} {task}..."\n'
            f'sbatch --export=ALL,CKPT_PATH="$CKPT_PATH" "${{SCRIPT_DIR}}/{sbatch_file}"\n\n'
            'echo "Done. Monitor with: squeue -u $USER"\n'
        )
    else:
        return (
            "#!/bin/bash\n"
            f"{usage_line}"
            "set -euo pipefail\n\n"
            'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n\n'
            f'echo "Submitting step {step}: pretrain on {ds} {task}..."\n'
            f'sbatch "${{SCRIPT_DIR}}/{sbatch_file}"\n\n'
            'echo "Done. Monitor with: squeue -u $USER"\n'
        )


def _eval_case_block(task: str, task_cfg: dict) -> str:
    """Generate one case arm inside run_eval()."""
    ttag  = task_tag(task)
    tname = task_cfg["task_name"]

    # Build per-task extra_args list
    arg_pairs: list[tuple[str, Any]] = [
        ("task_name", tname),
        ("n_way",     task_cfg["n_way"]),
        ("n_shots",   '"$shots"'),
        ("n_query",   task_cfg["n_query"]),
        ("zero_shot", '"$([[ "$shots" == "0" ]] && echo True || echo False)"'),
    ]
    if "dataset_len_cap" in task_cfg:
        arg_pairs.append(("dataset_len_cap", task_cfg["dataset_len_cap"]))
    arg_pairs += [
        ("val_len_cap",     task_cfg["val_len_cap"]),
        ("test_len_cap",    task_cfg["test_len_cap"]),
    ]
    if "epochs" in task_cfg:
        arg_pairs.append(("epochs", task_cfg["epochs"]))
    arg_pairs += [
        ("eval_step",       task_cfg["eval_step"]),
        ("checkpoint_step", task_cfg["checkpoint_step"]),
    ]
    for k, v in task_cfg.items():
        if k not in _STANDARD_EVAL_KEYS:
            arg_pairs.append((k, v))
    arg_pairs.append(("prefix", f'"trained_on_${{model_name}}_eval_on_{ttag}_${{shots}}_shot"'))

    extra_args_str = fmt_args(arg_pairs)

    return (
        f"    {tname})\n"
        f"      task_tag={ttag!r}\n"
        f"      extra_args=(\n"
        + "\n".join(f"        {line.strip()}" for line in extra_args_str.splitlines())
        + "\n"
        "      )\n"
        "      ;;\n"
    )


def gen_eval_sbatch(
    ds_cfg: dict,
    eval_tasks: list[str],
    global_cfg: dict,
) -> str:
    proj     = global_cfg["project_root"]
    job      = f"eval_{ds_cfg['dataset_arg']}_all"
    mem      = ds_cfg["mem_eval"]
    time     = ds_cfg["time_eval"]
    workers  = ds_cfg["workers_eval"]

    common_args: list[tuple[str, Any]] = [
        ("dataset",               ds_cfg["dataset_arg"]),
        ("root",                  f'"${ds_cfg["root_var"]}"'),
        ("graph_filename",        f'"${ds_cfg["graph_var"]}"'),
        ("midterm_feature_subset", global_cfg["midterm_feature_subset"]),
        ("input_dim",             global_cfg["input_dim"]),
        ("original_features",     global_cfg["original_features"]),
        ("workers",               workers),
        ("device",                global_cfg["device"]),
        ("seed",                  global_cfg["seed"]),
        ("eval_only",             "True"),
        ("eval_test_before_train", "True"),
        ("eval_val_before_train", "True"),
        ("save_roc_curve",        "True"),
    ]
    # Insert dataset-level common_args (e.g. midterm_edge_view) after root
    extra_ds_args = ds_cfg.get("common_args", {})
    if extra_ds_args:
        insert_at = 3  # after graph_filename
        for k, v in reversed(extra_ds_args.items()):
            common_args.insert(insert_at, (k, v))

    common_block = "\n".join(f"  --{k:<28} {v}" for k, v in common_args)
    case_blocks  = "".join(
        _eval_case_block(t, ds_cfg["eval_tasks"][t])
        for t in eval_tasks
        if t in ds_cfg.get("eval_tasks", {})
    )
    task_calls = "\n".join(
        f'    run_eval "$model_name" "$ckpt_path" {ds_cfg["eval_tasks"][t]["task_name"]} "$shots"'
        for t in eval_tasks
        if t in ds_cfg.get("eval_tasks", {})
    )

    return f"""\
{_slurm_header(job, mem, time, proj)}

{_env_setup(proj)}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: sbatch $0 <model_list.txt> [shots_csv]" >&2
  echo "  <model_list.txt>: each non-comment line is '<model_name> <ckpt_path>'" >&2
  echo "  shots_csv default: 1,5,10" >&2
  exit 1
fi

MODEL_LIST="$1"
SHOTS_CSV="${{2:-1,5,10}}"
if [[ ! -f "$MODEL_LIST" ]]; then
  echo "Model list not found: $MODEL_LIST" >&2
  exit 1
fi

IFS=',' read -r -a SHOT_LIST <<< "$SHOTS_CSV"
[[ ${{#SHOT_LIST[@]}} -eq 0 ]] && {{ echo "No shots in '$SHOTS_CSV'" >&2; exit 1; }}

COMMON_ARGS=(
{common_block}
)

run_eval() {{
  local model_name="$1"
  local ckpt_path="$2"
  local task_name="$3"
  local shots="$4"

  local task_tag="" extra_args=()
  case "$task_name" in
{case_blocks}\
    *)
      echo "Unknown task: $task_name" >&2
      exit 1
      ;;
  esac

  echo "=== Evaluating model=${{model_name}} task=${{task_name}} shots=${{shots}} ==="
  conda run -p "$CONDA_ENV" --no-capture-output \\
    python3 experiments/run_single_experiment.py \\
    "${{COMMON_ARGS[@]}}" \\
    --pretrained_model_run "$ckpt_path" \\
    "${{extra_args[@]}}"
}}

while IFS= read -r raw_line || [[ -n "$raw_line" ]]; do
  line="${{raw_line#"${{raw_line%%[![:space:]]*}}"}}"
  line="${{line%"${{line##*[![:space:]]}}"}}"
  [[ -z "$line" ]] && continue
  [[ "$line" == \\#* ]] && continue

  read -r first second extra <<< "$line"
  if [[ -n "${{second:-}}" ]]; then
    model_name="$first"; ckpt_path="$second"
    [[ -n "${{extra:-}}" ]] && {{ echo "Bad line: '$raw_line'" >&2; exit 1; }}
  else
    ckpt_path="$first"
    model_name="$(basename "$(dirname "$ckpt_path")")"
  fi

  [[ ! -f "$ckpt_path" ]] && {{ echo "Checkpoint not found: $ckpt_path" >&2; exit 1; }}

  for shots in "${{SHOT_LIST[@]}}"; do
{task_calls}
  done
done < "$MODEL_LIST"
"""


def gen_eval_submit(
    exp_id: int,
    eval_step: int,
    eval_ds: str,
    ds_cfg: dict,
    model_name: str,
    sbatch_file: str,
    shots: list[int],
) -> str:
    shots_csv = ",".join(str(s) for s in shots)
    ds_arg    = ds_cfg["dataset_arg"]

    return (
        "#!/bin/bash\n"
        f"# Usage: bash <this_script>.sh <ckpt>\n"
        "set -euo pipefail\n\n"
        'if [[ $# -ne 1 ]]; then\n'
        '  echo "Usage: $0 <ckpt>" >&2\n'
        '  exit 1\n'
        'fi\n\n'
        'CKPT="$1"\n'
        'if [[ ! -f "$CKPT" ]]; then\n'
        '  echo "Checkpoint not found: $CKPT" >&2\n'
        '  exit 1\n'
        'fi\n\n'
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        f'MODEL_LIST="${{SCRIPT_DIR}}/step{eval_step}_eval_{eval_ds}_model_list.txt"\n\n'
        f'cat > "$MODEL_LIST" <<EOF\n'
        f'# Exp {exp_id}: eval on {ds_arg}\n'
        f'{model_name} ${{CKPT}}\n'
        'EOF\n\n'
        f'echo "Submitting eval on {eval_ds} (tasks={", ".join(["NM","LP","PL"])}, shots={shots_csv})..."\n'
        f'sbatch "${{SCRIPT_DIR}}/{sbatch_file}" "$MODEL_LIST" "{shots_csv}"\n\n'
        'echo "Done. Monitor with: squeue -u $USER"\n'
    )


def gen_readme(
    exp_id: int,
    dataset_seq: list[str],
    task_seq: list[str],
    held_out: list[str],
    training_regime: str,
) -> str:
    steps_desc = " → ".join(
        f"{ds} {t}" for ds, t in zip(dataset_seq, task_seq)
    )
    held_desc = ", ".join(held_out)

    pipeline_lines = []
    for i, (ds, t) in enumerate(zip(dataset_seq, task_seq), 1):
        verb = "Pretrain" if i == 1 else "Fine-tune"
        pipeline_lines.append(f"# Step {i} — {verb.lower()} on {ds} {t}")
        if i == 1:
            pipeline_lines.append(f"bash step{i}_submit_{ds}.sh")
        else:
            pipeline_lines.append(f"bash step{i}_submit_{ds}.sh <ckpt_from_step{i-1}>")
        pipeline_lines.append("")
    eval_step = len(dataset_seq) + 1
    for held_ds in held_out:
        pipeline_lines.append(f"# Step {eval_step} — eval on {held_ds} (NM + LP + PL)")
        pipeline_lines.append(f"bash step{eval_step}_submit_eval_{held_ds}.sh <ckpt_from_step{len(dataset_seq)}>")
        pipeline_lines.append("")

    pipeline_str = "\n".join(pipeline_lines).rstrip()

    return (
        f"# Experiment {exp_id}: {steps_desc} → eval {held_desc}\n\n"
        f"## Design\n\n"
        f"```\n"
        f"Train:  {steps_desc}\n"
        f"Eval:   {held_desc} (NM + LP + PL, all shot counts)\n"
        f"Regime: {training_regime}\n"
        f"```\n\n"
        f"## Pipeline\n\n"
        f"```bash\n"
        f"{pipeline_str}\n"
        f"```\n\n"
        f"## Commands Run\n\n"
        f"```bash\n"
        f"# (fill in after execution)\n"
        f"```\n"
    )


# ---------------------------------------------------------------------------
# Experiment directory builder
# ---------------------------------------------------------------------------

def generate_experiment(
    exp_id: int,
    dataset_seq: list[str],
    task_seq: list[str],
    held_out: list[str],
    config: dict,
    eval_tasks: list[str],
    shots: list[int],
    output_dir: Path,
    dry_run: bool,
) -> None:
    global_cfg = config["global"]
    datasets   = config["datasets"]
    exp_dir    = output_dir / f"experiment_{exp_id}"

    # Build a label for the trained model (used in eval model list)
    model_name = "_to_".join(
        f"{ds}_{task_tag(t)}" for ds, t in zip(dataset_seq, task_seq)
    )
    # Training regime label (e.g. NM→LP)
    regime = "→".join(task_tag(t).upper() for t in task_seq)

    # Training steps
    for i, (ds, task) in enumerate(zip(dataset_seq, task_seq), 1):
        is_first    = i == 1
        is_finetune = not is_first
        ds_cfg      = datasets[ds]
        task_cfg    = ds_cfg["train_tasks"][task]
        prefix      = model_name if i == len(dataset_seq) else "_to_".join(
            f"{d}_{task_tag(t)}" for d, t in zip(dataset_seq[:i], task_seq[:i])
        )

        sbatch_fname  = f"step{i}_{'train' if is_first else 'finetune'}_{ds}.sbatch"
        submit_fname  = f"step{i}_submit_{ds}.sh"

        sbatch_content = gen_train_sbatch(
            exp_id, i, ds, task, ds_cfg, task_cfg, global_cfg, prefix, is_finetune
        )
        submit_content = gen_train_submit(
            i, ds, task, sbatch_fname, is_finetune
        )

        write(exp_dir / sbatch_fname,  sbatch_content, dry_run=dry_run)
        write(exp_dir / submit_fname,  submit_content, executable=True, dry_run=dry_run)

    # Eval step(s)
    eval_step = len(dataset_seq) + 1
    for held_ds in held_out:
        ds_cfg       = datasets[held_ds]
        sbatch_fname = f"eval_{ds_cfg['dataset_arg']}_model_list_all_tasks.sbatch"
        submit_fname = f"step{eval_step}_submit_eval_{held_ds}.sh"

        eval_sbatch  = gen_eval_sbatch(ds_cfg, eval_tasks, global_cfg)
        eval_submit  = gen_eval_submit(
            exp_id, eval_step, held_ds, ds_cfg, model_name, sbatch_fname, shots
        )

        write(exp_dir / sbatch_fname, eval_sbatch,  dry_run=dry_run)
        write(exp_dir / submit_fname, eval_submit, executable=True, dry_run=dry_run)

    # README
    readme = gen_readme(exp_id, dataset_seq, task_seq, held_out, regime)
    write(exp_dir / "README.md", readme, dry_run=dry_run)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config",      required=True, help="Path to experiment_config.yaml")
    parser.add_argument("--datasets",    nargs="+", required=True, help="Dataset names (must be in config)")
    parser.add_argument("--train-tasks", nargs="+", required=True, metavar="TASK",
                        help="Tasks available for training steps (e.g. NM LP)")
    parser.add_argument("--eval-tasks",  nargs="+", required=True, metavar="TASK",
                        help="Tasks to evaluate on the held-out set (e.g. NM LP PL)")
    parser.add_argument("--k",           type=int,  required=True,
                        help="Number of datasets to leave out for evaluation")
    parser.add_argument("--shots",       nargs="+", type=int, default=[1, 5, 10],
                        help="Shot counts for eval (default: 1 5 10)")
    parser.add_argument("--output-dir",  default=".", help="Root directory for experiment folders")
    parser.add_argument("--start-id",    type=int,  default=1,
                        help="First experiment number (default: 1)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Print what would be created without writing files")
    args = parser.parse_args()

    config = load_config(args.config)
    output = Path(args.output_dir)

    n = len(args.datasets)
    k = args.k
    if k < 1 or k >= n:
        sys.exit(f"k={k} must be in [1, n-1] where n={n}")

    for ds in args.datasets:
        if ds not in config["datasets"]:
            sys.exit(f"Dataset '{ds}' not found in config. Available: {list(config['datasets'])}")
    for task in args.train_tasks + args.eval_tasks:
        if task not in TASK_TAGS:
            sys.exit(f"Unknown task '{task}'. Known: {list(TASK_TAGS)}")

    # Count total experiments up front
    n_splits       = sum(1 for _ in itertools.combinations(args.datasets, k))
    n_orderings    = 1
    for i in range(1, n - k + 1):
        n_orderings *= i  # (n-k)!
    n_task_combos  = len(args.train_tasks) ** (n - k)
    total          = n_splits * n_orderings * n_task_combos

    print(f"Generating {total} experiments  "
          f"(C({n},{k})={n_splits} splits × {n_orderings} orderings × "
          f"{len(args.train_tasks)}^{n-k}={n_task_combos} task combos)")
    print(f"  train-tasks : {args.train_tasks}")
    print(f"  eval-tasks  : {args.eval_tasks}")
    print(f"  shots       : {args.shots}")
    print(f"  output-dir  : {output}")
    print(f"  start-id    : {args.start_id}")
    if args.dry_run:
        print("  [DRY RUN — no files written]")
    print()

    exp_id = args.start_id
    for held_out in itertools.combinations(args.datasets, k):
        train_datasets = [d for d in args.datasets if d not in set(held_out)]
        for dataset_seq in itertools.permutations(train_datasets):
            for task_seq in itertools.product(args.train_tasks, repeat=len(dataset_seq)):
                regime  = "→".join(task_tag(t).upper() for t in task_seq)
                summary = " → ".join(
                    f"{ds}/{t}" for ds, t in zip(dataset_seq, task_seq)
                )
                print(f"  Exp {exp_id:>4}  [{regime}]  {summary}  →  eval {', '.join(held_out)}")
                if args.dry_run:
                    exp_dir = output / f"experiment_{exp_id}"
                    print(f"    {exp_dir}/")
                generate_experiment(
                    exp_id=exp_id,
                    dataset_seq=list(dataset_seq),
                    task_seq=list(task_seq),
                    held_out=list(held_out),
                    config=config,
                    eval_tasks=args.eval_tasks,
                    shots=args.shots,
                    output_dir=output,
                    dry_run=args.dry_run,
                )
                exp_id += 1

    print(f"\nDone. Created experiments {args.start_id}–{exp_id - 1}.")


if __name__ == "__main__":
    main()
