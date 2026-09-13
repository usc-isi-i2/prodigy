---
name: prodigy-tucker
description: Operate PRODIGY training, evaluation, graph construction, embedding generation, checkpoints, logs, worktrees, GPUs, or tmux sessions on the Tucker cluster. Use for Tucker and /dataMeR1 work; do not use for ordinary local-only code edits or plots.
---

# PRODIGY on Tucker

Use Tucker for training, evaluation, graph construction, embedding generation, and
other GPU-heavy work. Keep local work to editing, notebooks, plots, and lightweight
checks.

## Before acting

1. Establish the exact Tucker worktree, branch, run name, checkpoint, log directory,
   and GPU involved. Do not infer them from the local checkout.
2. Inspect tmux sessions and relevant process/GPU state before changing a Tucker
   worktree or scheduling work.
3. For anything beyond read-only inspection, read
   [Tucker operations](../../../docs/agent_guides/tucker.md).
4. For high-throughput neighbor-matching training, also read
   [fast training](../../../docs/fast_training.md).
5. For evaluation or result parsing, read
   [repository traps](../../../docs/agent_guides/repo_traps.md).

## Operating constraints

- Use only GPUs 0-3; leave GPUs 4-7 untouched.
- Give heavy or long runs their own Git worktree and tmux session.
- Never pull or switch revisions in a worktree that has a running job.
- Always pull an explicitly named branch on Tucker. Do not trust configured upstreams.
- Treat `state/` and `log/` as worktree-local, gitignored state. Evaluate from the
  training worktree or pass absolute paths.
- The user generally launches large or long-running jobs. Do not launch one unless
  the user explicitly asks; prepare and validate commands when launch is not authorized.
- Reading files, logs, graph metadata, and process state on Tucker is allowed.

## Workflow

- Reuse the experiment's checked-in launcher/config and the shared harness.
- Run an available dry-run mode before a large sweep.
- Activate the documented conda environment inside the actual shell or tmux command
  that launches Python.
- Label smoke runs as smoke validation, never as completed experimental evidence.
- After execution, verify the session/process, log creation, selected device, revision,
  and expected output path. Report the Tucker worktree and branch.
