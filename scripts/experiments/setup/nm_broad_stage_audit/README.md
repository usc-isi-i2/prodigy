# Four-graph canonical NM stage audit

Replay the seed-0 HK, Ukraine, COVID, and Midterm specialists on the published
fixed 512-episode streams for those four targets. The run records pre-metagraph
mean-support cosine, final logits, compact episode inputs, and per-query ranks.
It performs no training or new experimental sampling: episode plans are rebuilt
from the deterministic fixed-test protocol and must match the published raw and
observed fingerprints.

The published final-core stream uses randomized member assignment. This differs
from the later detailed HK/Ukraine canonical audit's lowest-sorted member policy;
the two protocols must be reported separately.

The process loads the all-nine merged graph once, runs on one GPU, and releases
each target before moving to the next. Outputs are private Tucker artifacts;
only compact aggregate findings belong in git.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
python scripts/experiments/setup/nm_broad_stage_audit/run.py --self-test
python scripts/experiments/setup/nm_broad_stage_audit/run.py --dry-run
python -u scripts/experiments/setup/nm_broad_stage_audit/run.py \
  --device cuda:0 \
  --episode-plan-root /dataMeR1/phil/gfm/final_core_episode_plans_045ba527_missing \
  --out /dataMeR1/phil/gfm/error_audit/nm_broad_stage_20260909
```

Use a new output directory and an isolated worktree. All checkpoint, reference,
device, target, and model paths are overrideable.
