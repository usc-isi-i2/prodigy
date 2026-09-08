# Social checkpoints on the FB15K-237 downstream task

This adapter evaluates four existing seed-0, step-2,500 social-graph PRODIGY
specialists on the original paper's FB15K-237 20-way, 3-shot relation-type
classification task. The selected sources are Ukraine/Russia, COVID, TwiBot-20,
and Midterm.

This is not a native-paper architecture comparison. The social checkpoints use
`S,U,M` with 768-dimensional node features. Native Wiki-to-FB15K-237 PRODIGY
uses `S2,UX,M2`, 770-dimensional node inputs with head/tail flags, and KG edge
features. The explicit `--kg_social_checkpoint_adapter` keeps the social model
unchanged by omitting the two endpoint flags and KG edge-feature modules. FB15K
episodes, labels, support/query counts, and relation-classification objective are
otherwise native.

Run on Tucker in a dedicated worktree and tmux session. The launcher uses owned
GPUs 0-3, one checkpoint per GPU, and writes only beneath
`log/social_to_fb15k_adapter/`.

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
bash scripts/experiments/setup/social_to_fb15k_adapter/run_tucker.sh
```
