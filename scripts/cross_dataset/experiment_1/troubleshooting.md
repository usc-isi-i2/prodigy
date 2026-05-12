# Troubleshooting: Cluster Setup (discovery1)

## 1. conda: command not found

**Error:**
```
/var/spool/slurm/d/job.../slurm_script: line 14: conda: command not found
/var/spool/slurm/d/job.../slurm_script: line 14: /etc/profile.d/conda.sh: No such file or directory
```

**Cause:** Conda is not in the default PATH for batch jobs. The cluster uses a module system.

**Fix:** Replace `source "$(conda info --base)/etc/profile.d/conda.sh"` with `module load conda` in sbatch scripts.

---

## 2. CondaError: Run 'conda init' before 'conda activate'

**Error:**
```
CondaError: Run 'conda init' before 'conda activate'
```

**Cause:** Batch jobs don't source `.bashrc`, so conda shell hooks are not initialized.

**Fix:** Use `conda run` instead of `conda activate`:
```bash
conda run -p /home1/singhama/.conda/envs/prodigy-env --no-capture-output python3 ...
```

---

## 3. conda env create fails on PyG pip packages

**Error:**
```
ERROR: Could not find a version that satisfies the requirement torch-cluster==1.6.3+pt20cu117
ERROR: No matching distribution found for torch-cluster==1.6.3+pt20cu117
CondaEnvException: Pip failed
```

**Cause:** PyG packages with `+pt20cu117` build tags are not on PyPI — they are only available from PyG's own wheel server.

**Fix:** Create the env (conda packages will succeed), then install torch and PyG manually:
```bash
# Torch (CUDA 11.7 — backward compatible with cluster's CUDA 12.9 driver)
conda run -p /home1/singhama/.conda/envs/prodigy-env pip install \
  torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2+cu117 \
  --index-url https://download.pytorch.org/whl/cu117

# PyG packages from PyG wheel server
conda run -p /home1/singhama/.conda/envs/prodigy-env pip install \
  torch-scatter==2.1.2 torch-sparse==0.6.18 torch-cluster==1.6.3 \
  torch-spline-conv==1.2.2 torch-geometric==2.3.1 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu117.html

# Remaining packages
conda run -p /home1/singhama/.conda/envs/prodigy-env pip install \
  wandb==0.24.2 sentence-transformers==2.2.2 transformers==4.29.2 \
  scikit-learn==1.7.2 ogb==1.3.6 numpy==1.26.4 pandas==2.3.3 \
  tqdm==4.67.3 networkx==3.4.2 lmdb==1.7.5 ray==2.53.0
```

**Note:** Run installs on an interactive GPU node, not the login node:
```bash
srun --partition=gpu --gres=gpu:1 --mem=32G --cpus-per-task=4 --pty bash
```

---

## 4. ImportError: cannot import name 'cached_download' from 'huggingface_hub'

**Error:**
```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**Cause:** Installing `sentence-transformers==2.2.2` upgraded `huggingface_hub` to a version that removed `cached_download`.

**Fix:** Pin `huggingface_hub` back to the version in the original environment:
```bash
conda run -p /home1/singhama/.conda/envs/prodigy-env pip install huggingface-hub==0.14.1
```

---

## 5. wandb login fails interactively via conda run

**Error:**
```
wandb: ERROR Find detailed error logs at: /tmp/SLURM_.../debug-cli.singhama.log
Error: No API key configured. Use `wandb login` to log in.
ERROR conda.cli.main_run:execute(127): `conda run wandb login` failed.
```

**Cause:** `conda run` does not support interactive prompts.

**Fix:** Call the wandb binary directly:
```bash
/home1/singhama/.conda/envs/prodigy-env/bin/wandb login
```

Find your API key at: https://wandb.ai/authorize
