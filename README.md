# PRODIGY: Enabling In-context Learning Over Graphs

A pretraining framework for few-shot, in-context learning over graphs — pretrain a GNN once, then adapt to diverse tasks on unseen graphs without parameter updates.

Paper: [PRODIGY: Enabling In-context Learning Over Graphs](https://arxiv.org/abs/2305.12600) (SPIGM @ ICML 2023)

![In-context few-shot prompting over graphs with prompt graph for edge classification in PRODIGY.](fig.png)

---

## Setup

### On the cluster (GPU / Linux)

```bash
conda env create -f environment.yml
conda activate prodigy
```

> Note: `environment.yml` pins `torch==2.0.1` with CUDA 11.7 wheels. Remove the `prefix:` line at the bottom before creating the environment on a new machine.

### Local / CPU (macOS or Linux without GPU)

```bash
pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.0.1+cpu.html
```

Known dependency pin: if `sentence-transformers==2.2.2` fails with a `cached_download` error, pin `huggingface_hub==0.14.1`.

### Sanity check

Run a tiny arXiv experiment on CPU to verify the install (downloads data automatically):

```bash
WANDB_MODE=offline python experiments/run_single_experiment.py \
  --dataset arxiv \
  --root <DATA_ROOT> \
  --original_features True \
  --input_dim 128 \
  -ds_cap 10 -val_cap 5 -test_cap 5 \
  --epochs 1 -eval_step 5 -ckpt_step 5 \
  -task classification \
  -bs 1 -way 3 -shot 1 -qry 1 \
  --device cpu \
  --prefix SANITY_ARXIV
```

---

## Datasets

### Twitter datasets (this repo)

Three Twitter retweet graphs are supported, each with its own data-preparation pipeline. See the dataset README for step-by-step instructions including how to build embeddings, build the graph, validate, and run experiments.

| Dataset | README | Raw format | Raw data location (cluster) |
|---|---|---|---|
| Midterm 2022 | [data/data/midterm/README.md](data/data/midterm/README.md) | CSV | `/project2/ll_774_951/midterm/*/*.csv` |
| Ukraine-Russia Twitter | [data/data/ukr_rus_twitter/README.md](data/data/ukr_rus_twitter/README.md) | CSV | `/project2/ll_774_951/uk_ru/twitter/data/*/*.csv` |
| COVID-19 Twitter | [data/data/covid19_twitter/README.md](data/data/covid19_twitter/README.md) | JSON | `/scratch1/eibl/data/covid19_twitter/raw/*/*.json` |

All three share the same graph schema (node features, edge features, temporal splits) and can be used interchangeably as `--dataset midterm`, `--dataset ukr_rus_twitter`, or `--dataset covid19_twitter`.

### Original PRODIGY datasets (arXiv, MAG240M, KG)

For arXiv and MAG240M the data is downloaded automatically. For knowledge graphs:

- Download preprocessed [Wiki](http://snap.stanford.edu/prodigy/Wiki.zip) and [FB15K-237](http://snap.stanford.edu/prodigy/FB15K-237.zip) to `<DATA_ROOT>`.
- NELL and ConceptNet: follow links in [CSR](https://github.com/snap-stanford/csr).
- MAG240M adjacency matrix (if memory issues during processing): download from [here](http://snap.stanford.edu/prodigy/mag240m_adj_bi.pt) and place under `<DATA_ROOT>/mag240m`.

---

## Running experiments

Checkpoints are saved to `state/<PREFIX>_<timestamp>/checkpoint/` and logged to W&B.

### Twitter datasets

Training scripts are in `scripts/`. Each is a self-contained SLURM `.sbatch` script — submit from the repo root:

```bash
# Midterm
sbatch scripts/submit_train1_midterm_lp.sh   # temporal link prediction
sbatch scripts/submit_train1_midterm_nm.sh   # neighbor matching
sbatch scripts/submit_train1_midterm_pl.sh   # political leaning classification

# Ukraine-Russia
sbatch scripts/submit_train1_ukr_rus_twitter_lp.sh
sbatch scripts/submit_train1_ukr_rus_twitter_nm.sh
sbatch scripts/submit_train1_ukr_rus_twitter_pl.sh

# COVID-19
sbatch scripts/submit_train1_covid19_twitter_lp.sh
sbatch scripts/submit_train1_covid19_twitter_nm.sh
```

> The scripts have hardcoded paths under `/scratch1/eibl/` and `/home1/eibl/`. Update `--root`, `--graph_filename`, and the `cd` / log paths to your own directories before submitting.

Cross-dataset transfer (train on one, eval on other) — see [CROSS_DATASET_EVAL.md](CROSS_DATASET_EVAL.md) for the full flow:

```bash
# Midterm <-> Ukraine-Russia
sbatch scripts/submit_eval_midterm_to_ukr_rus_all_tasks.sh
sbatch scripts/submit_eval_ukr_rus_to_midterm_all_tasks.sh

# Evaluate any model list on a COVID-19 target graph
sbatch scripts/eval_covid19_twitter_model_list_all_tasks.sbatch \
  scripts/eval1_covid_model_list.txt

# Evaluate any model list on a Ukraine-Russia target graph
sbatch scripts/eval_ukr_rus_twitter_model_list_all_tasks.sbatch \
  scripts/eval1_model_list.txt
```

### MAG240M pretraining

```bash
python experiments/run_single_experiment.py \
  --dataset mag240m --root <DATA_ROOT> \
  --original_features True \
  -ds_cap 50010 -val_cap 100 -test_cap 100 \
  --epochs 1 -ckpt_step 1000 \
  --layers S2,U,M -lr 3e-4 \
  -way 30 -shot 3 -qry 4 \
  -eval_step 1000 -task cls_nm_sb \
  -bs 1 -aug ND0.5,NZ0.5 -aug_test True \
  -attr 1000 --device 0 --prefix MAG_PT_PRODIGY
```

### arXiv evaluation (zero-shot transfer from MAG)

```bash
python experiments/run_single_experiment.py \
  --dataset arxiv --root <DATA_ROOT> \
  -ds_cap 510 -val_cap 510 -test_cap 500 \
  -eval_step 100 --epochs 1 \
  --layers S2,U,M -way 3 -shot 3 -qry 3 -lr 1e-5 \
  -bert roberta-base-nli-stsb-mean-tokens \
  -pretrained <PATH_TO_CHECKPOINT> \
  --eval_only True --train_cap 10 --device 0
```

### Knowledge graphs

See `kg_commands.py` for copy-paste commands covering Wiki, FB15K-237, NELL, and ConceptNet.

---

## Key parameters

| Flag | Description |
|---|---|
| `--dataset` | Dataset name: `midterm`, `ukr_rus_twitter`, `covid19_twitter`, `arxiv`, `mag240m`, `Wiki`, … |
| `--root` | Root directory containing the dataset folder |
| `--graph_filename` | Graph `.pt` filename inside `--root` (Twitter datasets) |
| `--layers` | Model architecture, e.g. `S2,U,M` (2× SAGE + Upsample + Metagraph) |
| `--task_name` | Task: `temporal_link_prediction`, `neighbor_matching`, `classification`, … |
| `-way / -shot / -qry` | Few-shot setup: number of classes, support examples, query examples |
| `--input_dim` | Node feature dimension (384 for MiniLM embeddings) |
| `--original_features` | Use pre-computed node features instead of BERT re-encoding |
| `-aug` | Augmentation: e.g. `ND0.5,NZ0.5` (node drop 50% + zero 50%) |
| `--device` | GPU index or `cpu` |
| `--prefix` | Run name prefix for W&B and checkpoint directory |

Layer notation: `S` = GraphSAGE, `U` = Upsample, `M` = Metagraph, `A` = Average pool. Numbers indicate repetition (e.g. `S2` = 2 SAGE layers).

---

## Adding a custom dataset

See [CUSTOM_DATASET_GUIDE.md](CUSTOM_DATASET_GUIDE.md) for a step-by-step walkthrough.

---

## Citation

```bibtex
@article{Huang2023PRODIGYEI,
  title={PRODIGY: Enabling In-context Learning Over Graphs},
  author={Qian Huang and Hongyu Ren and Peng Chen and Gregor Kr\v{z}manc and Daniel Zeng and Percy Liang and Jure Leskovec},
  journal={ArXiv},
  year={2023},
  volume={abs/2305.12600}
}
```

This repo reuses code from [CSR](https://github.com/snap-stanford/csr) for KG dataset loading.
