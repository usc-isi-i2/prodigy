# Pretrain saturation — the already-trained half (no training)

Does downstream transfer performance saturate early in pretraining? This folder supplies
twelve of the eighteen points on that curve from checkpoint trajectories that already
exist on Tucker. The other six (steps 100 and 500) come from
[`../pretrain_saturation_dense/`](../pretrain_saturation_dense/). Both halves feed one
analysis folder, `analysis/transfer/ablations/saturation/prodigy_nm/one_hop/pretrain_saturation/`.

## The curve

Three pretraining corpora × six steps = 18 encoders, scored on the same downstream
benchmark:

| arm | corpus | historical run | surviving trajectory |
|---|---|---|---|
| `all8` | 8-source merge, `graph_id`/`balanced` episode confinement | `merged_ukr_rus_covid_midterm_all8_nm_wb_09_07_2026_15_10_30` | 1000…43000 every 1k |
| `ukr` | `ukr_rus_twitter` alone (ladder rung 1) | `ukr_only_nm_14_06_2026_16_39_00` | 1000…119000 every 1k |
| `covid` | `covid19_twitter` alone | `covid_only_nm_14_06_2026_16_38_56` | 1000…119000 every 1k |

Steps: **100, 500** (dense folder) and **1000, 2000, 10000, 40000** (here). Broad-corpus
versus narrow-corpus is the contrast — if a wider mixture saturates later, the `all8`
curve keeps climbing after the two single-source curves flatten.

Arm definitions, the step split, and the `sat_<arm>_s<step>` model-key convention live in
[`arms.py`](arms.py), which the dense folder imports by path. Change them there, once.

## Tasks

Node regression (10-shot, log1p) on the four graphs carrying profile targets, and node
classification (10-shot) on the four labeled graphs:

```
12 ckpts × (4 graphs × 2 targets  +  4 graphs) = 144 jobs
```

**Two regression targets, not the three used by `nm_ladder_downstream`:**
`followers_count` and `account_age_days`. Followers is the topology-explainable target
(it tracks in-degree); account age has no topological route at all. The pair brackets the
range, so a difference in saturation timing between them is informative.
`statuses_count` is another scale measure and would largely retrace the followers curve.
Both picks are in the existing sweeps' target set, so these rows stay comparable to the
ladder arms in the shared CSVs.

**Neighbor matching is off by default.** The question is when *downstream* transfer
saturates, and NM is the pretraining objective itself. `WITH_NM=1` adds it (+8 jobs per
checkpoint) if you want rows directly comparable to the NM ladder tables.

**Static and temporal LP are out of scope.** The runner's episodic `slp` path is void —
center-blind scoring, frozen random prototypes, degree-confounded negatives (see
AGENTS.md and `analysis/objectives/multitask_ssl/multitask_ssl/FINDINGS_rescore.md`). Temporal LP has the same
unrepaired defect. Valid static LP would have to go through
`scripts/eval/pair_link_eval.py`.

## Running it

```bash
STATE_DIR=/dataMeR1/phil/gfm/prodigy/state python3 make_model_list.py
```

```bash
DRY_RUN=1 bash run_eval_sweep.sh --gpus 0,1
```

```bash
bash run_eval_sweep.sh --gpus 0,1
```

`make_model_list.py --dry-run` prints the plan and resolves nothing. The sweep exits
nonzero on a missing checkpoint rather than producing a curve with holes, which would
read as a flat region instead of as missing data.

**GPU etiquette:** checked 2026-07-27 — a vLLM worker is pinned across GPUs **2 and 3**
(~76 GB each), so only 0 and 1 are actually free. This moves; verify before launching:

```bash
ssh tucker nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
```

## Checkpoint semantics: these files are one step off from the dense half

All three historical runs predate the 2026-07-26 checkpoint-naming fix. The old trainer
tested and named an in-loop save by the pre-increment loop variable, so a historical
**`state_dict_N` holds N+1 completed optimizer steps**. The dense retrains use the fixed
trainer, where `state_dict_N` holds exactly N.

So the spliced curve mixes a 1001-step point labelled 1000 with a 500-step point labelled
500. One step in a thousand is numerically irrelevant and the artifacts are **not**
renamed — but the discrepancy is real, and it is what
[`../pretrain_saturation_dense/check_splice.py`](../pretrain_saturation_dense/check_splice.py)
exploits to test the splice: a dense `state_dict_1001` should equal a historical
`state_dict_1000` tensor-for-tensor.

Do not resolve "the highest-numbered checkpoint" across pre- and post-fix runs; pin the
step, which this folder does.

## Why splicing across the code boundary is safe here

`ukr` and `covid` were trained on 2026-06-14, and ~20 commits have touched `data/` since.
Read against the June-era HEAD (`f3e0cd8`), none of them changes the plain-NM path:

- `experiments/sampler.py` and `experiments/model.py` are unchanged.
- `NeighborTask.sample()` was split into `_sample_uniform`/`_sample_stratified`/
  `_sample_confined`, but `strata` defaults to `None`, so a single-source run takes
  `_sample_uniform` — the old body with the inner block extracted verbatim, same `rng`
  calls in the same order.
- `models/multilayer_gnn.py` added `multi_readout`, but with `gnn_type: sage` it is
  `False` and `reset_mlp_m` stays `Linear(emb_dim, emb_dim)` — same shape, same RNG draw,
  so the random init is unchanged. `experiments/layers.py` replaced a hardcoded
  `gnn_type="sage"` with a parameter defaulting to `"sage"`.
- Everything in `data/ukr_rus_twitter.py` sits inside `static_link_prediction` /
  `classification` / regression branches, and `_select_target_from_feature` returns early
  on an empty `target_feature`.
- The `hard_negatives`, `fp_mask_*`, `mix_*`, `e4_*` and `neighbor_sampling_*` kwargs
  threaded through `trainer.py` are read only by branches a NM run never enters.

That is a reading, not a measurement, which is why `check_splice.py` exists. Run it
before trusting the joined curve.
