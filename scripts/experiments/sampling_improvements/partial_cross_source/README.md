# Partial Cross-Source Episodes (sampling remedy #4)

**Question.** The prior cross-source-shortcut study (`../../nm_cross_source_shortcut/`)
found that confining each NM episode to a *single* source ("within-source") beats naive
merged sampling on cross-domain transfer — the model stops exploiting a source-level
shortcut. But full confinement removes **all** cross-source negatives, and the sampling
doc (`../README.md`, risk #4) notes we *do* want the model to learn to tell sources
apart. So: is the best policy full confinement (p=0), pure naive (p=1), or **something in
between**?

This experiment sweeps a single new knob that interpolates between the two already-
characterized endpoints and asks whether a small dose of cross-source episodes beats both.

## The one variable

`neighbor_sampling_cross_source_prob = p` — the probability that a given NM episode is a
naive **mixed-source** episode instead of being confined to one source. Requires
`neighbor_sampling_episode_source=graph_id`.

| p | behaviour | reproduces |
|---|---|---|
| **0.00** | every episode confined to one source | `nm_cross_source_shortcut` within-source run |
| 0.10 | 10% mixed-source episodes | — (new) |
| 0.25 | 25% mixed-source episodes | — (new) |
| 0.50 | 50% mixed-source episodes | — (new) |
| **1.00** | every episode naive/mixed | `nm_transfer_matrix` merged-naive run |

Everything else is held **byte-identical** across the five configs (plain arch, no aug,
ukr+covid merge, `n_way=30 n_shots=3 n_query=4 n_hop=1`, 120k episodes, seed 0,
source weighting `proportional`). The confined branch stays proportional so the per-node
center marginal never changes — the *only* thing that varies is the fraction of episodes
whose negatives may come from the other source. One variable, five points.

Because p=0 and p=1 reproduce established runs, they double as an **internal-validity
check** that the new code path didn't perturb anything (at p=0 the sampler short-circuits
before drawing from the RNG, so the stream is bit-identical to the old within-source run).

## Testbed choice

ukr+covid (not covid+midterm). ukr is ~31% of the merge, so per-domain **exposure** is
not the dominant confound — which isolates the cross-source-**shortcut** axis this knob is
meant to probe. (covid+midterm's ~1.5% imbalance mixes in an exposure effect; that's the
balanced-within-source story, a different remedy.)

## Files

- `p000_within.yaml … p100_naive.yaml` — the 5 sweep configs (diff them: one line differs).
- `train_tucker.sh <cfg> --device N` — train one arm.
- `run_all_train_tucker.sh` — launch all 5 in detached tmux (`GPUS="0 1 2 3 2"` overridable).
- `make_model_list.sh` — collect each run's final checkpoint into `model_list.txt`.
- `eval_tucker.sh` — eval all 5 on {ukr, covid} at NM 3-shot / 30-way.
- `build_sweep.py` — parse eval logs → table ordered by p + endpoint-vs-interior verdict.

## Run (Tucker)

```bash
# 1. train (check nvidia-smi first; our GPUs are 0-3). NM jobs are ~2 GB / per-step bound.
GPUS="0 1 2 3 2" scripts/experiments/sampling_improvements/partial_cross_source/run_all_train_tucker.sh

# 2. once checkpoints exist:
scripts/experiments/sampling_improvements/partial_cross_source/make_model_list.sh
scripts/experiments/sampling_improvements/partial_cross_source/eval_tucker.sh --device 2

# 3. table + verdict:
python3 scripts/experiments/sampling_improvements/partial_cross_source/build_sweep.py \
  --log-root log --shots 3 --n-way 30 --metric all --out-csv \
  scripts/experiments/sampling_improvements/partial_cross_source/sweep.csv
```

## Reading the result

For each test domain, `build_sweep.py` reports the argmax over p and whether it's an
ENDPOINT or INTERIOR value:
- **Interior optimum** (e.g. p≈0.1–0.25 best) → remedy #4 holds: a little cross-source
  signal helps beyond pure within-source.
- **p=0 best / monotone down** → confinement is strictly best; cross-source episodes only
  hurt. (Simplest story; consistent with the shortcut hypothesis taken to its limit.)
- **p=1 best** → the shortcut finding doesn't hold on this testbed under this budget.

All results are **1 seed** (matching the rest of this line of work). Deltas here are
expected to be small (the prior within-vs-naive gap was ~1–2 acc points); a seed sweep is
the follow-up if an interior optimum appears. The `eval-episode-seed-per-split` caveat
applies — eval episodes are seeded per split, so cross-domain agreement (both ukr and
covid pointing the same way) is the reliability signal, not a single cell.
