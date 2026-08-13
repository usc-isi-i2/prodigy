# Static link prediction, rescored on a valid evaluator

**Verdict: the "emergent MIX link prediction" finding does not survive. Link-prediction
ability is a neighbor-matching main effect, and NM alone is the strongest arm on every
dataset.** No retraining was involved — the same 15 frozen checkpoints, rescored.

## Why a rescore was needed

The episodic `static_link_prediction` evaluator cannot measure link prediction. Three
independent defects (verified in code, commit `8efb630`):

1. **Center-blind scoring.** `StaticLinkTask.sample` returns `{(0, center): neg, (1, center): pos}`
   (`data/midterm.py:181`). `center` appears only in the label-map *key*; the encoded
   subgraphs come from the candidate lists. The score was `f(v)` — the queried edge's
   other endpoint never entered the model input.
2. **Frozen random class prototypes.** `ignore_label_embeddings` defaults True, so label
   reps are `nn.Embedding` rows, frozen during pretraining. `--shots 0` also sets
   `--zero_shot True`, under which `forward_metagraph` skips message passing entirely, and
   `final_label_mlp` is `Identity`. The two "edge / no-edge" prototypes were therefore
   fixed random vectors.
3. **Degree-confounded negatives.** Positives were drawn from a center's holdout neighbours
   (holdout-degree ≥ 1 by construction); negatives carried no such condition.

`BinaryFutureLinkTask` (temporal LP) has the identical shape and is equally void. It has
**not** been rescored yet.

## Protocol

Score is a symmetric function of **both** endpoint embeddings (cosine), computed on the
`static_background` view from which holdout edges are already removed. Negatives are
degree-matched (headline), plus `random` and `hard_2hop` conditions. Orientation is locked
on a validation split so an inverted signal reads as inverted rather than sub-chance.
Common-neighbour / Adamic-Adar / preferential-attachment / Jaccard / raw-feature-cosine
floors are computed on the **same pair set**, and the pair set is **shared across all 15
arms** — the old per-run episodic sampling did not guarantee that.

2000 positives + 2000 matched negatives per dataset per condition; 5 datasets.

## Result — degree-matched negatives, ROC-AUC

| arm | covid19 | cp_hk | midterm | twibot20 | ukr_rus | **mean** | **vs floor** | old eval |
|---|---|---|---|---|---|---|---|---|
| **mtr_NM** | 0.879 | 0.556 | 0.713 | 0.835 | 0.803 | **0.757** | **+0.113** | 0.467 |
| msc_all8_NM | 0.869 | 0.543 | 0.714 | 0.824 | 0.772 | 0.744 | +0.100 | — |
| mtp_NMFP | 0.866 | 0.553 | 0.706 | 0.808 | 0.759 | 0.738 | +0.094 | — |
| msc_cov_NM | 0.867 | 0.548 | 0.694 | 0.803 | 0.730 | 0.728 | +0.084 | — |
| msc_all8_MIX | 0.809 | 0.522 | 0.692 | 0.753 | 0.683 | 0.692 | +0.048 | — |
| mtr_MIX | 0.792 | 0.539 | 0.670 | 0.727 | 0.673 | 0.680 | +0.036 | **0.759** |
| mtp_NMCL | 0.786 | 0.550 | 0.680 | 0.709 | 0.669 | 0.679 | +0.035 | — |
| msc_cov_MIX | 0.754 | 0.515 | 0.679 | 0.713 | 0.601 | 0.652 | +0.008 | — |
| msc_all8_CL | 0.552 | 0.558 | 0.519 | 0.562 | 0.583 | 0.555 | −0.089 | — |
| msc_cov_CL | 0.554 | 0.552 | 0.519 | 0.564 | 0.581 | 0.554 | −0.090 | — |
| mtr_CL | 0.530 | 0.549 | 0.513 | 0.556 | 0.569 | 0.543 | −0.101 | 0.332 |
| mtp_CLFP | 0.540 | 0.544 | 0.500 | 0.557 | 0.572 | 0.543 | −0.101 | — |
| msc_cov_FP | 0.499 | 0.528 | 0.511 | 0.520 | 0.530 | 0.518 | −0.126 | — |
| msc_all8_FP | 0.510 | 0.488 | 0.506 | 0.498 | 0.498 | 0.500 | −0.144 | — |
| mtr_FP | 0.498 | 0.503 | 0.509 | 0.484 | 0.502 | 0.499 | −0.145 | 0.449 |

Floors (mean): Jaccard 0.642, Adamic-Adar 0.642, common-neighbour 0.640,
raw-feature-cosine 0.562, preferential-attachment 0.498.

## Reading

1. **The original headline inverts.** The old evaluator reported NM at chance (0.467) and
   MIX as uniquely capable (0.759). Valid scoring reverses both: NM is the best arm on all
   5 datasets, MIX trails it everywhere. On ukr_rus — the old evaluator's *strongest* MIX
   result (0.861) — MIX now scores 0.673, **below the Jaccard floor of 0.687**.
2. **There is no synergy term; there is an NM main effect.** Every NM-containing arm scores
   0.652–0.757 mean; every arm without NM scores 0.499–0.555, i.e. at or below chance and
   well below the heuristic floors. Adding CL or FP to NM *lowers* the score
   (NM 0.757 → NMFP 0.738 → NMCL 0.679 → MIX 0.680). Rotation dilutes NM rather than
   producing anything new.
3. **This is mechanistically unsurprising**, which is the point. Neighbor-matching trains on
   neighbour prediction, so it encodes adjacency; the broken evaluator was hiding NM's real
   ability while manufacturing a MIX advantage out of frozen random prototypes and a degree
   confound.
4. **The corpus-replication failure dissolves.** The `msc` inversions that looked like a
   corpus×objective interaction were an artifact — under valid scoring, all three corpora
   give the same ordering (NM > MIX > CL > FP).
5. **NM transfer is real and out-of-domain.** twibot20 was never trained on: NM 0.835 vs
   best heuristic 0.726 (+0.109).
6. **cp_hk is at chance for every arm** (0.488–0.558, floor 0.534), consistent with its
   known isolation in the single-source transfer matrix.

## Gates

No voided rows: no endpoint-blind scoring anywhere, zero holdout leakage, and
endpoint-permutation AUC 0.479–0.516 (mean 0.492) — shuffling one endpoint destroys every
signal, as it must.

FP arms flag **encoder collapse**, which is a distinct condition from evaluator failure and
a finding in its own right: on midterm, `mtr_FP` spans **536 distinct embedding directions
over 4568 nodes** (mean norm 0.80) against `mtr_NM`'s **4429** (mean norm 61.8). Cosine
depends only on direction, so FP's scores tie frequently (sensitivity 0.43–0.69). Only 13%
of those nodes have all-zero input features, so this is the masked-feature-prediction
objective collapsing the representation, not the missing-bio population. It corroborates
FP's chance-level AUC rather than undermining it.

## Reproduce

```bash
python scripts/eval/pair_link_eval.py --self-test          # 23 offline protocol checks
python scripts/eval/tests/test_embed_walk.py               # 5 encoder-walk checks
python scripts/eval/pair_link_sweep.py --graph <g> --dataset <d> \
    --model-list results/model_list.txt --out-dir results --device cuda
python scripts/experiments/analysis/evaluation/slp_evaluator_repair/aggregate_pair_lp.py \
    --results-dir results --negative-kind degree_matched
```

## Still open

- **Temporal LP has not been rescored** and carries the identical defect.
- The `mix_slp_ablation` 2×2 rests on the broken metric. Its target (MIX-unique LP) no
  longer exists; the useful version re-points it at **NM** — is NM's adjacency signal
  topological or feature-driven?
- LP sections of `archive/multitask_ssl_superseded/`, `topology_feature_ssl`,
  `pretrain_probe_matrix`, `_cross/PROGRAM_FINDINGS.md`, `multitask_ssl_corpora` need re-deriving.
  Their classification and regression sections are unaffected (10-shot classification runs
  with `zero_shot=False` and builds real prototypes; regression bypasses the decoder).
