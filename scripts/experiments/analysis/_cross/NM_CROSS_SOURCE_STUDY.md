# NM Cross-Source Transfer Study: Merged vs. Single-Source Pretraining

**Question.** When we pretrain a neighbor-matching (NM) model on a *merged* retweet
graph (two event datasets concatenated disjointly) vs. on a *single* source, does the
merged model transfer worse to each single source? An early, informal run suggested
**yes** (an "inversion": single-source beat merged cross-domain). This study tests
whether that holds under a fair, controlled comparison — first for **ukr/covid**, then
validating on **covid/midterm**.

> TL;DR: Under a fair comparison the inversion **does not reproduce** — merged is ≥
> single-source. A targeted "cross-source-shortcut" fix (confining each NM episode to a
> single source) gives a small additional, consistent gain. All findings are **1 seed**
> and need a seed sweep to be called significant.

---

## 1. Background & the motivating observation

- **Task.** Neighbor matching (NM): few-shot episodes where the model matches query
  nodes to their center among `n_way` candidates, using learned node/neighborhood
  embeddings. Node features are 768-d gte-multilingual-base bio embeddings.
- **Graphs.** Disjoint retweet graphs per event (ukr_rus, covid19, midterm). A "merged"
  graph is a **disjoint block-concat** (users namespaced per source, no cross-source
  edges), carrying per-node `graph_id` + `source_graph_names` provenance.
- **Original observation.** Training NM on ukr and testing on covid appeared to beat
  training on merged ukr+covid and testing on covid (and symmetrically) — i.e.
  single-source seemed to transfer better than merged. This study asks whether that's
  real or an artifact.

### Working hypotheses for *why* merged might underperform
1. **Episode-mixture imbalance** — merged is covid-dominant, so naive sampling gives
   few episodes centered on the smaller source.
2. **Cross-source shortcut** — in a mixed episode the negatives can come from the *other*
   source, so the model separates positives from negatives via source-level feature
   differences instead of within-source neighborhood structure. Useless at single-source
   test time.
3. **Reduced per-domain exposure** — fixed total compute means each domain sees fewer
   episodes in the merged run.
4. **Construction / eval mismatch** — edge views, feature subsets, or eval protocol
   differing between runs.

---

## 2. Experiment 1 — Fair transfer matrix (ukr/covid)

**Folder:** `scripts/experiments/nm_transfer_matrix/`

### Design (everything fixed except the training data)
The original comparison was **unfair**: the single-source configs used a plain default
architecture, while the merged model it was compared against used a larger architecture
**and** augmentation. We re-ran with three byte-identical configs differing only in data
source:
- Plain default architecture, **no** augmentation, **no** attr-regression.
- `neighbor_matching`, `n_way=30`, `n_shots=3`, `n_hop=1`, `edge_view=default`, seed 0.
- **Fixed per-domain exposure**: single-source = 60k episodes; merged = 120k (2×), so
  each domain is seen ≈ as often. (Merge is proportional/as-is; covid-dominant.)
- Train {ukr, covid, merged}; test each on each → train×test AUC matrix.

### The eval bug we found (and fixed)
First eval showed **every cell ≈ chance**, including in-domain. Root cause was **not**
training — verified by an in-domain protocol sweep:

| protocol | accuracy | chance | roc_auc |
|---|---|---|---|
| 3-way, **0-shot** | 0.32 | 0.33 | 0.49 |
| 3-way, **3-shot** | **0.85** | 0.33 | **0.97** |
| 30-way, **3-shot** | **0.51** | 0.03 | **0.95** |
| 30-way, 0-shot | 0.03 | 0.03 | 0.50 |

**Zero-shot NM is degenerate** — no support shots → no class prototypes → pure guessing,
and `roc_auc` collapses to 0.5. The eval shells had pinned `--shots 0`. Checkpoint
loading, features, and architecture were all verified correct (`missing=0 unexpected=0`).
**Lesson: always eval NM at `shots ≥ 3`.** Also: at 3-way 3-shot AUC is near-ceiling, so
we added a `--nm-n-way` flag and report **30-way** as the discriminative setting, plus
**accuracy/f1** (more discriminative than AUC here; f1 ≈ accuracy for balanced episodes).

### Results (3-shot, 30-way, 1 seed; checkpoints stopped early at 50k/90k)

**Accuracy (full 3×3):**
```
train\test   ukr     covid   merged
ukr          0.5151  0.6142  0.6156
covid        0.4589  0.6641  0.6238
merged       0.4888  0.6536  0.6872
```
**AUC (key cross-domain columns):**
```
train\test   ukr     covid
ukr          0.9497  0.9741
covid        0.9245  0.9815
merged       0.9411  0.9801
```

### Conclusion
**The inversion does NOT reproduce.** Merged ≥ single-source on both cross-domain cells:
- test covid: merged 0.654 vs single-ukr 0.614 acc (**+0.039**)
- test ukr: merged 0.489 vs single-covid 0.459 acc (**+0.030**)

The original effect was an artifact of the unfair architecture/augmentation mismatch
and/or the degenerate zero-shot eval.

---

## 3. Experiment 2 — Cross-source-shortcut test (ukr/covid)

**Folder:** `scripts/experiments/nm_cross_source_shortcut/`

### Hypothesis & intervention
Hypothesis 2 above: naive merged sampling lets episode negatives come from the *other*
source, enabling a source-discrimination shortcut. We added a flag,
`--neighbor_sampling_episode_source graph_id`, that **confines every episode to a single
source** (source picked **proportional to node count**, so the per-node center marginal
is identical to naive sampling — the *only* variable changed is that an episode's
negatives all share a source). Implemented in `NeighborTask._sample_confined`
(`data/dataloader.py`), wired through `params.py` → trainer kwargs → the merged-graph
loader. Same plain config, same 120k-episode budget, seed 0.

### Results (3-shot, 30-way; accuracy shown — most discriminative)
```
regime                 test:ukr  test:covid
single ukr              0.5151    0.6142
single covid            0.4589    0.6641
merged proportional     0.4888    0.6536
merged within-source    0.5077    0.6666   <- best or tied on both
```
(AUC same ordering: within-source 0.9468 / 0.9819 vs proportional 0.9411 / 0.9801.)

### Conclusion
Within-source episodes beat the proportional merged baseline on **both** domains and
match/exceed the best single-source model on each — directionally exactly what the
shortcut hypothesis predicts. Effect: **+0.019 acc (test ukr), +0.013 acc (test covid)**.
Consistent in sign across both domains and all three metrics, but **small**, and the
proportional merged model wasn't even worse than single-source — so this is *consistent
with* the shortcut, not proof of it.

---

## 4. Methodological lessons (carried into all later experiments)
1. **Never eval NM at zero-shot.** Use `shots ≥ 3`; zero-shot has no prototypes → chance.
2. **3-way 3-shot AUC is near-ceiling** (0.95–0.99). Use **30-way** and read **accuracy**
   for discrimination; f1 ≈ accuracy for balanced episodes.
3. **Fairness requires identical architecture + augmentation + eval**, not just data.
4. **Verify checkpoint loading** (`strict=False` silently drops mismatched keys).
5. **One variable at a time.** The shortcut flag keeps the center marginal fixed and only
   removes cross-source negatives.

---

## 5. Experiment 3 — Validation on covid/midterm (IN PROGRESS)

**Goal:** confirm both findings (no inversion; within-source helps) generalize beyond
ukr/covid, on a second pair with a *much* more extreme size imbalance.

**Note on imbalance:** midterm ≈ 0.34M nodes vs covid ≈ 23M → midterm is ~**1.5%** of the
merge (vs ukr's ~31% in ukr/covid). Under proportional sampling midterm is heavily
under-exposed — which is exactly why we add a **balanced** within-source variant here.

### Atomized steps
1. **Graph code** — `scripts/graph_construction/merge_covid_midterm.yaml` + register
   `merged_covid_midterm` in the eval DATASETS, `data_loader_wrapper.py`, and trainer
   dispatch. ✅ done (commit `0ed82ec`).
2. **Build graph** — run the disjoint merge → `covid_midterm_retweet_graph.pt`; verify
   with `inspect_graphs.py` (768-d features match, merged == covid+midterm, graph_id
   present). ⏳ running.
3. **Scaffold** — add a **balanced-confined** sampling mode (source picked 50/50, new
   shared-code option) + new experiment folder mirroring Exps 1–2.
4. **Train 5 models:** single midterm, single covid, **merged** (naive/proportional),
   **merged-within** (confined, source ∝ size), **merged-within-balanced** (confined,
   source 50/50).
5. **Eval** each on {covid, midterm, merged} at 3-shot/30-way; build matrix + compare
   (accuracy/f1/auc).

### Model legend
| name | training graph | episode sampling |
|---|---|---|
| midterm | midterm | standard |
| covid | covid | standard |
| merged | covid+midterm merge | naive (mixed-source episodes) |
| merged-within | covid+midterm merge | confined to one source, source ∝ size |
| merged-within-balanced | covid+midterm merge | confined to one source, source 50/50 |

---

## 6. Reproducibility

- **Cluster ops:** Tucker, repo at `/dataMeR1/phil/gfm/prodigy`, conda env `prodigy`
  (`source /home/mhchu/miniconda3/etc/profile.d/conda.sh`). Data under
  `/dataMeR1/phil/data`. Train in tmux; one run per GPU.
- **Each experiment folder** has its own README.md + RESULTS.md with exact commands.
- **Build a matrix / comparison** (metrics already in the eval JSONs):
  ```bash
  python3 scripts/experiments/nm_transfer_matrix/build_auc_matrix.py \
    --log-root log --shots 3 --n-way 30 --metric all
  python3 scripts/experiments/nm_cross_source_shortcut/compare_shortcut.py \
    --log-root log --shots 3 --n-way 30 --metric all
  ```
- **Key flags added:** `--nm-n-way` (eval), `--neighbor_sampling_episode_source`
  (within-source confinement; balanced variant added in Exp 3).

## 7. Caveats & next steps
- **All results are 1 seed**; deltas are ≤ ~4 accuracy points. A **3–5 seed sweep** is the
  single highest-value next step to claim significance.
- **Checkpoints stopped early** (50k/90k vs 60k/120k budget) — clean full-budget reruns
  would be tidier, ideally folded into the seed sweep.
- **covid/midterm extreme imbalance** makes the proportional merged run barely train on
  midterm; the balanced within-source variant is the controlled way to address it.
