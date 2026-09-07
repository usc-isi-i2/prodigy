# Literature note: reporting label efficiency in GNN and graph-foundation-model experiments

> **Status: non-authoritative research note (2026-08-30).** This document records
> one targeted literature audit and a provisional interpretation for discussion.
> It is **not** an experiment contract, project decision, implementation request,
> instruction to future agents, or source of truth for the paper. Do not change
> experiments or reporting solely because this file exists. Before acting, check
> the current user request, the canonical experiment README/configuration, actual
> completed results, and the cited primary papers. Later project decisions and
> newer evidence supersede every recommendation below.

## Scope and limitations

- The paper sample was selected for relevance, not through a systematic-review
  protocol; its counts must not be generalized into field-wide frequencies.
- The categories below mix node classification, graph classification,
  meta-learning, and graph-foundation-model evaluation. Their protocols answer
  different scientific questions.
- Statements labeled as recommendations are the note author's judgment, not an
  established consensus or an approved project plan.
- Dataset counts and percentage conversions should be recomputed from the
  artifacts used by the final run before publication.

## Provisional recommendation for discussion

For the PRODIGY unseen-family adaptation experiment, use **labeled nodes per
class** as the primary x-axis:

> **Labels per class: 1, 10, 100, 1K**

This matches the quantity controlled by the experiment and the dominant
convention in few-shot node-classification and graph-foundation-model work.
Report the corresponding **total number of labels** and **percentage of the
eligible target training pool** in the caption, appendix, or a companion table.

Call the overall experiment **label-efficient target adaptation**, rather than
calling the entire curve few-shot. The 1- and 10-labels-per-class points are
few-shot; 100 and especially 1,000 labels per class extend the experiment into
low-label or partial-supervision adaptation.

If compute permits, add a secondary percentage-normalized sensitivity analysis
at **0.1%, 1%, and 10%** of each target's training pool. This should complement,
not replace, the existing per-class experiment.

## Main finding from the literature audit

There is no single universal label-budget schedule across graph learning. The
reporting convention depends on the experimental question:

| Setting | Predominant convention |
|---|---|
| Classic transductive node classification | Fixed labeled nodes per class, commonly 20/class |
| Few-shot node classification | N-way K-shot; K is the number of labels per class |
| Graph in-context learning | K-shot per class |
| Graph-foundation-model adaptation | Usually K-shot per class |
| Semi-supervised graph classification | Percentage of the labeled training set, commonly 1% and 10% |
| General fine-tuning/sample-efficiency studies | Percentage of the official training set |
| Operational annotation-cost studies | Sometimes total labeled examples |

For multiclass node classification, reporting only a total label count is hard
to compare across datasets: 100 labels can cover a binary task comfortably but
cannot cover all classes in a 349-class task. A balanced K-shot-per-class budget
avoids this ambiguity. Percentages answer a different question: what fraction of
the ordinarily available labeled training set is required?

## Evidence from published work

This was a targeted audit of **13 representative published papers**, covering
classic semi-supervised GNNs, few-shot node classification, graph pretraining,
and recent graph foundation models. It is not a formal systematic review.

### Classic semi-supervised node classification

1. **Planetoid** sampled **20 labeled instances per class**, with 1,000 test
   instances, on its citation-network node-classification benchmarks. This
   established the split later reused by much of the early GNN literature.
   [Yang, Cohen, and Salakhutdinov, ICML 2016](https://proceedings.mlr.press/v48/yanga16.html)

2. **GCN** evaluated on the standard Planetoid citation-network splits, thereby
   inheriting the 20-labels-per-class protocol.
   [Kipf and Welling, ICLR 2017](https://openreview.net/forum?id=SJU4ayYgl)

3. **GAT** likewise used the standard transductive citation-network benchmark
   splits.
   [Velickovic et al., ICLR 2018](https://openreview.net/forum?id=rJXMpikCZ)

These papers support fixed, class-balanced label counts as a standard node-level
semi-supervised protocol, although the single 20/class split is now understood
to be too narrow for a complete sample-efficiency evaluation.

### Few-shot node classification

4. **Meta-GNN** formulates graph few-shot node classification through episodic
   N-way K-shot tasks.
   [Zhou et al., CIKM 2019](https://arxiv.org/abs/1905.09718)

5. **G-Meta** evaluates graph meta-learning using shot-based episodic tasks,
   including 1-shot and 5-shot settings.
   [Huang and Zitnik, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/412604be30f701b1b1e3124c252065e6-Abstract.html)

6. **Transductive Linear Probing** reports 1-shot and 5-shot node-classification
   tasks with 2-way or 5-way episodes. A shot is one labeled support node per
   class.
   [Tan et al., LoG 2022](https://proceedings.mlr.press/v198/tan22a.html)

7. **X-FNC** reports **5, 10, and 20 labels per class** on smaller graph
   benchmarks and **50, 100, and 200 labels per class** on OGBN-Arxiv. This is
   direct precedent for increasing per-class budgets on a larger graph instead
   of treating 1/5-shot as universal.
   [Wang et al., WSDM 2023](https://arxiv.org/abs/2301.02708)

These papers make K-shot per class the clearest comparison language for
few-shot node classification. They also show that the actual K values vary with
dataset scale and experimental purpose.

### Graph pretraining and semi-supervised graph classification

8. **GraphCL** studies semi-supervised graph classification under reduced
   labeled-data regimes, using label-rate-style evaluation rather than only
   class-balanced node shots.
   [You et al., NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/3fe230348e9a12c13120749e3f9fa4cd-Abstract.html)

9. **JOAO** reports semi-supervised results at **1% and 10% label rates**.
   [You et al., ICML 2021](https://proceedings.mlr.press/v139/you21a.html)

Percentage reporting is especially natural for graph classification, where
examples are independent graphs and dataset sizes vary substantially. It is
also natural when the central claim is that pretraining replaces some fraction
of an otherwise fully labeled training set.

### Recent graph foundation models

10. **OFA** evaluates zero-shot and few-shot transfer with per-class shot
    budgets, including 1-, 3-, 5-, and in some tasks 10-shot settings.
    [Liu et al., ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/57faf5642eb06e0602b95f6aa989b38a-Abstract-Conference.html)

11. **GFT** reports N-way K-shot evaluation and also varies the number of
    fine-tuning instances per class.
    [Graph Foundation Model with Transferable Tree Vocabulary, NeurIPS 2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/c23ccf9eedf87e4380e92b75b24955bb-Abstract-Conference.html)

12. **SAMGPT** defines an m-shot downstream task as m labeled examples per
    class and emphasizes low-shot settings, particularly m <= 5.
    [SAMGPT: Text-free Graph Foundation Model](https://openreview.net/forum?id=e7eb78c5335ae090d5cce751d1c6ed62a16b7928)

13. **Task-Trees** evaluates graph in-context learning using class-balanced
    episodic tasks, including 5-way 3-shot evaluation.
    [Towards Graph Foundation Models: Learning Generalities Across Graphs via Task-Trees](https://openreview.net/forum?id=BSqf2k01ag)

Among these 13 papers, **11 use fixed labels per class or K-shot as the relevant
low-label node/GFM protocol**, while the two graph-pretraining papers highlighted
above prominently use percentage label rates for semi-supervised graph-level
evaluation. The sample was deliberately selected to cover the literature most
relevant to PRODIGY; this count should not be interpreted as a field-wide
meta-analysis.

## Interpretation of the current PRODIGY experiment

The RQ1 experiment uses **1, 10, 100, and 1,000 target training nodes per
class**. All four tasks are binary, so the corresponding total annotation
budgets are **2, 20, 200, and 2,000 nodes**.

The experiment uses a stratified 60/20/20 train/validation/test split. Relative
to the eligible labeled training pool, the budgets are approximately:

| Target | Eligible labeled training pool | 1/class | 10/class | 100/class | 1K/class |
|---|---:|---:|---:|---:|---:|
| COVID Political | 47,203 | 0.004% | 0.042% | 0.424% | 4.24% |
| Election 2020 | 47,359 | 0.004% | 0.042% | 0.422% | 4.22% |
| Ukraine Suspended | 33,863 | 0.006% | 0.059% | 0.591% | 5.91% |
| TwiBot-20 | 7,095 | 0.028% | 0.282% | 2.82% | 28.2% |

This reveals an important limitation of a shared K-shot axis: 1,000/class is a
4-6% label regime on three targets but a 28% label regime on TwiBot-20. The
paired pretrained-versus-scratch comparison remains valid within every target,
but aggregated cross-target summaries should acknowledge the different label
fractions.

## Recommended paper presentation

### Main plot

- Title the experiment **Label-efficient adaptation to unseen graph families**.
- Label the x-axis **Labeled target nodes per class**.
- Use the existing logarithmic ticks: **1, 10, 100, 1K**.
- Show individual target panels or target-specific curves; do not imply that a
  given K represents the same label fraction on every target.
- Report uncertainty across independently sampled labeled sets, not only model
  initialization seeds.

Suggested caption language:

> Each budget is the number of labeled target training nodes per class. All
> targets are binary, so 1/10/100/1K labels per class correspond to 2/20/200/2K
> total labels. Depending on the target, these budgets span approximately
> 0.004%-28.2% of the eligible labeled training split. Paired scratch and
> pretrained runs use identical sampled labels, splits, minibatches, and model
> selection.

### Companion table or appendix

For every target and budget, report:

1. Number of classes.
2. Labels per class.
3. Total labels used.
4. Percentage of the eligible training pool.
5. Number of independent label samplings.
6. Mean performance and uncertainty across those samplings.

### Optional normalized sensitivity analysis

To rule out an artifact caused by different dataset sizes, repeat the comparison
at **0.1%, 1%, and 10% of each target's eligible training labels**, using
stratified sampling. This would directly support a statement such as
"pretraining reduces the fraction of target labels required." The current
K-shot experiment should remain primary because it matches the GFM/few-shot
literature and preserves explicit class coverage.

## Terminology to use

- **K-shot**: K labeled examples per class.
- **Few-shot**: reserve primarily for the 1- and 10-shot points in this study.
- **Label-efficient adaptation**: the full 1-1,000 labels/class curve.
- **Label rate**: percentage of the eligible labeled training pool used.
- **Total annotation budget**: total labeled nodes across all classes.

Avoid describing "1,000-shot" as an extreme few-shot condition, especially on
TwiBot-20, where it consumes approximately 28% of the eligible training labels.
