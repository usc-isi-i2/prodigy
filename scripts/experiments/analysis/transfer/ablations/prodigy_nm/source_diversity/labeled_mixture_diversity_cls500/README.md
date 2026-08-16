# Labeled-mixture diversity at fixed compute

Analysis for `setup/labeled_mixture_diversity_cls500/`. The experiment trains every
nonempty proper subset of five labeled graphs for 500 optimizer steps and evaluates
only on absent targets using 500 paired 10-shot classification episodes. Endpoint
controls compare the held-out four-source mixture with target-only and all-five
pretraining.

`analyze.py` validates the complete 75-cell held-out matrix and 10 endpoint cells,
then writes standalone data, summaries, and figures beneath this folder.
