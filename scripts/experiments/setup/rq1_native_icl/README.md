# RQ1 native in-context evaluation

Zero-update native PRODIGY evaluation following the original paper's separation:
support examples come from the target graph's labeled training split and queries come
from its test split. Compare leave-one-family-out neighbor-matching pretraining with
the identical randomly initialized prompt-graph architecture at 1, 3, 5, and 10 shots
per class, five support-pool seeds, and 500 fixed test episodes.
