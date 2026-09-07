# RQ1 label-efficient unseen-family transfer

This analysis is populated only after the complete 96-cell paired grid finishes:
four targets × four budgets × scratch/pretrained × three seeds. `analyze.py`
rejects incomplete grids, mismatched paired label samples or splits, unhashed
pretraining checkpoints, and duplicate cells before writing any headline result.

The primary figure is `figures/rq1_label_efficiency_by_target.{png,pdf}`. The
primary numerical estimand is paired pretrained-minus-scratch test ROC-AUC at
each label budget; label-efficiency AULC is secondary.
