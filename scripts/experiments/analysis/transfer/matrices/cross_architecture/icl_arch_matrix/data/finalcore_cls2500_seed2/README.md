# Final-core downstream classification at step 2,500

`classification_auc.tsv` contains the complete 5-source × 5-target PRODIGY
classification panel for final-core training seed 2 at checkpoint step 2,500.
All cells use evaluation seed 0, 128 fixed episodes, 10-shot support, and the
same episode fingerprints as the earlier architecture-matrix classification
panel.

`classification_trajectory_300_900.tsv` contains the complete 5-source ×
5-target classification matrices at checkpoint steps 300 and 900: 50 cells in
total. `classification_self_trajectory.tsv` is the ten-cell diagonal subset
collected first. All trajectory cells use the same training seed, fixed
128-episode target streams, and evaluation protocol as the step-2,500 panel;
both files retain ROC-AUC, accuracy, and F1.

The full evaluation rows, including accuracy and F1, remain on Tucker in:

- `/dataMeR1/phil/gfm/prodigy-ladder-xeval/log/ladder_cross_task_eval/downstream2500/worker{0,1}.jsonl`
- `/dataMeR1/phil/gfm/prodigy-ladder-xeval/log/ladder_cross_task_eval/downstream2500_supp/worker{0,1}.jsonl`
- `/dataMeR1/phil/gfm/prodigy-archtraj/log/icl_arch_matrix/finalcore_cls2500_facebook/worker{0,1}.jsonl`
- `/dataMeR1/phil/gfm/prodigy-archtraj/log/icl_arch_matrix/finalcore_cls_self_300_900_seed2/gpu{0,1}/results/*.jsonl`
- `/dataMeR1/phil/gfm/prodigy-archtraj/log/icl_arch_matrix/finalcore_cls_offdiag_300_900_seed2/gpu{0,1}/results/*.jsonl`

The Facebook-target cells were evaluated at commit `afcca20`; the original
four-target cells use the protocol implemented on `codex/ladder-cross-task-eval`.
