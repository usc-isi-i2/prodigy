# Final-core downstream classification at step 2,500

`classification_auc.tsv` contains the complete 5-source × 5-target PRODIGY
classification panel for final-core training seed 2 at checkpoint step 2,500.
All cells use evaluation seed 0, 128 fixed episodes, 10-shot support, and the
same episode fingerprints as the earlier architecture-matrix classification
panel.

The full evaluation rows, including accuracy and F1, remain on Tucker in:

- `/dataMeR1/phil/gfm/prodigy-ladder-xeval/log/ladder_cross_task_eval/downstream2500/worker{0,1}.jsonl`
- `/dataMeR1/phil/gfm/prodigy-ladder-xeval/log/ladder_cross_task_eval/downstream2500_supp/worker{0,1}.jsonl`
- `/dataMeR1/phil/gfm/prodigy-archtraj/log/icl_arch_matrix/finalcore_cls2500_facebook/worker{0,1}.jsonl`

The Facebook-target cells were evaluated at commit `afcca20`; the original
four-target cells use the protocol implemented on `codex/ladder-cross-task-eval`.
