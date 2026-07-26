# Superseded: multitask_ssl_rotation / multitask_ssl_pairs

Retired 2026-07-26. Everything here rests on the static-link-prediction evaluator that
was found invalid on 2026-07-23 (center-blind scoring, frozen random prototypes,
degree-confounded negatives). **Do not cite any LP number from these files.**

The live analysis is [`../../multitask_ssl/`](../../multitask_ssl/) — the valid
classification/regression content from both runs is consolidated there, together with
the rescored LP.

## What is kept and why

| path | why kept |
|---|---|
| `rotation/FINDINGS.md`, `rotation/FINDINGS_v1_archived.md`, `pairs/FINDINGS.md` | the record of what we believed and published internally before the rescore |
| `rotation/analysis.ipynb` | **contains the "Impact of adding each objective" boxplot (cell 73), which was never `savefig`'d anywhere.** Regenerate from here if it is needed for a deck. Its LP panels are void. |
| `*/static_link_prediction_VOID.csv` | the raw void sLP sweeps, so the defect can be re-examined against the rescored numbers |
| `rotation/0_capability_plane.*`, `rotation/2_static_link_prediction.*`, `rotation/perf_by_model_static_link_prediction.*`, `pairs/perf_by_k.*` | figures whose y-axis is void sLP |
| `rotation/plot_capability_plane.py`, `pairs/plot_perf_by_k.py`, `rotation/aggregate_results.py`, `rotation/build_results_xlsx.py` | the code that produced them; `aggregate_results.py` here is the 3-arm rotation version, superseded by the 7-arm pairs version now in `multitask_ssl/` |
| `rotation/multitask_ssl_rotation_results.xlsx` | the workbook handed around at the time |

Setup/reproduction for the underlying runs is untouched, in
`../../../setup/multitask_ssl_rotation/` and `../../../setup/multitask_ssl_pairs/`.
