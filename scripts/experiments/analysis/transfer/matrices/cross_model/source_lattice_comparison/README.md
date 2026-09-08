# Recent source-pair impact across models

`plot_impact.py` compares the completed 36-pair lattices for PRODIGY/NM,
GraphSAGE/LP, GraphSAGE/GraphMAE, and SAMGPT/GraphCL on five classification
targets. Each dot is pair ROC-AUC minus a constituent singleton on the same
model/objective and target. The second panel retains the better constituent
baseline. Black bars mark medians; target colors are consistent across panels.

All 180 pair-target cells per model are included, including targets seen during
pretraining. The 360 directional contrasts per model reuse each pair twice and
are not independent observations. These are single-seed descriptive results.

Budgets/readouts differ: PRODIGY uses update 2500 and seed 0; GraphSAGE uses
SSL-validation-selected checkpoints and seed 0; SAMGPT uses update 500 and
seed 39. GraphSAGE uses frozen linear probes, whereas PRODIGY and SAMGPT use
episodic few-shot evaluation. These are within-model composition effects, not
a controlled architecture-only comparison.

Input paths and SHA-256 hashes are recorded in `data/provenance.json`. Inputs
come from PRODIGY and the sibling `mixture-scaling-graphmae` and
`samgpt-social-lattice` repositories. Derived contrast CSVs preserve every dot.
The script validates full singleton/pair/LOO coverage, uniqueness, metric ranges,
and within-target episode fingerprints where the source table provides them.

Reproduce from this directory with `/opt/homebrew/bin/python3.11 plot_impact.py`.

## Native objectives

`plot_native_impact.py` produces separate metric-scale panels for PRODIGY
neighbor-matching macro ROC-AUC (nine targets), GraphSAGE/LP link-prediction
ROC-AUC (six targets), and SAMGPT/GraphCL positive-minus-negative probability
margin (nine targets, terminal update 500). The first two display percentage-point
AUC changes; SAMGPT displays raw margin changes. The sign always favors the pair
when positive. These are task-aligned evaluations; GraphSAGE's downstream LP
scorer/negative protocol is not identical to its training loss.

GraphMAE's saved lattice evaluation offers classification and static LP only;
neither measures its masked-feature reconstruction objective. Its training TSV
contains aggregate source-mixture validation loss, not a matched all-target
reconstruction grid, and is not used as a substitute. No new GPU evaluation was
launched. Native input hashes and complete contrast tables are under `data/` with
the `native_` prefix. `native_summary.csv` retains raw metric units.

Use `--heldout-only` to retain targets absent from both pair sources, preserving
consistent target-color order within each distribution. This produces separate
`native_heldout_` outputs. Add `--include-mae-lp` to include GraphSAGE/GraphMAE's
available downstream link-prediction results, explicitly labeled as such; these
outputs use the `native_heldout_with_mae_lp_` prefix. All variants retain their
own CSVs, provenance records, PNGs, and PDFs.
