# Final-core figures

These figures are generated from the canonical
`../data/results_full_long.tsv` table by `../plot_final_results.py`.

| figure | purpose |
|---|---|
| `specialist_transfer_matrices` | Specialist source-to-target transfer for each architecture on its native pretext. |
| `target_entry_effects` | Direction-aligned effect when each target enters the training mixture; positive always means improvement. |
| `target_entry_before_after` | Raw before/after values with architecture-appropriate favorable side of the diagonal. |
| `order_robustness` | Target-entry effect distributions for orders A, B, and C. |
| `prodigy_seed_stability` | Completed three-seed robustness evidence for PRODIGY. |
| `ladder_trajectories` | Complete per-target ladder trajectories for all three orders and both architectures. |
| `ladder_trajectories_loss` | Native-pretext loss trajectories for both architectures: NM loss for PRODIGY and GraphCL BCE loss for SAMGPT. |
| `ladder_trajectories_native_accuracy` | Native-pretext evaluation accuracy for both architectures, shown in separate architecture rows. |
| `samgpt_ladder_probability_diagnostics` | SAMGPT positive/negative pair probabilities and their separation margin over the ladder. |
| `ladder_trajectories_auc` | PRODIGY per-target ROC-AUC ladder trajectories recovered from the original fixed-test logs. |
| `prodigy_ladder_seed_bands` | PRODIGY ladder means with observed min–max bands across its three training seeds. |
| `prodigy_nm_vs_cls_auc_ladders` | Five held-out targets comparing mean NM and downstream classification AUC across the three final-core mixture orders at 2,500 steps. |
| `coverage_status` | Completion status for every architecture, component, and training seed. |

PNG files are convenient for review and Markdown. PDFs are vector versions for
papers and presentations. Raw PRODIGY and SAMGPT metric values never share an
axis: cross-architecture effects are direction-aligned within separate panels.

The `neutral_detailed/` subfolder is a separate descriptive suite: raw primary
metrics, one matrix per observed seed, and one nine-target ladder breakdown per
architecture, seed, and order, without finding-oriented annotations.
