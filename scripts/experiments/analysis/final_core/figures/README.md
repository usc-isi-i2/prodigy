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
| `coverage_status` | Observed and pending experiment blocks, including the two remaining SAMGPT seeds. |

PNG files are convenient for review and Markdown. PDFs are vector versions for
papers and presentations. Raw PRODIGY and SAMGPT metric values never share an
axis: cross-architecture effects are direction-aligned within separate panels.
