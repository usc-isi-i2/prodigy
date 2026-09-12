# Election 2020 extension

Extend the interleaved pair sweep from eight graphs to all nine, preserving the protocol in `interleaved_mlp_pairs.md`. Exactly eight new pairs contain Election 2020. The existing 28 completed pairs are referenced through checked symlinks in a separate output root; their checkpoints and original results are untouched. All 36 models are evaluated on all nine existing uniform target pair sets (324 cells) into the extension root.

Run `python -m mixture_scaling.interleaved_election plan --root <root>` before launch. Then, inside a dedicated Tucker worktree and tmux session:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_interleaved_election.sh /dataMeR1/phil/gfm/mixture-scaling/state/interleaved_mlp_pairs_all9_s0
```

The extension refuses incomplete originals, conflicting reuse paths, and a root equal to the original. Reused runs must match the requested training protocol. Four workers share the locked queue on GPUs 0–3. Only the new pairs train, from fresh initialization with shared AdamW, alternating batches, mean source-validation BCE selection and patience 3, and a 100,000-total-update cap. Old eight-graph analyses remain valid and separate; nine-graph held-out comparisons average over seven targets outside each pair.
