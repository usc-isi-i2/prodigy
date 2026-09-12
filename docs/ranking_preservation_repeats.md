# Ranking-preservation continuation-seed repeats

Repeat the two diagnostic pairs and rank weights 0/1 from `ranking_preservation.md` with continuation seeds 1, 2, and 3: twelve new models, evaluated on eight fixed targets (96 cells). Seed-0 pilot outputs remain separate.

Only the continuation training randomness changes: source-local positive-edge permutations and sampled negatives, plus global RNG initialization. Both arms within a pair/seed use identical sampling seeds. The singleton weights and AdamW states, graph split/context cache seed, and source-validation/evaluation pair seeds remain 0. Metadata records `seed: 0` for the original protocol and `run_protocol.continuation_seed` for the repeat. No retraining of singleton checkpoints and no resampling of test pairs.

All other choices remain fixed: penalty weight 1 versus 0, frozen A teacher, equal A/B alternation, 100k-total-update cap, 2k validation interval, three-check patience, minimum mean-BCE improvement 1e-4, and step zero eligible for selection. See the pilot document for the ranking KL definition and post-hoc case-selection caveat. These repeats quantify continuation sampling variability, not pretraining or evaluation uncertainty. Source validation selects checkpoints; test scores will not select seeds or penalty weights.

Launch in isolated Tucker worktree/tmux on GPUs 0–3:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
bash scripts/run_ranking_preservation_repeats.sh /dataMeR1/phil/gfm/mixture-scaling/state/ranking_preservation_repeats_s1_s3
```
