# ogbl-collab regime H1 prerequisite

This bounded no-training audit asks whether the observable repeat/novel split can
turn existing AA-DC and compact-joint complementarity into a temporally stable
mixture. It uses the same three joint checkpoints on 2017 and forward 2018.

The rule is fixed before execution: use AA-DC for pairs previously observed in the
historical graph and the seed-matched joint scorer for novel pairs. Each expert is
mapped through an empirical CDF fitted to its pooled 2017 positive and negative
scores. Those calibration arrays and the regime rule are applied unchanged to 2018.
The negative cutoff is recomputed after mixing.

Primary comparison: seed-matched joint scorer. Secondary anchors: AA-DC and the
better single expert on each panel. The prerequisite passes only when every seed
improves over the seed-matched joint model on both years, mean improvement is at
least 0.5 percentage point on both years, forward-2018 repeat recall loses at most
0.1 point versus AA-DC, and forward-2018 overall Hits@50 is at least 0.5 point above
the better single expert. Otherwise stop before H1 training.

This is a mechanism prerequisite, not a leaderboard candidate. It does not access
2019 scores, train a gate, tune a threshold, or authorize subsequent training.
Prior 2019 exploration elsewhere in the research program remains disclosed.

On Tucker:

```bash
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
python scripts/experiments/setup/ogbl_collab_regime_h1/audit.py --dry-run
python scripts/experiments/setup/ogbl_collab_regime_h1/audit.py \
  --out /dataMeR1/phil/gfm/ogbl_collab_compact_joint/regime_h1_prerequisite_v1
```

