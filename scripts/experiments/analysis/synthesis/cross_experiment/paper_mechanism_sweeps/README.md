# Paper mechanism sweeps

This analysis is preregistered before the matched sweep completes. It answers three
questions from the paper presentation at fixed eight-source corpus composition:

1. Does the probability that an episode spans graphs have an interior optimum, or is
   graph-local training best?
2. Does transfer improve with pretraining updates when source count is held fixed?
3. Does a 512-dimensional encoder change the NM data-scale trajectory relative to the
   256-dimensional baseline?

The primary ratio comparison uses the 10k checkpoint and five probabilities on both
the nine-target NM panel and five-target downstream classification panel. The data-scale
curves use fixed 2k, 4k, 6k, 8k, and 10k checkpoints. Capacity is NM-only. Every mean is
formed across the fixed target panel inside a training seed; the seed is the replication
unit and shading is the observed three-seed range.

A common ratio winner is reported only when the same probability leads both tasks by at
least 0.001 ROC-AUC over the runner-up and its worst seed-mean target effect versus p=0
is at least -0.001. Otherwise the result is reported as task- or target-dependent. A
positive data-scale result requires the 10k-minus-2k macro effect to be at least 0.001,
positive in at least two seeds, and no seed-mean target effect below -0.001. Capacity uses
the same endpoint and worst-target safety margins. These rules do not convert the three
seeds into a population confidence interval.

Expected inputs are exactly 810 NM cells and 375 classification cells, with identical
checkpoint hashes, steps, training revisions, source sets, and training seeds across
tasks for the 75 standard-width physical checkpoints.
