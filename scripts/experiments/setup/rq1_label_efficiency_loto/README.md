# RQ1: label-efficient adaptation to unseen graph families

This experiment tests whether multi-graph neighbor-matching pretraining reduces
the labeled supervision required to adapt PRODIGY's encoder to an unseen target.

## Contract

- Targets: `covid_political`, `election2020`, `ukr_rus_suspended`, `twibot20`.
- The target family is absent from SSL pretraining. COVID excludes both `covid`
  and `covid_political`; Ukraine suspended excludes both `ukr_rus` and
  `ukr_rus_suspended`.
- Paired downstream arms use the same pooled PRODIGY encoder plus linear head:
  random initialization (`scratch`) versus leave-one-family-out SSL initialization.
- The entire encoder and head are fine-tuned on exactly 1, 10, 100, or 1,000
  target training nodes per class.
- Same split, selected nodes, minibatches, optimizer, validation checkpoints, and
  test set are used by paired arms.
- Validation selects convergence; test is evaluated only once from the saved best
  checkpoint. Expensive state is written atomically and every cell is resumable.
- Seeds run sequentially: all seed-0 work, then seed 1, then seed 2.

Pretraining uses validation checks every 250 updates, patience 8, and a 10,000
update ceiling. The earlier final-core runs reached 2,500 updates in roughly 17
minutes; the ceiling is intentionally much larger than prior trajectories.
