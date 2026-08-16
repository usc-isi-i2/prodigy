# Facebook-trained PRODIGY trajectory

This folder records the Facebook single-source model trajectory added to the
PRODIGY checkpoint analysis.

- Model: `ss_facebook_page_reference`
- Training objective: neighbor matching on `facebook_page_reference`
- Checkpoints: 20, 60, and 100 training steps
- Classification targets: COVID political, Election 2020, Ukraine suspended,
  TwiBot-20, and Facebook page reference
- Evaluation regime: frozen 10-shot in-context classification, seed 0, 128
  episodes
- Native pretext evaluation: 3-shot, 30-way neighbor matching on Facebook,
  seed 0, 128 episodes
- Evaluated: 2026-08-15 18:39:46–18:42:50 PDT
- Evaluator commit: `361212c`

`classification.jsonl` contains the complete 20/60/100 trajectory. The four
non-Facebook step-100 rows were already present in
`../raw_aggregate/prodigy.jsonl`; they are repeated here so this source's
trajectory is self-contained. The step-0 classification points in the figure
come from the shared random-initialization controls in `../random_init/raw/`
and `../facebook_trajectory/prodigy.jsonl`.

`neighbor_matching.jsonl` contains the native Facebook pretext trajectory,
including the shared random-initialization step 0.
