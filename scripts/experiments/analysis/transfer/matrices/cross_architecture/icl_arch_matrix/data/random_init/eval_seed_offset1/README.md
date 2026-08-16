# VISION alternate-episode random-initialization check

`vision.jsonl` repeats the seed-0 untrained VISION evaluation with
`--eval-episode-seed-offset 1`. Model initialization and the 10-shot protocol are
unchanged; only the deterministic evaluation episodes differ. All four episode
fingerprints differ from `../raw/vision.jsonl`.

The result was produced on Tucker from commit `7dbefa5` using the pinned VISION
upstream revision recorded in `setup/icl_arch_matrix/upstream_pins.json`.
