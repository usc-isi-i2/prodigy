# Midterm decoder test

Node + fixed 10-neighbor LP, disjoint context. Three fresh source-training runs: raw dot, dot plus bias initialized at -log(5), and positive learned scale times dot plus bias. Same seed-0 encoder initialization, pair generators, split, optimizer and stopping protocol. AdamW updates decoder parameters alongside the encoder. Validation every 2,000 steps, patience 3, safety cap 100,000; all stopped on plateau.

Held-out uniform 1:5 pairs are identical to the existing Midterm evaluation. No target post-hoc calibration.

| Decoder | Test BCE | Test AUC | Best step |
|---|---:|---:|---:|
| Raw dot | 0.61773 | 0.95522 | 30,000 |
| Dot + bias | 0.10782 | 0.98224 | 18,000 |
| Learned scale + bias | 0.10737 | 0.98201 | 18,000 |

Constant p=1/6 BCE: 0.45056. Adding bias lowered held-out BCE about 82.5%; adding scale gave negligible additional benefit on this graph. Learned bias was about -6.83; learned scale in the affine arm was 2.73. This supports a decoder parameterization bottleneck, not failure of BCE to penalize negatives. It does not establish the same effect on every graph or cross-graph transfer; one graph, one seed. Training also changed representation quality, as indicated by increased AUC.

Source revision 32af7e9, branch codex/decoder-bias-test. Tucker worktree /dataMeR1/phil/gfm/mixture-scaling-decoder-test, state/midterm_s0, tmux decoder-test. W&B offline only. Existing transfer matrices were not replaced.
