# Gradient-isolated encoder and solver training

Hypothesis: allowing a learned task solver to shape the encoder can create
source-specific coadaptation. The motivation is readout-dependent schedule
reversals, not a claim that stop-gradient or differentiable ridge is new.

Four objectives on identical source-private episode streams and initialization:
native, joint native+ridge loss, isolated native+ridge loss (native path detached
at U1), and ridge-only. Joint versus isolated changes only gradient access to U1.
Isolated and ridge-only encoder updates must match under controlled randomness.
Ridge uses normalized support/query vectors, lambda1 and fixed logit scale1.
No target tuning of temperature or loss weights. Equal steps are not equal FLOPs;
record training walltime and the extra ridge computation.

First compare blocked and interleaved four-source schedules, seed0,2500 updates.
Smoke all eight arms for20 updates before full training. Use exact original
source-private seeds380100/480100, sampling/config and per-source exposure.
Rerun native in this checkout rather than assume an old checkpoint matches.
Compare actual initial state and consumed-episode audit receipts before claiming
matched training. Code is experimental until integration and Tucker smoke pass.

Frozen target panel: political, Election2020, Facebook page-reference, TwiBot20,
Ukraine/Russia suspended. Report all, with separate four-target supported mean
matching prior work; never choose targets based on the new outcome. Evaluate
both full and U1 readouts. Do not claim success just because U1 improves.

Go: isolated training beats native and joint in panel macro-F1/AUC, and its full
solver beats its own ridge readout with benefits not confined to one target.
Approximately one percentage point merits a three-seed expansion, not a final
significance claim. No-go: if detached solver remains unnecessary, do not market
R2D2-style encoder learning plus an unused decoder as a new architecture.

Launch from this worktree in the Tucker prodigy environment with WANDB_MODE=offline:

```
python -m scripts.experiments.setup.encoder_solver_isolation.run --output log/isolation_smoke --gpu 0
```

Default is dry-run. Add --execute only after checking owned GPU availability;
full mode uses --steps2500 (as separate arguments: --steps 2500) and a new output.
The launcher is sequential on one selected GPU and preserves any failed output.

For downstream classification evaluation, explicitly set
`--encoder_solver_objective native`: the objective flag controls pretraining,
not the deployment readout. Preserve the original training objective in the
checkpoint inventory. Ordinary inference must not recompute the training loss.

`python -m scripts.experiments.setup.encoder_solver_isolation.verify --root <run>`
checks complete checkpoints, objective metadata, exact per-source consumed
examples across all eight arms, and reports isolated/ridge-only encoder tensor
differences. New runs save step-zero weights and the verifier checks exact model
initialization equality across all eight arms. These are mandatory for the full
2500-step experiment; the initial legacy smoke did not capture them.
