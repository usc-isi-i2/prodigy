# Disjoint-context neighbor LP retraining

Correct the positive-edge shortcut in fixed-neighbor training: partition the
original 70% training edge pool in half, using seed+15485863. Approximately 35%
of all unique undirected edges form a fixed context graph; the other 35% are
positive supervised training pairs. The two are disjoint. Validation and final
test edges retain the original 15%/15% partition exactly. Negatives still exclude
all known edges, including context, supervision, validation and test.

A fixed sample of up to ten distinct context neighbors is averaged and
concatenated with the node feature vector. The same context is used for model
training, validation and target evaluation. Isolates have zero neighbor context.

To preserve the exact existing final comparison pairs, degree matching uses the
original 70% training pool; it does not change the model's input context. Verify
cached test/calibration endpoints and labels against the original experiment.
Architecture, optimizer, batch size, stopping rule and seed remain unchanged.
This changes BOTH context density and supervised positive coverage, so it is not
a pure attribution of any performance change to shortcut removal alone.

Use a fresh state root and dedicated Tucker worktree. `run_mini_lp_tucker.sh`
on this branch runs only the corrected neighbor stage. Offline W&B remains the
default; explicit sync uploads the completed result after verification.
Original experiments and gallery are retained as historical comparisons.
