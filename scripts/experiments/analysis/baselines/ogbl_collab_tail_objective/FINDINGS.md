# Extreme-tail ranking objective: rejected under the frozen rule

Six required cells completed. BCE plus fixed coefficient1 top50 pairwise softplus
lost0.6441 percentage points in mean validation-selected2018 Hits@50 relative to
matched BCE. Two of three seeds worsened, failing the all-positive/mean+0.5pp gate.
No earlier-year expansion, final refit, or2019 scoring was performed.

| Seed | BCE best step / Hits@50 | Tail best step / Hits@50 | Difference pp | Rescued / lost positives |
|---|---|---|---|---|
|0|150 /68.7155%|1800 /68.7304%|+0.0150|1037 /1028|
|1|100 /69.0250%|350 /67.7252%|−1.2998|751 /1532|
|2|100 /68.9320%|100 /68.2844%|−0.6474|1047 /1436|
|Mean|68.8908%|68.2467%|−0.6441| |

The loss does substantially change training dynamics: final validation performance
is much less degraded. At step2000, BCE averages55.6615% validation versus66.9807%
for tail. Same-year probe means are66.9180% versus70.7409%, respectively, whereas
seen-negative training means are99.9133% versus82.9976%. Thus less fitting of the
fixed negative panel can coexist with worse validation-selected peak performance.
A flatter curve and improved final probe are not sufficient evidence of a better
candidate. The probe deliberately uses training positives; it is not a benchmark
or independent forward-year evaluation. Seed0's9-positive net gain is near-total
cancellation of1037 rescues and1028 lost existing hits.

## Evidence contract and checks

Producing revision87bf66f80316d45943abeb5acdf5573dd24248af. Unchanged architecture,
Adam, fixed2017 fresh-negative panel,2000 updates and50-update evaluation cadence.
Objective: sampled balanced BCE plus mean softplus(1+n−p) over all2048 sampled
positives × top50 training negatives, refreshed every10 updates. No positive
boundary oversampling, loss-coefficient sweep, or architecture search. Controls and
treatments use exactly matching positive/uniform-negative/mining-slot draw hashes;
model-dependent hard-negative identities can differ. Earliest maximum standalone
2018 Hits@50 selects each checkpoint; no fusion selector or test input is used.

All240 required history rows exist. All120 BCE control values reproduce original
archived standalone curves exactly (declared tolerance1e−6, observed maximum0).
All six selected checkpoints replay exactly; scores match official OGB evaluator.
Laptop independent full-sort strict-threshold evaluation matches saved metrics;
all result-referenced file hashes and earliest-max selections validate. Exactly
496385 neural parameters; conservative full inference-state bound496448 scalars.

All six runs used GPU1 in isolated Tucker worktree
/dataMeR1/phil/gfm/prodigy-collab-h2-tail, branch codex/collab-h2-tail, tmux
collab-h2-tail. Runtime /dataMeR1/phil/gfm/ogbl_collab_compact_joint/h2_tail_v1.
Fresh process per cell:57.2–60.5 seconds measured loop time, peak allocated GPU
memory2530213888 bytes; about4GB process footprint observed. Offline W&B IDs and
directories, source/panel hashes, complete curves and checkpoint hashes are in data/.
Large score arrays and checkpoints remain private runtime artifacts.

This is one predeclared loss variant rejected on one heavily explored validation
year, not proof that all ranking losses fail. Three optimization seeds are not
three independent datasets. Extensive prior2019 test exploration and negative
leakage in earlier research remain disclosed. These experiments neither establish
a HyperFusion win nor resolve leaderboard eligibility.
