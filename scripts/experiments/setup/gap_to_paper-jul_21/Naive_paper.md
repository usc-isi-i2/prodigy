# Naive paper — the open recipe paper

Opening probe, reworded as measurable: NM embeddings carry no decodable adjacency/degree signal (rewire/permute 2×2 + synthetic degree probes) — not "cannot use topology" (intent language), not follower count (profile metadata; regression ran ≈ noise).

## Formulation (fixes "best cannot be proven")
Given D pretraining graphs, E eval graphs, T tasks at matched compute (40k steps), a recipe = one point in {sequential|interleaved} × {cross|within-source episodes} × {proportional|balanced} × {single|rotated objectives}. Claim: our recipe maximizes the min over task families among the 2⁴ recipes, and each lever is individually necessary. Falsifiable; never claims "best".

Open 50M model, 8-way curriculum; frozen-probe eval on 4 cls + 4 reg + sLP; floors in every table (features-only, random-init, raw-degree; AA/CN for sLP).

## BAD → BETTER → BEST
- BAD = three separable failures, one figure each: sequential forgets (per-source curves — the big NEW runs); naive merged sampling has a source-discrimination shortcut (p-sweep: done); proportional weighting starves small domains (midterm .31 vs .417 specialist).
- BETTER: pooled interleaving (field default; the [PretrainStrats] slot = Hu'20 + LM temperature sampling) fixes only forgetting.
- BEST: + within-source episodes (shortcut gone) + balanced (midterm .31→.427, above its specialist) + objective rotation (sLP .76 vs ≤.47 every subset, rewire-certified topological). NOT the small extra head — our E4 multi-head degrades both axes (cls .445); keep as the in-paper negative control: composition beats construction. The corpus×objective inversion goes in as a finding ("corpus-check the objective step"), not a buried caveat.
- Efficiency: don't assert 90%-in-10% — dense ckpt grid at launch, measure "X% of steps to 90% of final", floor-anchored.

## Why (≤2 pages, principle → recipe consequence)
ladder → include the target, don't chase diversity (rung7=rung8) · ablations → single pretexts waste the graph · divergence ρ≈−.9 → source-selection rule · rotation surgery → when the topology lever fires.

## Validate
Eval-seed fix first; 3 seeds on spine arms only; conv-swap GIN/GATv2 (cheap; PNA arm exists); one frozen-recipe run on a never-touched corpus (kills "recipe overfit"). Claim: the recipe ORDERING replicates, not the numbers.

## Why citable (scout, Jul '26)
No open-weights inductive social-graph FM exists (TwHIN transductive/proprietary; Meta's billion-scale closed, names our questions as open). No "FineWeb for graphs"; sequential-vs-interleaved never published as an ablation. ControlG (ICML'26) corroborates scheduling>blending. PRODIGY never released weights — 2026 papers exclude it as a baseline for that reason.

## Contributions
- first open social-graph ICL encoder + recipe + harness (floors, MDE gates, seeding fix)
- three diagnosed failure modes of naive multi-graph pretraining, each with its fix
- the corpus×objective interaction (unreported; qualifies all single-corpus multi-task-SSL rankings)

Budget ~16–24 new runs (race 9–12, objective sweep 4–6, encoder swap 2–4); P0 ≈ 2 wks. Venue: NeurIPS D&B / WWW resource first. Boundary: ml_route owns mechanism, ccs_route owns the atlas; this owns policies, curves, weights.
