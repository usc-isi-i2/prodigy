# Second uniform sample: twenty additional validation misses

Same retained fresh-negative joint baseline: seed 0, update 150, 2018 validation. Uniform sampling without replacement from the **18,777 remaining misses**, explicitly excluding the first 20. PCG64 seed **20260915**, first draw, original draw order. No case selection after inspecting attributes.

All 20 are true collaborations missed under Hits@50. “Negatives ahead” counts negative scores at least as high as the positive score; a hit needs fewer than 50. Inactive means no graph-recorded activity in 2017. Similarity is the supplied author-feature cosine, not a verified semantic relationship. Historical collaborator counts are distinct graph neighbors. A zero cached length-three feature is not a general absence-of-paths claim, especially for common-neighbor pairs where this feature is not used as a path census. Interpretations below are descriptive, not causal feature attributions.

| # | Node pair | Negatives ahead | Discussion |
|---|---|---:|---|
| 1 | 51928–219154 | 81 | Histories of 3 and 61 collaborators, one inactive endpoint, and a length-three signal. Similarity 0.955. Relatively close to the boundary, but none of the compared scorers recovers it. |
| 2 | 204917–61677 | 224 | Substantial histories (69 and 110), similarity 0.959, and the largest cached length-three feature in this batch. Seed 2 and warm-only seed 0 recover it; frozen AA does not. A recoverable indirect-connectivity case. |
| 3 | 154599–101468 | 62 | Both active, similarity 0.965, but no common collaborator or cached length-three signal. Only 62 negatives ahead: a close miss despite favorable activity and similarity. |
| 4 | 65600–77438 | 13,322 | Histories of 1 and 10 collaborators, one inactive endpoint, no short-path signal. A deep sparse-history miss shared by all comparators. |
| 5 | 123391–205443 | 4,027 | Both active, but only 1 and 6 historical collaborators and similarity 0.873. No short-path signal. Recent activity does not compensate for the limited observed history here. |
| 6 | 110281–228737 | 35,571 | Histories of 1 and 4, similarity 0.777, one inactive endpoint, no short-path signal. The deepest miss in this batch; not plausibly fixed by a small cutoff adjustment alone. |
| 7 | 2668–150026 | 233 | Both active, similarity 0.979, and substantial histories (55 and 111), yet no short-path signal. All comparators miss it: strong individual profiles do not identify the correct pair. |
| 8 | 12288–143839 | 94 | Both active, exactly 13 historical collaborators each, similarity 0.959, no short-path signal. A relatively close miss that all comparators share. |
| 9 | 208942–14557 | 448 | One common collaborator, histories of 3 and 92, both inactive. The 2018-calibrated AA reference recovers it, but frozen AA does not. It must not be counted as a frozen-AA rescue. |
| 10 | 115360–152285 | 1,010 | High similarity (0.957), histories of 13 and 45, one inactive endpoint, no short-path signal. Still below 1010 negatives; similarity alone is insufficient. |
| 11 | 12278–154774 | 96 | One common collaborator despite small histories (2 and 5), with one inactive endpoint. Warm-only seed 0 recovers it; frozen AA and the other baseline seeds do not. |
| 12 | 197109–221189 | 410 | Histories of 24 and 64, similarity 0.941, one inactive endpoint, and a small length-three signal. Indirect support exists but none of the compared scorers recovers it. |
| 13 | 192123–129763 | 2,822 | Both active, histories of 24 and 78, similarity 0.933, no short-path signal. A deep miss despite substantial history and recent activity; sparse endpoints are not the only problem. |
| 14 | 47438–112249 | 1,017 | Histories of 3 and 8, one inactive endpoint, no short-path signal. Another sparse-history miss, just beyond 1000 negatives ahead. |
| 15 | 164833–35586 | 6,621 | Histories of 1 and 5, one inactive endpoint, similarity 0.872, no short-path signal. Deep miss with limited local evidence. |
| 16 | 194150–54240 | 14,168 | Histories of 3 and 14, one inactive endpoint, similarity 0.858, no short-path signal. Far below the boundary and missed by every comparator. |
| 17 | 230986–35533 | 83 | Two common collaborators, similarity 0.980, histories of 23 and 47, both inactive. Frozen AA, seed 2 and warm-only seed 0 all recover it. Existing information demonstrably supports a successful prediction. |
| 18 | 82616–185641 | 507 | Similarity 0.957, histories of 11 and 14, one inactive endpoint, and a small length-three signal. All comparators miss it; the indirect evidence is not enough at their respective cutoffs. |
| 19 | 213580–14943 | 588 | Histories of 6 and 7, one inactive endpoint, similarity 0.908, and a length-three signal. Short indirect connectivity exists, but all comparators miss it. |
| 20 | 188935–158992 | 67 | One common collaborator, histories of 16 and 133, one inactive endpoint, similarity 0.940. Frozen AA recovers this close miss; neither other baseline seed nor warm-only seed 0 does. |

This batch again contains **16/20 pairs without a common collaborator**, **8/20 below more than 1,000 negatives**, and **20/20 novel collaborations**. Fifteen have at least one inactive endpoint; nine have an endpoint with at most five historical collaborators. These overlapping counts describe this sample, not separate categories.

Frozen AA recovers **2/20** (17 and 20), versus 4/20 in the first batch. Case 9 is recovered only by the separately 2018-calibrated AA reference, so it is not interchangeable with the frozen scorer. Other baseline seed 2 recovers cases 2 and 17; warm-only seed 0 recovers 2, 11 and 17. Individual rescues do not undo the negative full-panel warm-only result or establish a successful ensemble.

Across the **40 distinct sampled misses**, 32 lack a common collaborator, 16 score below more than 1,000 negatives, and frozen AA recovers six. Six cases have only 50–70 negatives ahead. This reinforces a mixed diagnosis: some useful structural predictions are lost, while many missed novel pairs lack observed short-path support. Cases 3, 7, 8 and 13 also show that activity and substantial histories alone do not resolve the problem. The sample does not establish whether better use of current features or additional information would help most.

Input and score hashes were verified; common-neighbor and degree counts were checked directly against the cached historical graph. No new model training, test access, or rule fitting. Validation has been repeatedly explored; prior test exploration remains disclosed.
