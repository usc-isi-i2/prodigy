# TRACE schedule scaling analysis

Status: training in progress.

This leaf tests the predeclared transition from two-source sequential gains to
larger-mixture forgetting under an order-only intervention. It also tests whether a
bounded 100-episode replay schedule preserves the useful continuity of blocked
training without its larger-mixture forgetting, and whether target-side U1 health
tracks schedule effects.

Primary outcomes are fresh-stream held-out accuracy and ROC-AUC on Election-2020,
Ukraine/Russia suspended, TwiBot-20, Hong Kong, and Facebook page-reference. The
original stream is discovery data; it may select a conventional baseline but never
supplies a reported fresh outcome. TRACE schedule routing uses only episode supports,
intermediate predictions, and full-model predictions. The chance-level suspended
target is reported and subject to the fixed `.55` support-competence abstention rule.

The primary scaling interaction is:

`(blocked - interleaved at rung 2) - mean(blocked - interleaved at rungs 3 and 4)`.

Uncertainty for that interaction and for TRACE-health correlations uses a crossed
bootstrap over training seeds and target datasets. The all-target interaction is
primary; an explicitly labeled secondary view excludes the support-incompetent,
chance-level suspended target. Episode streams are fixed across models; they are not
treated as independent checkpoint seeds.
