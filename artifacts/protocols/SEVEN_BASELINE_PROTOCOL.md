# Seven-method matched resource baseline experiment

Run from the repository root with the frozen subject-held-out High-head-only checkpoint `results/refine_adapt_subject_seed42/best.pt` (SHA256 `7932eecd3c0192d383db1af8ea08a028ea3d373836c3b7301274db9c67b8e976`). Backbone, High convolution and head weights never change. Use all 252 established validation clips and all 288 official test clips. Validation seeds 1201–1205 and official-test seeds 1301–1305 are fixed now; they are separate exogenous supply streams. Both data splits have been used historically, so all new results are exploratory even though these particular supply seeds are new. No tuning using validation or official-test labels.

Common resource environment: DVS128 Gesture, `d64_v50` supply, four local regions, capacity 4 and initial reserve 2 each, delay 4 bins, original top-two High requests at each of 16 bins (32 per clip), one-token maximum adjacent transfer per bin, identical activity, request and supply hashes per seed. Every original request remains in the RCIR denominator even if a method cannot serve it. Execute High only for actually served requests. Network MAC estimates use the same base and extra High operators for all policies; CPU controller arithmetic and GPU latency are separate/unmeasured here.

Methods and fixed parameters, without a new search:
1. Fixed Allocation: uniform 1/4 supply routing, no sharing.
2. Instant Gating: current local low-path spike activity routes supply with softmax gain 4, no sharing; no original request is suppressed.
3. EMA Gating: causal continuous-stream activity EMA rho 0.9 routes supply with softmax gain 4, no sharing; no original request is suppressed.
4. Reserve Threshold: conventional request-plus-reserve supply allocator from `traditional_share`, adjacent transfer with fixed donor reserve 1. This is the strong fixed-safe sharing comparator.
5. Balanced Sharing: same conventional supply allocator, adjacent transfer with donor reserve 0.
6. Predictive Reserve: existing `predictive_share` with fixed donor reserve 1, causal demand predictor fitted on training data, and the same frozen Stage-3 configuration.
7. PIRC (Ours): existing `pirc_full` with adaptive local safe reserve, slow demand/supply feedback, causal forecast and adjacent sharing, same frozen Stage-3 configuration.

All four methods with sharing have the same transfer graph and upper bound; the first three choose not to share by their definitions. The existing PIRC configuration was selected historically from a class-skewed first-256 fit subset and is a provenance limitation. This experiment compares the locked implementations; it does not certify globally optimal tuning for any baseline. Report every method on both splits, Accuracy, RCIR, outage service, executed network MAC/clip, High calls/clip, overflow, and supply/request equality. Physical replay should verify a subset of each mask. Do not hide strong conventional or predictive baselines if they rival PIRC.
