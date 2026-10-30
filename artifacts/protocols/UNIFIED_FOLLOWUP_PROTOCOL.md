# Fit-only follow-up to the unified system experiment

The existing checkpoint, DVS128 Gesture split, 32 original High requests per clip, `d64_v50` capacity-4 external supply, four-bin delay and adjacent transfer rule are unchanged. The previous validation result is used for diagnosis only; controller selection below uses fit-subject clips and fit supply seeds, never official test labels. All attempted candidates and their fit outcomes are saved.

Mechanism diagnosis: the original adaptive donor safety mean was 0.63 token/region versus 1 for fixed safety; it transferred more tokens in normal periods and had less reserve just before outage. The full history forecast predicted four-step demand more accurately than current-only features, but its unit-weight use in safety/allocation did not improve task outcomes. These observations motivate testing a fixed safety floor and forecast weights, not changing the resource scenario.

Stage 1 is a fixed 36-candidate grid on the first 256 fit-subject clips and seeds 701–703:
- safety floor {0.5, 1.0, 1.5};
- gap-feedback gain {2, 4};
- causal forecast allocation gain {0, 1, 2};
- forecast donor-safety gain {0, 1};
- rho 0.95 and reserve allocation gain 2 fixed.

The traditional comparator is the existing fixed one-token safety sharing controller under the same fit clips and seeds. A candidate is feasible only if it has at least the comparator's mean Accuracy, no higher mean RCIR, and no lower mean outage service on these fit traces. Select feasible candidates by greatest minimum normalized improvement among the three metrics, then mean improvement, then lexicographic parameters. If none is feasible, keep the Pareto set and diagnose the binding constraint; do not evaluate all candidates on the validation split.

Any second modification must be motivated by saved fit-only evidence, logged as a separate attempt, and selected on fit clips. The final selected controller will be evaluated once on the same 252 validation clips and seeds 731–740 as the unified comparison, with matched original requests and external supply. Prior validation use makes this exploratory rather than a fresh holdout. Record all attempts and report failure if the three criteria are not jointly achieved. Do not use official test labels for fitting or selection.

Stage 1 returned zero feasible candidates. It showed that a flat safety floor close to the fixed comparator was strongest, while adding a forecast directly to the delayed allocator did not help. Stage 2 therefore changes only the action mapping: use a causal EMA of observed total supply to raise donor safety when supply has recently increased, and use the already-trained full activity-history forecast for donor protection and optionally transfer priority. This is a scenario-specific hypothesis, not advance knowledge of the next outage.

Stage 2 is a fixed 48-candidate fit-only grid on the same 256 fit clips/seeds:
- base donor safety {0.25, 0.5, 0.75, 1.0};
- burst safety gain {0, 1, 2} applied to max(EMA_0.9(observed supply)−2, 0);
- history-forecast donor-safety gain {0.5, 1.0};
- transfer-priority forecast gain {0, 1};
- local demand-minus-incoming EMA gain 0.5, rho 0.95, reserve allocation gain 2, direct forecast allocation gain 0 fixed.

All candidates thus use the activity-history forecast with nonzero safety gain. The same three feasibility conditions and ranking apply. Validation remains unopened for selection. If none meets all three on fit, record the negative result and stop this controller-only search rather than repeatedly adjusting the same validation scenario.

Stage 2 also returned zero feasible candidates. A distinct structural attempt, Stage 3, addresses the measured ~13% supply spill: the delayed allocator estimates local reserve when a new allocation can arrive. It uses only current reserve, already queued allocation weights, an EMA of observed supply, and the fitted causal four-step demand forecast. Projected reserve is clip(current reserve + queued expected supply − 4×forecast demand, 0, capacity); allocation favors projected headroom and current requests. The traditional fixed-safety comparator remains unchanged. This is a new projected-headroom action rule rather than further threshold tuning on validation results.

Stage 3 fit-only grid, same first 256 fit-subject clips and seeds 701–703: current-request weight {0,2,4}, projected-headroom weight {2,4,8}, base donor safety {0.5,0.75,1}; other settings fixed at gap gain 0.5, burst gain 2, history safety gain 1, direct forecast allocation gain 1, transfer-priority gain 0. The same three feasibility conditions and ranking apply. If no candidate meets all three on fit, do not run a new validation parameter search. Full prior validation and both failed grids remain part of the record.
