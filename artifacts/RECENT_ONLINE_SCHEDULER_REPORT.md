# 2025 online resource schedulers on DVS128 Gesture

This comparison evaluates **only two new literature-inspired adapters**. PIRC was not rerun, as requested. The numbers below are our SNN adaptations and are **not results reported by the original papers**.

| Split | Method | Accuracy (%) ↑ | RCIR (%) ↓ | Outage service (%) ↑ | Executed MAC/clip (M) ↓ |
|---|---|---:|---:|---:|---:|
| Validation | DSM 2025 adapted | 73.333 | 29.539 | 7.617 | 171.479 |
| Validation | ECSS 2025 adapted | 72.302 | 29.258 | 7.637 | 171.486 |
| Official test | DSM 2025 adapted | 79.514 | 29.334 | 8.247 | 171.484 |
| Official test | ECSS 2025 adapted | 78.958 | 29.128 | 8.333 | 171.489 |

DSM denotes [Dynamic Service Scheduling and Resource Management in Energy-Harvesting Multi-access Edge Computing](https://arxiv.org/abs/2510.27317) (2025). Its forecasted supply–demand classification and surplus-to-deficit migration were mapped to causal adjacent resource-token sharing and supply allocation. The paper's DVFS and DAG service migration cannot be executed by this fixed, indivisible SNN High module. ECSS denotes [An energy-conscious scheduling framework for serverless edge computing in IoT](https://link.springer.com/article/10.1186/s13677-025-00780-7) (2025). Its residual-energy/workload pre-scheduler and sufficient/constrained/inadequate regimes were mapped to local token allocation and adjacent sharing. The original framework's external helper nodes and serverless functions are unavailable here. Neither adapter should be described as an exact paper reproduction.

## Adapter parameters and decisions

| Adapter | Validation grid | Selected parameter | Causal decision used in this SNN |
|---|---|---:|---|
| DSM | safety margin {0, 0.5, 1.0} | 0.0 | Forecast the four-bin resource balance per region, protect donors below the projected margin, transfer at most one token to a currently deficient adjacent region, and direct delayed supply toward predicted deficits. |
| ECSS | energy-regime threshold {0.25, 0.5, 1.0} | 0.5 | Compare mean current available resources with mean current requests, switch between sufficient, constrained and inadequate allocation rules, and share adjacent tokens subject to the regime's donor protection. |

For DSM, the projected balance is `available + queued_supply − 4 × forecast_demand`. Its donor protection is `clip(safety_margin − projected_balance, 0, 4)`; allocation logits are `4 × current_request + 2 × projected_deficit + 1.5 × reserve_gap/4`. For ECSS, the regime margin is `mean(available) − mean(current_request)`; the regimes use donor protection 0, `threshold/2`, and 0, and reserve-gap logit coefficients 1, 2, and 3. Both use a causal supply EMA with coefficient 0.9 where needed, the same existing local-demand predictor, softmax delayed supply allocation, and the same bounded adjacent transfer function. Controller arithmetic is excluded from the reported network MAC count. Full code is in `local_pirc/recent_online_scheduler.py`.

## Matched protocol

All runs use DVS128 Gesture, the same frozen Low backbone (`results/refine_adapt_subject_seed42/best.pt`, SHA-256 `7932eecd3c0192d383db1af8ea08a028ea3d373836c3b7301274db9c67b8e976`) and Compact High head (`results/compact_high_candidate_v2/best.pt`, SHA-256 `717df1cdc60c13e93a572d22f542df7f7df25ecb3637b6798e85872fc9c50de2`). The scenario is `d64_v50`: 4 regions, reserve capacity 4 per region, initial reserve 2, 4-bin allocation delay, one adjacent transfer token per bin, and 32 original High requests per clip. Both methods see identical exogenous supply and requests for each seed. Validation seeds are 1201–1205 (252 clips); official-test seeds are 1301–1305 (288 clips).

One parameter per adapter was selected from a three-value grid using validation resource traces 1201 and 1202 and the preset objective `0.5 × RCIR − 0.5 × outage service`. No class labels entered this selection; official-test labels were used only for scoring. The selected DSM safety margin was 0.0; the ECSS regime threshold was 0.5. The official test split had been inspected during earlier project iterations, so these results are exploratory rather than an untouched confirmatory test.

Unit testing checked token conservation, causal dependence, request count, and equal request/supply hashes. A 16-clip GPU execution audit for each adapted scheduler agreed with cached logits to `3.81e-6` maximum absolute difference and had no classification mismatches. Its first-pass latency was affected by cold-start effects and is not used for method ranking.

DSM obtained 0.556 percentage points higher official-test accuracy, while ECSS had 0.206 percentage points lower RCIR and 0.087 percentage points higher outage service. Their executed MACs differ by less than 0.005 M per clip. Neither dominates the other across all reported metrics. No PIRC superiority claim follows from this two-method-only run.

Archived raw per-seed data: `artifacts/results/recent_online_schedulers_2025/{validation,official_test}_raw.csv`; summaries: `artifacts/results/recent_online_schedulers_2025/{validation,official_test}_summary.csv`; selection audit: `artifacts/results/recent_online_schedulers_2025/validation_calibration.csv`; protocol: `artifacts/results/recent_online_schedulers_2025/manifest.json`; GPU audit log: `artifacts/results/recent_online_schedulers_2025/physical_audit.log`.

Reproduce from the project root: `.venv/bin/python -m pytest -q tests/test_recent_online_scheduler.py` and `.venv/bin/python -m local_pirc.recent_online_eval`. GPU audit: `.venv/bin/python -m local_pirc.recent_online_physical_audit cuda:0`. The evaluation refuses to overwrite existing summary files; use a new result directory for a fresh run.

Source experiment commit: `bbb99c4` on the server branch `recent-online-scheduler-baselines-2025`. This release copies the code and raw outputs without importing that research branch history.
