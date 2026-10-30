# Archived experimental artifacts

The original archived experiments were copied from the source project at `ef24896`. The two recent online-scheduler adapters and their outputs were copied from source experiment commit `bbb99c4`. The original experiment directories and names are preserved below `artifacts/results/`; newly copied CSVs were normalized to LF line endings with unchanged cells. `SHA256SUMS` records the hash of every included file. The [reproduction index](REPRODUCIBILITY.md) maps the archived experiments to their parameter grids, selected configurations, metrics and limits. This is a selected archive of completed results relevant to the CSCWD system comparison, not a dataset or a new experimental run.

| Directory | Contents | Data / interpretation |
| --- | --- | --- |
| `controller_selection/` | Fit-only Stage 1–3 PIRC parameter grids, failed candidate sets, selected Stage-3 configuration and lock | Selection used a class-ordered first-256 fit subset with only four classes; not an independent generalization estimate. |
| `local_seed42/` | Shared DVS128 Gesture Backbone checkpoint, fit/validation split, training log and initial hardware audit | Backbone used by the later High-head work; initial audits are a separate experiment. |
| `refine_adapt_subject_seed42/` | Original High-head checkpoint, configuration, training log, validation and official-test summaries and physical audits | The seven scheduling policies use this same High checkpoint. |
| `compact_high_candidate_v2/` | Compact High-head checkpoint, training history and log | Used only by the later Compact High PIRC system comparison. |
| `seven_baseline_head_v3/` | Seven-method main table, per-seed raw data, split summaries, strong-baseline paired deltas, physical audit and logs | Fixed Allocation, Instant Gating, EMA Gating, Reserve Threshold, Balanced Sharing, Predictive Reserve and PIRC. Same checkpoint, 32 original requests per clip, capacity 4, initial reserve 2, four-bin allocation delay, one adjacent transfer token per bin, and matched supply seeds within each split. `paper_main_table.csv` includes Accuracy, RCIR, outage service, executed network MACs, High calls and overflow. |
| `recent_online_schedulers_2025/` | Two 2025 literature-inspired online resource-scheduler adapters: validation calibration grid, per-seed validation and official-test rows, summaries, manifest, GPU audit and run log | Same Compact High checkpoint, original requests and supply seeds for both adapters; **PIRC was not rerun**. See [`RECENT_ONLINE_SCHEDULER_REPORT.md`](RECENT_ONLINE_SCHEDULER_REPORT.md) for adaptation details and limitations. |
| `pirc_only_compact_high/` | Compact High mixed-model comparison, raw PIRC rows, timing diagnostic and logs | PIRC uses the Compact High checkpoint; the other six rows retain the original High checkpoint. This is an end-to-end mixed-model comparison, **not** an isolated scheduler comparison. |
| `unified_system_seed42/` | Main system table, mechanism ablation table, raw and paired results, physical audit | Earlier held-out-training-subject validation experiment using `local_seed42/best.pt`. It is a different model/protocol from the seven-method table. |
| `demand_diagnostic_seed42/` | 16/32-step risk prediction AUPRC/AUROC tables, fitted coefficients, policy-selection grid and per-scenario/per-seed deployment results | Validation-only mechanism diagnostic; compare reserve-only and activity-history variants within the same scenario. |
| `local_sharing_seed42/` | RCIR, outage service, CRSR and executed-MAC sharing diagnostics; fit grid, selected parameters, per-seed raw rows and group details | Earlier exploratory scenarios; not matched to the final seven-method table. |
| `predictive_online_seed42/` | Predictive resource control diagnostic, fit grid, selected parameters, predictor, per-seed raw rows, paired and overhead tables | Earlier exploratory scenarios; not matched to the final seven-method table. |

Checkpoint SHA-256:

- Shared Backbone `local_seed42/best.pt`: `240d3305ec7876999d660af9051479cdcb4e51e8f6f943d4c712e3100aed811d`
- Original High `refine_adapt_subject_seed42/best.pt`: `7932eecd3c0192d383db1af8ea08a028ea3d373836c3b7301274db9c67b8e976`
- Compact High `compact_high_candidate_v2/best.pt`: `717df1cdc60c13e93a572d22f542df7f7df25ecb3637b6798e85872fc9c50de2`

To place pretrained weights at the paths expected by the scripts, from the repository root:

```bash
mkdir -p results/local_seed42 results/refine_adapt_subject_seed42 results/compact_high_candidate_v2
cp artifacts/results/local_seed42/best.pt results/local_seed42/best.pt
cp artifacts/results/refine_adapt_subject_seed42/best.pt results/refine_adapt_subject_seed42/best.pt
cp artifacts/results/compact_high_candidate_v2/best.pt results/compact_high_candidate_v2/best.pt
```

The official DVS128 Gesture recordings, derived feature caches and event arrays are not included. The README in the repository root gives data preparation and full reproduction commands. The original seven-method validation and official-test supply seeds are in `seven_baseline_head_v3/manifest.json`; all seven methods face the same original requests and supply within each split. Official-test labels were not used for the final table's tuning, but both splits had been inspected during earlier development, so these numbers are not a fresh blind test.

`physical_audit.csv` in the seven-method and Compact High directories records a small, single-pass GPU execution check. Warm-up and pass-order variation make those timings diagnostic only; they do not support a latency ranking. The earlier `local_seed42/controllers/final/hardware_randomized.csv` contains repeated timings for a different model/protocol and must remain separate. Executed MACs in the main tables are the more comparable computation measure and exclude CPU controller arithmetic.
