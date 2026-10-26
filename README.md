# Pericyte-Inspired Cooperative Resource Homeostasis for Reliable Spiking Neural Networks

Research code for DVS128 Gesture classification under a finite, synthetic local compute supply. A three-stage Conv–BN–LIF Backbone always produces Low logits and four regional spike activities. When a regional request is served, an optional High correction is physically executed; otherwise its work is skipped. PIRC combines causal activity-history demand prediction, slow local demand/supply feedback, adaptive donor reserve protection, and bounded sharing between adjacent regions.

The resource tokens, regional stores, and sharing edges are computational abstractions, not a physiological simulation. No dataset, checkpoint, cache, log, or result table is included.

## Contents

- `local_pirc/model.py`, `compact_high.py`: LIF Backbone, original local convolutional High path, and causal accumulated-state Compact High path.
- `local_pirc/prepare_data.py`, `train.py`: DVS128 event binning and shared Backbone training.
- `local_pirc/outage_sweep.py`, `predictive_online.py`: matched original requests, synthetic supply, and causal activity-history predictor.
- `local_pirc/local_sharing.py`, `unified_system.py`: finite reserves, safe adjacent transfers, delayed allocation, PIRC and mechanism ablations.
- `local_pirc/seven_baselines.py`: fixed allocation, instant gating, EMA gating, reserve threshold, balanced sharing, predictive reserve, and PIRC under one original High checkpoint.
- `local_pirc/refine_head_features.py`, `refine_head_adapt_v3.py`, `compact_high_features.py`, `compact_high_train.py`: frozen-Backbone High-head training.
- `local_pirc/pirc_only_compact_eval.py`, `scripts/release_ablations.py`: mixed-head system comparison and matched-head ablations.
- `local_pirc/final_eval.py`: physical sparse High execution and GPU timing helper.
- `configs/resource_controller.json`: frozen controller coefficients used for the published exploratory comparison.

## Environment and data

Use Python 3.11 and a CUDA-capable PyTorch installation for training and the physical-execution audit:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Download the **official IBM DVS128 Gesture** archive and prepare it for SpikingJelly's `DVS128Gesture(root=..., data_type='event')` loader. The official train/test recording lists and annotations must be retained. The original IBM archive is available from the [IBM DVS128 Gesture distribution](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794); the archive used during development had MD5 `8a5c71fb11e24e5ca5b11866ca6c00a1`. Point `--root` to the SpikingJelly dataset root; it is not bundled here. Each recording is converted to 16 equal-duration ON/OFF event bins at 64×64, with no frame repetition.

All commands below run from the repository root. Change `cuda:0` to your visible GPU. Output paths are relative and ignored by Git.

## Reproduction sequence

**1. Prepare official training recordings and train the shared Backbone.** The script creates a fit/validation partition from the official training recordings; neural states reset between clips.

```bash
.venv/bin/python -m local_pirc.prepare_data --root data/dvsgesture --output data/prepared --split train
.venv/bin/python -m local_pirc.train --data data/prepared --out results/local_seed42 --device cuda:0 --seed 42
.venv/bin/python -m scripts.prepare_caches --split fit --device cuda:0
.venv/bin/python -m scripts.prepare_caches --split validation --device cuda:0
.venv/bin/python -m scripts.fit_predictor
```

**2. Fit the original High correction head and the Compact High head using only training recordings and validation selection.** The Backbone remains frozen in both stages. `PIRC_DEVICE` controls scripts that train or audit the High heads.

```bash
export PIRC_DEVICE=cuda:0
.venv/bin/python -m local_pirc.refine_head_features --split fit --device cuda:0
.venv/bin/python -m local_pirc.refine_head_features --split validation --device cuda:0
.venv/bin/python -m local_pirc.refine_head_adapt_v3
.venv/bin/python -m local_pirc.compact_high_features --split fit --device cuda:0
.venv/bin/python -m local_pirc.compact_high_features --split validation --device cuda:0
.venv/bin/python -m local_pirc.compact_high_train
```

**3. Prepare the official test split only after model and controller choices are fixed, then evaluate.** Supply seeds, 32 original High requests per clip, capacity 4 per region, four-bin allocation delay, and one-token adjacent transfer budget are identical across policies. The PIRC ablations use the same original High checkpoint as the main seven-method table.

```bash
.venv/bin/python -m local_pirc.prepare_data --root data/dvsgesture --output data/prepared --split test
.venv/bin/python -m scripts.prepare_caches --split official_test --device cuda:0
.venv/bin/python -m local_pirc.refine_head_features --split official_test --device cuda:0
.venv/bin/python -m local_pirc.compact_high_features --split official_test --device cuda:0
.venv/bin/python -m local_pirc.seven_baselines
.venv/bin/python -m scripts.release_ablations --split validation
.venv/bin/python -m scripts.release_ablations --split official_test
.venv/bin/python -m local_pirc.pirc_only_compact_eval
```

Key outputs are `results/seven_baseline_head_v3/{validation,official_test}_summary.csv`, `results/release_ablations/{validation,official_test}.csv`, and `results/pirc_only_compact_high/{validation,official_test}_mixed_model_table.csv`. These scripts refuse to overwrite their key outputs. `pirc_only_compact_eval` also performs a small physical GPU sparse-execution audit and writes its timing; it requires CUDA.

## Metrics and interpretation

RCIR is interrupted original High requests divided by all original High requests. Outage service is served original requests during zero-supply bins divided by all original requests during those bins. Executed MACs count the always-on Backbone and physically served High corrections; they do not include CPU controller arithmetic or imply GPU latency. Classification accuracy is the fraction of correctly classified clips. For the Compact High mixed-head table, MACs use the Compact High cost for PIRC and the original High cost for the six frozen baselines.

The seven-method table isolates scheduling with one shared original High checkpoint. The Compact High table compares **different High heads** and therefore measures an end-to-end system trade-off, not a PIRC-only scheduling advantage. The controller coefficients in `configs/resource_controller.json` come from historical exploratory selection affected by a class-ordered first-256-recording subset; that subset contained only four classes. Both validation and official test were inspected during earlier development. New training can produce different checkpoints and numerical results. Do not represent the included protocol as a new blind test or tune on official test labels.

The resource simulation keeps reserves and delayed supply across a seeded clip stream. The activity-history predictor resets its neural history at each clip; Low/High LIF state also resets between clips. Policies receive identical original requests and exogenous supply within each split and seed.
