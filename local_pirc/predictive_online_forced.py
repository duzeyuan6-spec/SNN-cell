"""Report best fit-selected nonzero forecast controls as a transparent sensitivity."""
import csv
import json
from pathlib import Path
import numpy as np
from .outage_sweep import SCENARIOS, stream_inputs
from .predictive_online import (CAPS, VAL_CLIPS, VAL_SEEDS, accuracy, predict,
    save_csv, simulate_online)


def main():
    root = Path('results/predictive_online_seed42')
    out = root / 'forced_forecast_validation.csv'
    assert not out.exists(), 'Refuse overwrite'
    fit = list(csv.DictReader((root / 'fit_candidates.csv').open()))
    selected = {}
    for method in ('predictive_threshold', 'predictive_pirc'):
        candidates = [r for r in fit if r['method'] == method
            and json.loads(r['params'])['forecast_gain'] > 0]
        row = min(candidates, key=lambda r: (float(r['objective']), r['params']))
        selected[method] = json.loads(row['params'])
    (root / 'forced_forecast_chosen.json').write_text(json.dumps(selected, indent=2))
    cache = np.load('results/local_seed42/controllers/validation_cache.npz')
    coef = np.array(json.loads((root / 'predictor.json').read_text())['coef'])
    rows = []
    for cap in CAPS:
        for scenario in SCENARIOS:
            for seed in VAL_SEEDS:
                inp = stream_inputs(cache['rate'][:VAL_CLIPS], scenario, seed)
                pred = predict(inp, coef)
                for method, params in selected.items():
                    mask, st, _ = simulate_online(inp, pred, method, params, cap)
                    rows.append({'capacity': cap, 'scenario': scenario, 'seed': seed,
                        'method': method + '_forced', 'accuracy': accuracy(cache, mask), **st})
            print('FORCED', cap, scenario, flush=True)
    save_csv(out, rows)

if __name__ == '__main__':
    main()
