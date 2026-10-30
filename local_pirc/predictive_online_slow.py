"""Fit-selected slow-feedback sensitivity, rho at least 0.9."""
import csv
import json
from pathlib import Path
import numpy as np
from .outage_sweep import SCENARIOS,stream_inputs
from .predictive_online import CAPS,VAL_CLIPS,VAL_SEEDS,accuracy,predict,save_csv,simulate_online


def main():
    root=Path('results/predictive_online_seed42')
    out=root/'slow_feedback_validation.csv'
    assert not out.exists(),'Refuse overwrite'
    fit=list(csv.DictReader((root/'fit_candidates.csv').open()))
    candidates=[r for r in fit if r['method']=='predictive_pirc'
        and json.loads(r['params'])['rho']>=.9]
    best=min(candidates,key=lambda r:(float(r['objective']),r['params']))
    params=json.loads(best['params'])
    (root/'slow_feedback_chosen.json').write_text(json.dumps({'params':params,'fit_objective':float(best['objective'])},indent=2))
    cache=np.load('results/local_seed42/controllers/validation_cache.npz')
    coef=np.array(json.loads((root/'predictor.json').read_text())['coef'])
    rows=[]
    for cap in CAPS:
        for scenario in SCENARIOS:
            for seed in VAL_SEEDS:
                inp=stream_inputs(cache['rate'][:VAL_CLIPS],scenario,seed)
                pred=predict(inp,coef)
                mask,st,_=simulate_online(inp,pred,'predictive_pirc',params,cap)
                rows.append({'capacity':cap,'scenario':scenario,'seed':seed,
                    'method':'predictive_pirc_slow','accuracy':accuracy(cache,mask),**st})
            print('SLOW',cap,scenario,flush=True)
    save_csv(out,rows)
    print('PARAMS',params,flush=True)

if __name__=='__main__':main()
