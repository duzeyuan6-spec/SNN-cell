"""Post-calibration validation classification check for deployed risk schedulers."""
import csv
import json
from pathlib import Path
import numpy as np
from .demand_diagnostic import REF,VAL_SEEDS,activity_history,RiskModel
from .outage_sweep import SCENARIOS,stream_inputs,simulate
from .select_controllers import score


def main():
    root=Path('results/demand_diagnostic_seed42')
    assert (root/'validation_policy.csv').exists()
    dest=root/'validation_accuracy.csv'
    assert not dest.exists(),'Refuse to overwrite existing accuracy'
    chosen=json.loads((root/'choices.json').read_text())
    saved=json.loads((root/'models.json').read_text())
    models={kind:RiskModel.from_dict(saved[f'16_{kind}']) for kind in ('reserve','activity')}
    z=np.load('results/local_seed42/controllers/validation_cache.npz')
    low,corr,rates,y=[z[k] for k in ('low','corr','rate','labels')]
    rows=[]
    for scenario in SCENARIOS:
        for seed in VAL_SEEDS:
            inp=stream_inputs(rates,scenario,seed)
            hist=activity_history(inp['activity'])
            for method in ('reserve_threshold','risk_reserve','risk_activity'):
                if method=='reserve_threshold':kw={}
                else:
                    kind=method.removeprefix('risk_')
                    kw={'risk_model':models[kind],'risk_gain':chosen[kind],'activity_features':hist}
                mask,st,_=simulate(inp,method,**REF,**kw)
                perf=score(low,corr,y,mask)
                rows.append({'scenario':scenario,'seed':seed,'method':method,
                    'accuracy':perf['accuracy'],'cross_entropy':perf['ce'],
                    'physical_high_calls':int(mask.sum()),'mac_per_recording':st['mac_per_recording']})
        print('ACCURACY',scenario,'done',flush=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
