"""Exploratory attribution: current firing rate versus current plus causal history."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from .demand_diagnostic import FIT_SEEDS,VAL_SEEDS,HORIZONS,trace_data,future_shortage,feature_matrix,fit_model,classification_metrics
from .outage_sweep import SCENARIOS


def xy(trace,hist,horizon):
    n=len(trace['reserve'])-horizon
    reserve=feature_matrix(trace['reserve'][:n],hist[:n],'reserve')
    current=np.concatenate((reserve,hist[:n,:,0,None]),axis=-1).reshape(-1,5)
    target=future_shortage(trace['denied'],horizon).reshape(-1)
    return current,target


def main():
    root=Path('results/demand_diagnostic_seed42');dest=root/'exploratory_current_only.csv'
    assert not dest.exists(),'Refuse overwrite'
    fitrates=np.load('results/longrun_seed42/fit_rates.npy')
    valrates=np.load('results/local_seed42/controllers/validation_cache.npz')['rate']
    models={}
    for horizon in HORIZONS:
        x=[];y=[]
        for sc in SCENARIOS:
            for seed in FIT_SEEDS:
                _,tr,hist=trace_data(fitrates,sc,seed)
                xx,yy=xy(tr,hist,horizon);x.append(xx);y.append(yy)
        model,info=fit_model(np.concatenate(x),np.concatenate(y),'current_only')
        models[horizon]=model
        print('FIT_CURRENT',horizon,info,flush=True)
    rows=[]
    for sc in SCENARIOS:
        for seed in VAL_SEEDS:
            _,tr,hist=trace_data(valrates,sc,seed)
            for horizon in HORIZONS:
                x,y=xy(tr,hist,horizon)
                m=models[horizon]
                p=expit((x-m.mean)/m.scale@m.coef[:-1]+m.coef[-1])
                rows.append({'scenario':sc,'seed':seed,'horizon':horizon,'method':'reserve_plus_current',
                             'n_rows':len(y),**classification_metrics(y,p)})
        print('CURRENT_VALIDATION',sc,'done',flush=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (root/'exploratory_current_only_model.json').write_text(json.dumps({str(h):m.to_dict() for h,m in models.items()},indent=2))
    full={(r['scenario'],r['seed'],r['horizon']):r for r in csv.DictReader((root/'validation_prediction.csv').open()) if r['method']=='activity'}
    current={(r['scenario'],r['seed'],r['horizon']):r for r in rows}
    rng=np.random.default_rng(1414);paired=[]
    for horizon in HORIZONS:
        for metric in ('auroc','auprc','log_loss','brier'):
            d=np.asarray([np.mean([float(full[(sc,str(seed),str(horizon))][metric])-
                    float(current[(sc,seed,horizon)][metric]) for sc in SCENARIOS]) for seed in VAL_SEEDS])
            boot=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            paired.append({'horizon':horizon,'metric':metric,'full_minus_current':float(d.mean()),
                'ci95_lo':float(np.quantile(boot,.025)),'ci95_hi':float(np.quantile(boot,.975)),
                'note':'exploratory_posthoc_attribution; bootstrap unit supply seed'})
    with (root/'exploratory_current_only_paired.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=paired[0]);w.writeheader();w.writerows(paired)

if __name__=='__main__':main()
