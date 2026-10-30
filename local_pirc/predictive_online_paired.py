"""Paired validation accuracy differences with clips as resampling unit."""
import json
from pathlib import Path
import numpy as np
from .outage_sweep import stream_inputs
from .predictive_online import VAL_CLIPS, VAL_SEEDS, predict, save_csv, simulate_online
from .model import T


def main():
    root=Path('results/predictive_online_seed42')
    cache=np.load('results/local_seed42/controllers/validation_cache.npz')
    coef=np.array(json.loads((root/'predictor.json').read_text())['coef'])
    chosen=json.loads((root/'chosen.json').read_text())
    forced=json.loads((root/'forced_forecast_chosen.json').read_text())
    slow=json.loads((root/'slow_feedback_chosen.json').read_text())['params']
    policies={'threshold':('threshold',chosen['threshold']),
      'predictive_threshold_forced':('predictive_threshold',forced['predictive_threshold']),
      'predictive_pirc':('predictive_pirc',chosen['predictive_pirc']),
      'predictive_pirc_forced':('predictive_pirc',forced['predictive_pirc']),
      'predictive_pirc_slow':('predictive_pirc',slow)}
    right={k:[] for k in policies}; stats={k:[] for k in policies}
    y=cache['labels'][:VAL_CLIPS]
    for seed in VAL_SEEDS:
        inp=stream_inputs(cache['rate'][:VAL_CLIPS],'d64_v50',seed)
        forecast=predict(inp,coef)
        for name,(method,params) in policies.items():
            mask,st,_=simulate_online(inp,forecast,method,params,4.)
            logits=cache['low'][:VAL_CLIPS]+np.einsum('ntg,ntgc->nc',mask,
                cache['corr'][:VAL_CLIPS],optimize=True)/T
            right[name].append((logits.argmax(1)==y).astype(float));stats[name].append(st)
    base=np.mean(right['threshold'],axis=0)
    rng=np.random.default_rng(42);draw=rng.integers(0,VAL_CLIPS,size=(10000,VAL_CLIPS))
    rows=[]
    for name in policies:
        diff=np.mean(right[name],axis=0)-base
        ci=np.quantile(diff[draw].mean(1),[.025,.975])
        rows.append({'method':name,'accuracy_delta_vs_threshold':float(diff.mean()),
           'clip_bootstrap_ci_low':float(ci[0]),'clip_bootstrap_ci_high':float(ci[1]),
           'rcir_delta_vs_threshold':float(np.mean([a['rcir']-b['rcir']
              for a,b in zip(stats[name],stats['threshold'])])),
           'outage_service_delta_vs_threshold':float(np.mean([a['outage_service']-b['outage_service']
              for a,b in zip(stats[name],stats['threshold'])]))})
    save_csv(root/'paired_primary.csv',rows)
    print(rows)

if __name__=='__main__':main()
