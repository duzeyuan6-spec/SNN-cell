"""Compare physical decisions made by the two frozen predicted-risk policies."""
import csv
import json
from pathlib import Path
import numpy as np
from .demand_diagnostic import VAL_SEEDS,REF,RiskModel,activity_history
from .outage_sweep import SCENARIOS,stream_inputs,simulate


def main():
    root=Path('results/demand_diagnostic_seed42')
    dest=root/'validation_action_audit.csv'
    assert not dest.exists(),'Refuse overwrite'
    models=json.loads((root/'models.json').read_text())
    chosen=json.loads((root/'choices.json').read_text())
    z=np.load('results/local_seed42/controllers/validation_cache.npz')
    rate=z['rate'];rows=[]
    for scenario in SCENARIOS:
        for seed in VAL_SEEDS:
            inp=stream_inputs(rate,scenario,seed)
            hist=activity_history(inp['activity'])
            runs={}
            for kind in ('reserve','activity'):
                model=RiskModel.from_dict(models[f'16_{kind}'])
                _,st,tr=simulate(inp,'risk_'+kind,**REF,risk_model=model,
                                 risk_gain=chosen[kind],activity_features=hist)
                runs[kind]=(st,tr)
            sa,ta=runs['activity'];sr,tr=runs['reserve']
            changed=ta['executed']!=tr['executed']
            rows.append({'scenario':scenario,'seed':seed,
                'allocation_l1_tokens_per_bin':float(np.abs(ta['allocation']-tr['allocation']).sum()/len(inp['supply'])),
                'different_call_decisions':int(changed.sum()),
                'different_call_fraction_of_requests':float(changed.sum()/inp['requested'].sum()),
                'different_outage_calls':int(changed[inp['outage']].sum()),
                'outage_call_gain':int(ta['executed'][inp['outage']].sum()-tr['executed'][inp['outage']].sum()),
                'overall_call_gain':int(sa['executed']-sr['executed']),
                'supply_total':float(inp['supply'].sum()),
                'original_requests':int(inp['requested'].sum())})
        print('ACTION',scenario,'done',flush=True)
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
