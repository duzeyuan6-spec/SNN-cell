"""Fit the causal activity-history demand predictor using fit split only."""
import json
from pathlib import Path
import numpy as np
from local_pirc.predictive_online import FIT_SEEDS,H,fit_predictor

def main():
    rates=np.load('results/longrun_seed42/fit_rates.npy')
    dest=Path('results/predictive_online_seed42/predictor.json')
    assert not dest.exists(),f'Refuse overwrite: {dest}'
    dest.parent.mkdir(parents=True,exist_ok=True)
    coef=fit_predictor(rates)
    dest.write_text(json.dumps({'horizon':H,'coef':coef.tolist(),
        'fit_nclips':len(rates),'fit_seeds':FIT_SEEDS},indent=2)+'\n')
    print('SAVED',dest,flush=True)
if __name__=='__main__': main()
