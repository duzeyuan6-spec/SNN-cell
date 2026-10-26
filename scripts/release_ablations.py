"""Matched-weight PIRC mechanism ablations; no classifier labels enter the scheduler."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from local_pirc.outage_sweep import stream_inputs
from local_pirc.predictive_online import predict
from local_pirc.seven_baselines import SEEDS,load_cache,score
from local_pirc.unified_system import current_only_coef,current_only_predict,simulate

POLICIES=('pirc_full','pirc_noreserve','pirc_nohistory','pirc_nosharing')
METRICS=('accuracy','rcir','outage_service','executed_mac_per_clip',
         'high_calls_per_clip','overflow_rate','global_crsr','adjacent_crsr')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--split',choices=('validation','official_test'),required=True)
    ap.add_argument('--out',default='results/release_ablations')
    a=ap.parse_args()
    root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
    dest=root/f'{a.split}.csv'
    assert not dest.exists(),f'Refuse overwrite: {dest}'
    cache=load_cache(a.split)
    rates=np.load('results/longrun_seed42/fit_rates.npy')
    current_coef=current_only_coef(rates)
    full_coef=np.asarray(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    cfg=json.loads(Path('configs/resource_controller.json').read_text())['config']
    rows=[]
    for seed in SEEDS[a.split]:
        inp=stream_inputs(cache['rate'],'d64_v50',seed)
        full=predict(inp,full_coef)
        current=current_only_predict(inp,current_coef)
        baseline=None
        for policy in POLICIES:
            forecast=current if policy=='pirc_nohistory' else full
            mask,st=simulate(inp,forecast,policy,cfg)
            pair=(st['original_requests_hash'],st['supply_hash'])
            if baseline is None:baseline=pair
            assert pair==baseline
            rows.append({'split':a.split,'seed':seed,'policy':policy,
                         'accuracy':score(cache,mask),**{k:st[k] for k in METRICS if k!='accuracy'}})
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)
    print('SAVED',dest,flush=True)
if __name__=='__main__':main()
