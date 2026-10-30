"""Equal-scenario descriptive summaries and paired supply-seed intervals."""
import csv
from pathlib import Path
import numpy as np
from .outage_sweep import SCENARIOS


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    root=Path('results/demand_diagnostic_seed42')
    prediction=read(root/'validation_prediction.csv')
    policy=read(root/'validation_policy.csv')
    accuracy=read(root/'validation_accuracy.csv')
    merged={(r['scenario'],r['seed'],r['method']):r for r in policy}
    for a in accuracy:
        row=merged[(a['scenario'],a['seed'],a['method'])]
        row['accuracy']=a['accuracy'];row['cross_entropy']=a['cross_entropy']
    write(root/'validation_policy_with_accuracy.csv',list(merged.values()))
    configs=[('prediction',prediction,('horizon','method'),('auroc','auprc','log_loss','brier','prevalence')),
             ('policy',list(merged.values()),('method',),('outage_service','rcir','service_rate','spill_fraction','accuracy','cross_entropy','mac_per_recording','high_calls_per_clip','preoutage_reserve'))]
    summaries=[]
    for label,rows,keys,metrics in configs:
        groups=sorted({tuple(r[k] for k in keys) for r in rows})
        for grp in groups:
            matches=[r for r in rows if tuple(r[k] for k in keys)==grp]
            for scenario in (*SCENARIOS,'all_equal_scenarios'):
                sub=[r for r in matches if r['scenario']==scenario or scenario=='all_equal_scenarios']
                for metric in metrics:
                    summaries.append({'table':label,'scenario':scenario,'horizon':grp[0] if label=='prediction' else '',
                        'method':grp[-1],'metric':metric,'mean':float(np.mean([float(r[metric]) for r in sub])),
                        'n_trace_rows':len(sub)})
    write(root/'validation_summary.csv',summaries)
    rng=np.random.default_rng(20261025)
    paired=[]
    for horizon in ('16','32'):
        lookup={(r['scenario'],r['seed'],r['method']):r for r in prediction if r['horizon']==horizon}
        for metric in ('auroc','auprc','log_loss','brier'):
            diffs=[]
            for seed in range(711,721):
                diffs.append(np.mean([float(lookup[(sc,str(seed),'activity')][metric])-float(lookup[(sc,str(seed),'reserve')][metric]) for sc in SCENARIOS]))
            d=np.asarray(diffs);boot=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            paired.append({'table':'prediction','horizon':horizon,'comparison':'activity_minus_reserve','metric':metric,
                'difference':float(d.mean()),'ci95_lo':float(np.quantile(boot,.025)),'ci95_hi':float(np.quantile(boot,.975)),
                'bootstrap_unit':'supply_seed_same_validation_clips'})
    lookup={(r['scenario'],r['seed'],r['method']):r for r in merged.values()}
    for other in ('risk_reserve','reserve_threshold'):
        for metric in ('outage_service','rcir','service_rate','spill_fraction','accuracy','mac_per_recording'):
            diffs=[]
            for seed in range(711,721):
                diffs.append(np.mean([float(lookup[(sc,str(seed),'risk_activity')][metric])-float(lookup[(sc,str(seed),other)][metric]) for sc in SCENARIOS]))
            d=np.asarray(diffs);boot=d[rng.integers(0,len(d),size=(5000,len(d)))].mean(1)
            paired.append({'table':'policy','horizon':'16','comparison':'risk_activity_minus_'+other,'metric':metric,
                'difference':float(d.mean()),'ci95_lo':float(np.quantile(boot,.025)),'ci95_hi':float(np.quantile(boot,.975)),
                'bootstrap_unit':'supply_seed_same_validation_clips'})
    write(root/'validation_paired.csv',paired)

if __name__=='__main__':main()
