"""Equal-seed scenario summaries; no parameter selection."""
import csv
from pathlib import Path
import numpy as np
from .model import T,dense_mac_per_t
from .outage_sweep import SCENARIOS
from .predictive_online import save_csv


def main():
    root=Path('results/predictive_online_seed42')
    rows=list(csv.DictReader((root/'validation_results.csv').open()))
    rows+=list(csv.DictReader((root/'forced_forecast_validation.csv').open()))
    rows+=list(csv.DictReader((root/'slow_feedback_validation.csv').open()))
    methods=('threshold','predictive_threshold','predictive_threshold_forced',
             'predictive_pirc','predictive_pirc_forced','predictive_pirc_slow')
    base,extra=dense_mac_per_t()
    out=[]
    for cap in (4.,12.):
        for scenario in (*SCENARIOS,'all_equal_scenarios'):
            for method in methods:
                sub=[r for r in rows if float(r['capacity'])==cap and r['method']==method
                    and (r['scenario']==scenario or scenario=='all_equal_scenarios')]
                assert len(sub)==(3 if scenario!='all_equal_scenarios' else 18)
                avg=lambda key:float(np.mean([float(r[key]) for r in sub]))
                out.append({'capacity':cap,'scenario':scenario,'method':method,
                    'n_traces':len(sub),'rcir':avg('rcir'),
                    'outage_service':avg('outage_service'),
                    'spill_fraction':avg('spill_fraction'),
                    'calls_per_clip':avg('high_calls_per_clip'),
                    'network_mac_per_clip':T*base+avg('high_calls_per_clip')*extra,
                    'estimated_controller_inclusive_mac_per_clip':avg('mac_per_clip'),
                    'accuracy':avg('accuracy')})
    save_csv(root/'summary.csv',out)
    print('SAVED',root/'summary.csv')

if __name__=='__main__':main()
