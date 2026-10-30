"""Equal-scenario summaries for local sharing reliability and accuracy."""
import csv
from pathlib import Path
import numpy as np
from .outage_sweep import SCENARIOS
from .local_sharing import save_csv


def main():
    root=Path('results/local_sharing_seed42')
    rows=list(csv.DictReader((root/'validation_results.csv').open()))
    out=[]
    fields=('rcir','outage_service','global_crsr','local_crsr','budget_crsr',
        'spill_fraction','transfer_tokens','calls_per_clip','network_mac_per_clip','accuracy')
    for cap in (4.,12.):
        for sc in (*SCENARIOS,'all_equal_scenarios'):
            for method in ('threshold','balance','pirc_share'):
                sub=[r for r in rows if float(r['capacity'])==cap and r['method']==method
                    and (r['scenario']==sc or sc=='all_equal_scenarios')]
                assert len(sub)==(3 if sc!='all_equal_scenarios' else 18)
                out.append({'capacity':cap,'scenario':sc,'method':method,
                    'n_traces':len(sub),**{f:float(np.mean([float(r[f]) for r in sub]))
                    for f in fields}})
    save_csv(root/'summary.csv',out)
    print('SAVED',root/'summary.csv')

if __name__=='__main__':main()
