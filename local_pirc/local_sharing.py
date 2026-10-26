"""Conservative adjacent-patch token sharing under identical external supply."""
import csv
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np
from .model import T, GROUPS, dense_mac_per_t
from .outage_sweep import DELAY, SCENARIOS, stream_inputs
from .predictive_online import predict, accuracy

G = GROUPS
NEIGHBORS = ((1,2),(0,3),(0,3),(1,2))
EDGES = ((0,1),(0,2),(1,3),(2,3))
FIT_SEEDS = (701,702,703)
VAL_SEEDS = (711,712,713)
FIT_CLIPS = 128
VAL_CLIPS = 64
CAPS = (4.,12.)
GAIN = 4.
BOOST = 2.
TRANSFER_BUDGET = 1.
RHO = .9


def softmax(x):
    z=np.exp(x-x.max());return z/z.sum()


def surplus_after_own_request(available,requested):
    return np.maximum(0.,available-requested.astype(float))


def transfer(available,requested,activity,forecast,slow,method,params,safe=None,recipient_priority=None):
    """All transfers are causal, adjacent, conserved, and before current calls."""
    budget=TRANSFER_BUDGET;used={edge:0. for edge in EDGES};total=0.
    if method=='threshold':return total,budget,used
    deficient=[r for r in range(G) if requested[r] and available[r]<1.-1e-9]
    if recipient_priority is not None:
        assert np.asarray(recipient_priority).shape==(G,)
        deficient.sort(key=lambda r:(-recipient_priority[r],1.-available[r],r))
    elif method in ('balance','safe_fixed','safe_adaptive'):
        deficient.sort(key=lambda r:(1.-available[r],r))
    else:
        rg=params['risk_gain'];sg=params['slow_gain']
        deficient.sort(key=lambda r:(-activity[r]-rg*forecast[r]-sg*slow[r],r))
    for r in deficient:
        need=max(0.,1.-available[r])
        if need<=1e-9 or need>budget+1e-9:continue
        donors=[]
        for d in NEIGHBORS[r]:
            edge=tuple(sorted((r,d)))
            headroom=available[d]-float(requested[d])
            if safe is not None:headroom-=safe[d]
            if method=='pirc_share':
                headroom-=params['protect']*(forecast[d]+slow[d])
            cap=max(0.,min(headroom,1.-used[edge]))
            if cap>1e-9:
                priority=(headroom if method in ('balance','safe_fixed','safe_adaptive') else
                    headroom-params['risk_gain']*forecast[d]-params['slow_gain']*slow[d])
                donors.append((priority,d,cap,edge))
        if sum(a[2] for a in donors)+1e-9<need:continue
        donors.sort(key=lambda v:(-v[0],v[1]))
        for _,d,cap,edge in donors:
            amount=min(cap,need)
            available[d]-=amount;available[r]+=amount
            used[edge]+=amount;budget-=amount;total+=amount;need-=amount
            if need<=1e-9:break
        assert need<=1e-8
    assert total<=TRANSFER_BUDGET+1e-8
    assert all(v<=1.+1e-8 for v in used.values())
    return total,budget,used


def simulate_sharing(inp,forecast,method,params,cap,*,fixed_incoming=None,record_incoming=False,record_trace=False):
    assert method in ('threshold','balance','pirc_share','safe_fixed','safe_adaptive')
    requested=inp['requested'];activity=inp['activity'];supply=inp['supply']
    n=len(supply);assert forecast.shape==(n,G)
    reserve=np.full(G,2.,dtype=float)
    queue=[np.full(G,1./G) for _ in range(DELAY)]
    served=np.zeros_like(requested);end_trace=np.empty((n,G),dtype=np.float32)
    incoming_total=np.zeros(G);demand_total=np.zeros(G);call_total=np.zeros(G)
    spill_total=np.zeros(G);transfer_in=np.zeros(G);transfer_out=np.zeros(G)
    slow=np.zeros(G);slow_demand=np.zeros(G);slow_supply=np.zeros(G)
    safe_sum=np.zeros(G);transfer_tokens=0.;share_events=0
    incoming_trace=np.empty((n,G),dtype=np.float64) if record_incoming else None
    transfer_trace=np.zeros(n,dtype=np.float32) if record_trace else None
    spill_trace=np.zeros(n,dtype=np.float32) if record_trace else None
    if fixed_incoming is not None:
        assert fixed_incoming.shape==(n,G)
        assert np.all(fixed_incoming>=-1e-9)
        assert np.allclose(fixed_incoming.sum(1),supply,atol=1e-8)
    global_crsr=0;local_crsr=0;budget_crsr=0;conflict_bins=0
    for t in range(n):
        weights=queue.pop(0)
        incoming=supply[t]*weights if fixed_incoming is None else fixed_incoming[t]
        if incoming_trace is not None:incoming_trace[t]=incoming
        incoming_total+=incoming
        pre=reserve+incoming
        spill=np.maximum(pre-cap,0.)
        spill_total+=spill
        if spill_trace is not None:spill_trace[t]=spill.sum()
        available=np.minimum(pre,cap)
        before=available.copy()
        if method=='safe_fixed':
            safe=np.full(G,params['safe'],dtype=float)
        elif method=='safe_adaptive':
            rho=params['rho']
            slow_demand=rho*slow_demand+(1.-rho)*requested[t].astype(float)
            slow_supply=rho*slow_supply+(1.-rho)*incoming
            pressure=4.*np.maximum(0.,slow_demand-slow_supply)
            safe=np.clip(params['safe_base']+params['safe_gain']*pressure,0.,cap)
        else:safe=None
        if safe is not None:safe_sum+=safe
        moved,remaining,edge_used=transfer(available,requested[t],activity[t],
            forecast[t],slow,method,params,safe=safe)
        transfer_tokens+=moved
        if transfer_trace is not None:transfer_trace[t]=moved
        share_events+=int(moved>1e-9)
        delta=available-before
        transfer_in+=np.maximum(delta,0.)
        transfer_out+=np.maximum(-delta,0.)
        assert abs(float(delta.sum()))<1e-8
        call=requested[t]&(available>=1.-1e-8)
        served[t]=call
        end=available-call.astype(float)
        failed=requested[t]&~call
        donor_surplus=surplus_after_own_request(available,requested[t])
        for r in np.flatnonzero(failed):
            need=1.-available[r]
            other=np.array([d for d in range(G) if d!=r],dtype=int)
            if np.any(donor_surplus[other]>1e-8):global_crsr+=1
            neighbor=np.array(NEIGHBORS[r],dtype=int)
            possible=sum(max(0.,min(donor_surplus[d],1.-edge_used[tuple(sorted((r,int(d))))]))
                         for d in neighbor)
            if possible+1e-8>=need:
                local_crsr+=1
                if remaining+1e-8>=need:budget_crsr+=1
        if np.any(failed) and np.any(donor_surplus>1e-8):conflict_bins+=1
        reserve=end
        demand_total+=requested[t]
        call_total+=call
        slow=RHO*slow+(1.-RHO)*failed.astype(float)
        target=softmax(GAIN*requested[t].astype(float)+
            BOOST*np.maximum(0.,(cap-reserve)/cap))
        queue.append(target)
        end_trace[t]=reserve
        assert np.all(reserve>=-1e-7) and np.all(reserve<=cap+1e-7)
    calls=int(served.sum());req=int(requested.sum());total_supply=G*2.+float(supply.sum())
    assert calls+int((requested&~served).sum())==req
    assert abs(total_supply-calls-spill_total.sum()-reserve.sum())<1e-4
    assert abs(transfer_in.sum()-transfer_out.sum())<1e-5
    out=inp['outage'];outreq=int(requested[out].sum())
    base,extra=dense_mac_per_t()
    stats={'requests':req,'calls':calls,'rcir':1.-calls/req,
        'outage_service':float(served[out].sum()/outreq),
        'global_crsr':global_crsr/req,'local_crsr':local_crsr/req,
        'budget_crsr':budget_crsr/req,'global_starved_calls':global_crsr,
        'local_starved_calls':local_crsr,'budget_starved_calls':budget_crsr,
        'conflict_bins':conflict_bins,'spill_tokens':float(spill_total.sum()),
        'spill_fraction':float(spill_total.sum()/total_supply),
        'transfer_tokens':transfer_tokens,'share_events':share_events,
        'supply_tokens':float(supply.sum()),'terminal_reserve':float(reserve.sum()),
        'calls_per_clip':calls/inp['nclips'],
        'network_mac_per_clip':T*base+calls/inp['nclips']*extra}
    mask=served.reshape(inp['nclips'],T,G)
    original=np.empty_like(mask);original[inp['order']]=mask
    detail={'demand':demand_total,'calls':call_total,'allocated_supply':incoming_total,
        'spill':spill_total,'transfer_in':transfer_in,'transfer_out':transfer_out,
        'mean_reserve':end_trace.mean(0),'terminal_reserve':reserve,
        'mean_safe_target':safe_sum/n}
    if incoming_trace is not None:detail['incoming_trace']=incoming_trace
    if transfer_trace is not None:
        detail['reserve_trace']=end_trace
        detail['transfer_trace']=transfer_trace
        detail['spill_trace']=spill_trace
    return original,stats,detail


def objective(s):
    return .5*s['rcir']+.5*(1.-s['outage_service'])+.1*s['spill_fraction']


def save_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def calibrate(rates,coef,root):
    traces=[]
    for seed in FIT_SEEDS:
        inp=stream_inputs(rates[:FIT_CLIPS],'d64_v50',seed)
        traces.append((inp,predict(inp,coef)))
    rows=[]
    for rg,sg,protect in itertools.product((0.,1.,2.),(.5,1.,2.),(0.,.25,.5)):
        params={'risk_gain':rg,'slow_gain':sg,'protect':protect}
        stats=[simulate_sharing(inp,pred,'pirc_share',params,4.)[1]
               for inp,pred in traces]
        rows.append({'params':json.dumps(params,sort_keys=True),
            'objective':float(np.mean([objective(s) for s in stats])),
            'rcir':float(np.mean([s['rcir'] for s in stats])),
            'outage_service':float(np.mean([s['outage_service'] for s in stats])),
            'global_crsr':float(np.mean([s['global_crsr'] for s in stats])),
            'local_crsr':float(np.mean([s['local_crsr'] for s in stats])),
            'transfer_tokens':float(np.mean([s['transfer_tokens'] for s in stats]))})
    save_csv(root/'fit_candidates.csv',rows)
    best=min(rows,key=lambda r:(r['objective'],r['params']))
    chosen=json.loads(best['params'])
    (root/'chosen.json').write_text(json.dumps({'params':chosen,'fit_objective':best['objective']},indent=2))
    return chosen


def main():
    root=Path('results/local_sharing_seed42');root.mkdir(parents=True,exist_ok=True)
    assert not (root/'validation_results.csv').exists(),'Refuse overwrite'
    fit=np.load('results/longrun_seed42/fit_rates.npy')
    cache=np.load('results/local_seed42/controllers/validation_cache.npz')
    assert fit.shape==(924,T,G) and cache['rate'].shape==(252,T,G)
    coef=np.array(json.loads(Path('results/predictive_online_seed42/predictor.json').read_text())['coef'])
    params=calibrate(fit,coef,root)
    print('CHOSEN',params,flush=True)
    rows=[];detail=[];masks={}
    for cap in CAPS:
        for scenario in SCENARIOS:
            for seed in VAL_SEEDS:
                inp=stream_inputs(cache['rate'][:VAL_CLIPS],scenario,seed)
                pred=predict(inp,coef)
                for method in ('threshold','balance','pirc_share'):
                    mask,st,groups=simulate_sharing(inp,pred,method,params,cap)
                    rows.append({'capacity':cap,'scenario':scenario,'seed':seed,'method':method,
                        'accuracy':accuracy(cache,mask),**st})
                    for g in range(G):
                        detail.append({'capacity':cap,'scenario':scenario,'seed':seed,
                            'method':method,'group':g,**{k:float(v[g]) for k,v in groups.items()}})
                    if cap==4. and scenario=='d64_v50' and seed==711:
                        masks[method]=mask
            print('VALIDATED',cap,scenario,flush=True)
    save_csv(root/'validation_results.csv',rows)
    save_csv(root/'group_detail.csv',detail)
    np.savez_compressed(root/'primary_masks.npz',**masks)
    ck=hashlib.sha256(Path('results/local_seed42/best.pt').read_bytes()).hexdigest()
    (root/'manifest.json').write_text(json.dumps({'protocol_commit':'fb89925',
        'checkpoint_sha256':ck,'fit_seeds':FIT_SEEDS,'val_seeds':VAL_SEEDS,
        'fit_clips':FIT_CLIPS,'val_clips':VAL_CLIPS,'capacities':CAPS,
        'scenarios':SCENARIOS,'transfer_budget_per_bin':TRANSFER_BUDGET,
        'neighbors':NEIGHBORS,'test_labels_used':False},indent=2))
    print('SAVED',root,flush=True)

if __name__=='__main__':main()
