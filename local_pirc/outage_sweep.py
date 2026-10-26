"""Outage-duration and volatile-supply comparison with conserved local tokens."""
import numpy as np
from .model import T,GROUPS,dense_mac_per_t

G=GROUPS
DELAY=4
RMAX=12.
INITIAL=2.
METHODS=('fixed','instant','ema','reserve_threshold','pirc','pirc_noreserve','risk_reserve','risk_activity')
SCENARIOS=tuple(f'd{duration}_v{volatility}' for duration in (32,64,128) for volatility in (0,50))


def scenario_spec(scenario):
    assert scenario in SCENARIOS,scenario
    duration=int(scenario.split('_')[0][1:])
    cv=int(scenario.split('_')[1][1:])/100.
    return duration,cv


def stream_inputs(rates,scenario,seed):
    """Identical exogenous supply and original requests for every policy."""
    rates=np.asarray(rates)
    assert rates.ndim==3 and rates.shape[1:]==(T,G)
    nclips=len(rates);ntime=nclips*T
    duration,cv=scenario_spec(scenario)
    order=np.random.default_rng(seed).permutation(nclips)
    activity=rates[order].reshape(ntime,G).astype(np.float64)
    priority=np.argsort(-activity,axis=1,kind='stable')[:,:2]
    requested=np.zeros((ntime,G),bool)
    np.put_along_axis(requested,priority,True,axis=1)
    assert np.all(requested.sum(1)==2)
    phase=np.arange(ntime)%512
    normal=phase<320;burst=(phase>=320)&(phase<384)
    outage=(phase>=384)&(phase<384+duration)
    recovery=phase>=384+duration
    supply=np.select((normal,burst,outage),(1.7,3.,0.),default=1.4).astype(np.float64)
    if cv:
        rng=np.random.default_rng(seed+100000+duration*1000+int(cv*100))
        innovation=rng.normal(size=ntime)
        latent=np.empty(ntime,dtype=np.float64)
        latent[0]=innovation[0]
        factor=np.sqrt(1-.9**2)
        for t in range(1,ntime):latent[t]=.9*latent[t-1]+factor*innovation[t]
        sigma=np.sqrt(np.log1p(cv*cv))
        supply*=np.exp(sigma*latent-.5*sigma*sigma)
    supply*=1.6/supply.mean()
    assert abs(supply.mean()-1.6)<1e-12 and np.all(supply[outage]==0)
    return {'activity':activity,'requested':requested,'supply':supply,'order':order,
            'scenario':scenario,'seed':seed,'nclips':nclips,'outage':outage,
            'normal':normal,'burst':burst,'recovery':recovery,
            'duration':duration,'cv':cv}

def _softmax(x):
    e=np.exp(x-x.max())
    return e/e.sum()


def _longest_interruption(blocked):
    longest=0
    for g in range(G):
        run=0
        for b in blocked[:,g]:
            run=run+1 if b else 0
            if run>longest:longest=run
    return longest


def simulate(inputs,method,*,gain=4.,rho=.9,reserve_gain=1.5,debt_gain=2.,fast_gain=0.,slow_gain=None,boost=4.,level=4.,return_masks=True,risk_model=None,risk_gain=0.,activity_features=None,reserve_max=RMAX,initial_reserve=INITIAL):
    assert method in METHODS
    activity=inputs['activity'];requested=inputs['requested'];supply=inputs['supply']
    ntime=len(supply);nclips=inputs['nclips']
    if method in ('risk_reserve','risk_activity'):
        assert risk_model is not None and activity_features is not None
        assert len(activity_features)==ntime
        assert risk_model.kind==method.removeprefix('risk_')
    reserve=np.full(G,initial_reserve,dtype=np.float64)
    slow=np.zeros(G);debt=np.zeros(G)
    queue=[np.full(G,1/G) for _ in range(DELAY)]
    executed=np.zeros((ntime,G),bool)
    denied=np.zeros((ntime,G),bool)
    reserves=np.zeros((ntime,G),np.float32)
    spilled=np.zeros((ntime,G),np.float32)
    allocated=np.zeros((ntime,G),np.float32)
    for t in range(ntime):
        alloc_weights=queue.pop(0)
        incoming=supply[t]*alloc_weights
        pre=reserve+incoming
        spill=np.maximum(pre-reserve_max,0.)
        available=np.minimum(pre,reserve_max)
        call=requested[t]&(available>=1.-1e-9)
        missing=requested[t]&~call
        reserve=available-call.astype(float)
        slow=rho*slow+(1-rho)*requested[t].astype(float)
        debt=rho*debt+(1-rho)*missing.astype(float)
        if method=='fixed':target=np.full(G,1/G)
        elif method=='instant':target=_softmax(gain*requested[t].astype(float))
        elif method=='ema':target=_softmax(gain*slow)
        elif method=='reserve_threshold':
            target=_softmax(gain*requested[t].astype(float)+boost*np.maximum(0.,(level-reserve)/level))
        elif method in ('risk_reserve','risk_activity'):
            risk=risk_model.predict(reserve,activity_features[t])
            target=_softmax(gain*requested[t].astype(float)+boost*np.maximum(0.,(level-reserve)/level)+risk_gain*risk)
        else:
            eta=0. if method=='pirc_noreserve' else reserve_gain
            mu=debt_gain
            target=_softmax(fast_gain*requested[t].astype(float)+(gain if slow_gain is None else slow_gain)*slow+eta*(1-reserve/reserve_max)+mu*debt)
        queue.append(target)
        executed[t]=call;denied[t]=missing;reserves[t]=reserve
        spilled[t]=spill;allocated[t]=incoming
        assert np.all(reserve>=-1e-8) and np.all(reserve<=reserve_max+1e-8)
    assert np.array_equal(requested,executed|denied)
    total_supply=G*initial_reserve+float(supply.sum())
    calls=int(executed.sum());lost=int(denied.sum());spill=float(spilled.sum())
    assert abs(total_supply-calls-spill-reserve.sum())<1e-3
    assert calls+lost==2*ntime
    group_req=requested.sum(0);group_calls=executed.sum(0)
    group_service=np.divide(group_calls,group_req,out=np.zeros(G,dtype=float),where=group_req>0)
    active=group_req>0
    jain=float(group_service[active].sum()**2/(active.sum()*np.square(group_service[active]).sum()))
    bin_service=executed.sum(1)/2.
    roll=np.convolve(bin_service,np.ones(128)/128,mode='valid')
    outage=inputs['outage'];normal=inputs['normal'];burst=inputs['burst']
    outage_requests=2*int(outage.sum())
    assert outage_requests>0
    base,extra=dense_mac_per_t()
    stats={'scenario':inputs['scenario'],'seed':inputs['seed'],'method':method,
           'nclips':nclips,'ntime':ntime,'requests':2*ntime,'executed':calls,'interrupted':lost,
           'outage_duration':inputs['duration'],'supply_cv':inputs['cv'],
           'rcir':lost/(2*ntime),'service_rate':calls/(2*ntime),
           'outage_requests':outage_requests,
           'outage_executed':int(executed[outage].sum()),
           'outage_rcir':float(denied[outage].sum()/outage_requests),
           'outage_service':float(executed[outage].sum()/outage_requests),
           'normal_service':float(executed[normal].sum()/(2*normal.sum())),
           'normal_rcir':float(denied[normal].sum()/(2*normal.sum())),
           'preoutage_reserve':float(reserves[(np.arange(ntime)%512)==383].mean()),
           'burst_spill_fraction':float(spilled[burst].sum()/max(supply[burst].sum(),1e-9)),
           'post_warmup_rcir':float(denied[min(256,ntime//2):].sum()/(2*(ntime-min(256,ntime//2)))),
           'rolling_service_p10':float(np.quantile(roll,.1)),
           'rolling_service_min':float(roll.min()),
           'rolling_service_std':float(roll.std()),
           'longest_interruption_run':_longest_interruption(denied),
           'group_service':group_service.tolist(),'jain_service':jain,
           'spill_tokens':spill,'spill_fraction':spill/total_supply,
           'depletion_fraction':float(np.mean(reserves<.01)),
           'mean_reserve':float(reserves.mean()),
           'total_supply':total_supply,'terminal_reserve':reserve.sum(),
           'controller_extra_mac_per_recording':T*G*(4 if method=='risk_reserve' else 8 if method=='risk_activity' else 0),
           'mac_per_recording':T*base+calls/nclips*extra+T*G*(4 if method=='risk_reserve' else 8 if method=='risk_activity' else 0)}
    if not return_masks:return stats
    ordered=executed.reshape(nclips,T,G)
    original=np.empty_like(ordered);original[inputs['order']]=ordered
    return original,stats,{'requested':requested,'executed':executed,'denied':denied,
                           'reserve':reserves,'spill':spilled,'allocation':allocated}
