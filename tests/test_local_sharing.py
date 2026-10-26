import numpy as np
from local_pirc.model import T,GROUPS
from local_pirc.outage_sweep import stream_inputs,simulate
from local_pirc.local_sharing import transfer,simulate_sharing


def test_adjacent_transfer_conserves_tokens_and_donor_call():
    available=np.array([.4,2.,0.,0.])
    req=np.array([True,True,False,False])
    before=available.copy()
    moved,remaining,used=transfer(available,req,np.array([.9,.8,.1,.1]),
        np.zeros(4),np.zeros(4),'balance',{})
    assert .59<moved<.61
    assert available[0]>=1.-1e-8
    assert available[1]>=1.
    assert abs(available.sum()-before.sum())<1e-8
    assert moved<=1. and remaining>=0.
    assert used[(0,1)]>0 and sum(used.values())<=1.


def test_threshold_matches_existing_supply_simulation():
    rates=np.zeros((32,T,GROUPS));rates[:,:,0]=.8;rates[:,:,1]=.7
    inp=stream_inputs(rates,'d64_v50',77)
    pred=np.full_like(inp['activity'],.5)
    oldmask,old,_=simulate(inp,'reserve_threshold',gain=4.,boost=2.,level=4.,reserve_max=4.)
    mask,st,_=simulate_sharing(inp,pred,'threshold',{},4.)
    np.testing.assert_array_equal(mask,oldmask)
    assert st['calls']==old['executed']
    assert abs(st['spill_tokens']-old['spill_tokens'])<1e-5
    assert st['transfer_tokens']==0


def test_sharing_does_not_modify_external_requests_or_supply():
    rng=np.random.default_rng(1)
    rates=rng.uniform(size=(32,T,GROUPS))
    inp=stream_inputs(rates,'d32_v50',12)
    pred=np.full_like(inp['activity'],.5)
    params={'risk_gain':1.,'slow_gain':1.,'protect':.25}
    for method in ('threshold','balance','pirc_share'):
        _,st,_=simulate_sharing(inp,pred,method,params,4.)
        assert st['requests']==int(inp['requested'].sum())
        assert st['supply_tokens']==float(inp['supply'].sum())
        assert 0<=st['transfer_tokens']<=len(inp['supply'])
        assert st['calls']+round(st['rcir']*st['requests'])==st['requests']


def test_fixed_incoming_replay_matches_baseline():
    rng=np.random.default_rng(9)
    rates=rng.uniform(size=(32,T,GROUPS))
    inp=stream_inputs(rates,'d64_v50',19)
    pred=np.full_like(inp['activity'],.5)
    mask,st,detail=simulate_sharing(inp,pred,'threshold',{},4.,record_incoming=True)
    replay,other,_=simulate_sharing(inp,pred,'threshold',{},4.,
                                    fixed_incoming=detail['incoming_trace'])
    np.testing.assert_array_equal(mask,replay)
    assert st['calls']==other['calls']
    assert abs(st['spill_tokens']-other['spill_tokens'])<1e-8


def test_failed_call_fraction_is_not_donor_surplus():
    from local_pirc.local_sharing import surplus_after_own_request
    available=np.array([.5,.8,1.6,0.])
    requested=np.array([True,True,True,False])
    np.testing.assert_allclose(surplus_after_own_request(available,requested),
                               [0.,0.,.6,0.])


def test_nonadjacent_surplus_cannot_rescue_request():
    available=np.array([.4,0.,0.,2.])
    req=np.array([True,False,False,False])
    moved,_,_=transfer(available,req,np.zeros(4),np.zeros(4),np.zeros(4),
                       'balance',{})
    assert moved==0.
    assert available[0]==.4
    assert available[3]==2.


def test_fixed_safety_reserve_prevents_donor_depletion():
    available=np.array([.4,2.,0.,0.])
    req=np.array([True,True,False,False])
    safe=np.array([0.,1.,0.,0.])
    moved,_,_=transfer(available,req,np.zeros(4),np.zeros(4),np.zeros(4),
                       'safe_fixed',{'safe':1.},safe=safe)
    assert moved==0.
    np.testing.assert_allclose(available,[.4,2.,0.,0.])


def test_adaptive_safety_is_causal_and_supply_conserving():
    rng=np.random.default_rng(4)
    rates=rng.uniform(size=(32,T,GROUPS))
    inp=stream_inputs(rates,'d64_v50',19)
    params={'safe_base':.5,'safe_gain':1.,'rho':.9}
    zero=np.zeros_like(inp['activity'])
    mask,stats,detail=simulate_sharing(inp,zero,'safe_adaptive',params,4.,
                                       record_trace=True)
    changed={key:value.copy() if isinstance(value,np.ndarray) else value
             for key,value in inp.items()}
    changed['supply'][12:]=0.
    changed['activity'][12:]=1.-changed['activity'][12:]
    changed['requested'][12:]=~changed['requested'][12:]
    changed['outage']=inp['outage']
    replay,_,replay_detail=simulate_sharing(changed,zero,'safe_adaptive',params,4.,
                                            record_trace=True)
    np.testing.assert_array_equal(mask[inp['order']].reshape(-1,GROUPS)[:8],
                                  replay[inp['order']].reshape(-1,GROUPS)[:8])
    np.testing.assert_allclose(detail['reserve_trace'][:8],
                               replay_detail['reserve_trace'][:8])
    assert stats['requests']==int(inp['requested'].sum())
    assert 0<=stats['transfer_tokens']<=len(inp['supply'])
