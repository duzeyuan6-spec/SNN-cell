import numpy as np
from local_pirc.outage_sweep import SCENARIOS,T,G,RMAX,DELAY,stream_inputs,simulate,scenario_spec


def test_six_supply_scenarios_same_budget_and_original_demand():
    rates=np.random.default_rng(31).uniform(.01,.2,(64,T,G))
    for scenario in SCENARIOS:
        duration,cv=scenario_spec(scenario)
        inp=stream_inputs(rates,scenario,701)
        assert np.isclose(inp['supply'].mean(),1.6)
        assert np.all(inp['supply'][inp['outage']]==0)
        assert inp['outage'][:512].sum()==duration
        assert inp['requested'].sum()==2*len(inp['supply'])
        assert np.array_equal(inp['requested'],stream_inputs(rates,scenario,701)['requested'])
        assert np.array_equal(inp['supply'],stream_inputs(rates,scenario,701)['supply'])
        if cv==0:
            assert np.unique(inp['supply'][inp['normal']]).size==1
        else:
            assert np.unique(inp['supply'][inp['normal']]).size>1


def test_all_controllers_conserve_tokens_and_skip_denied_calls_causally():
    rates=np.random.default_rng(71).uniform(.01,.2,(64,T,G))
    inp=stream_inputs(rates,'d64_v50',711)
    params={'fixed':{},'instant':{'gain':4.},'ema':{'gain':4.,'rho':.8},
            'reserve_threshold':{'gain':4.,'boost':4.,'level':4.},
            'pirc':{'fast_gain':6.,'slow_gain':2.,'rho':.95,'reserve_gain':4.,'debt_gain':2.},
            'pirc_noreserve':{'fast_gain':6.,'slow_gain':2.,'rho':.95,'reserve_gain':4.,'debt_gain':2.}}
    for method,kw in params.items():
        mask,stat,trace=simulate(inp,method,**kw)
        assert stat['requests']==2*len(inp['supply'])
        assert stat['requests']==stat['executed']+stat['interrupted']
        assert mask.sum()==stat['executed']
        assert np.all(trace['reserve']<=RMAX+1e-6)
        assert np.isclose(stat['total_supply'],stat['executed']+stat['spill_tokens']+stat['terminal_reserve'],atol=1e-3)
        assert np.allclose(trace['allocation'][:DELAY],inp['supply'][:DELAY,None]/G)
        assert np.isclose(stat['outage_service']+stat['outage_rcir'],1.)
        later=inp.copy();later['requested']=inp['requested'].copy()
        later['requested'][300:]=np.roll(later['requested'][300:],1,axis=1)
        later['supply']=inp['supply'].copy();later['supply'][300:]*=.5
        _,_,future=simulate(later,method,**kw)
        assert np.array_equal(trace['executed'][:300],future['executed'][:300])
        assert np.allclose(trace['reserve'][:300],future['reserve'][:300])
