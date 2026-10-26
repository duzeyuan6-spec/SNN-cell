import numpy as np

from local_pirc.outage_sweep import stream_inputs
from local_pirc.unified_system import simulate, POLICIES


def make_input():
    rng = np.random.default_rng(17)
    rates = rng.random((40, 16, 4))
    inp = stream_inputs(rates, 'd64_v50', 731)
    forecast = np.clip(.2 + .6 * inp['activity'], 0., 1.)
    return inp, forecast


def test_matched_demand_and_conservation():
    inp, forecast = make_input()
    reference = None
    for policy in POLICIES:
        mask, stats = simulate(inp, forecast, policy)
        assert mask.shape == (40, 16, 4)
        assert stats['requests'] == 40 * 32
        assert stats['calls'] == mask.sum()
        assert np.isclose(stats['rcir'], 1. - stats['calls'] / stats['requests'])
        assert 0. <= stats['outage_service'] <= 1.
        if policy == 'pirc_nosharing':
            assert stats['transferred_tokens'] == 0.
        if reference is None:
            reference = (stats['original_requests_hash'], stats['supply_hash'])
        else:
            assert (stats['original_requests_hash'], stats['supply_hash']) == reference


def test_causal_prefix():
    inp, forecast = make_input()
    second = dict(inp)
    second['supply'] = inp['supply'].copy()
    second['supply'][500:] *= 1.3
    second['activity'] = inp['activity'].copy()
    second['activity'][500:] = 0.
    second_forecast = forecast.copy()
    second_forecast[500:] = 1.
    for policy in POLICIES:
        first_mask, _ = simulate(inp, forecast, policy)
        other_mask, _ = simulate(second, second_forecast, policy)
        ordered_first = first_mask[inp['order']].reshape(-1, 4)
        ordered_other = other_mask[inp['order']].reshape(-1, 4)
        assert np.array_equal(ordered_first[:500], ordered_other[:500])


def test_supply_aware_config_is_causal_and_conservative():
    inp, forecast = make_input()
    cfg = {'safe_floor': .75, 'safe_gap_gain': .5, 'burst_gain': 2.,
           'safe_forecast_gain': 1., 'transfer_priority_gain': 0.,
           'alloc_forecast_gain': 0.}
    mask, stats = simulate(inp, forecast, 'pirc_full', cfg)
    changed = dict(inp)
    changed['supply'] = inp['supply'].copy()
    changed['supply'][500:] *= 1.7
    other, _ = simulate(changed, forecast, 'pirc_full', cfg)
    assert np.array_equal(mask[inp['order']].reshape(-1, 4)[:500],
                          other[inp['order']].reshape(-1, 4)[:500])
    assert stats['calls'] == mask.sum()
    assert 0. <= stats['overflow_rate'] <= 1.


def test_projected_headroom_policies_are_causal():
    inp, forecast = make_input()
    cfg = {'alloc_forecast_gain': 1., 'burst_gain': 2., 'fast_gain': 0.,
           'lookahead_gain': 2., 'safe_floor': .5,
           'safe_forecast_gain': 1., 'safe_gap_gain': .5,
           'transfer_priority_gain': 0.}
    changed = dict(inp)
    changed['supply'] = inp['supply'].copy()
    changed['supply'][500:] *= 1.4
    changed['activity'] = inp['activity'].copy()
    changed['activity'][500:] = 0.
    other_forecast = forecast.copy()
    other_forecast[500:] = 1.
    for policy in ('predictive_share',) + POLICIES:
        first, stats = simulate(inp, forecast, policy, cfg)
        other, _ = simulate(changed, other_forecast, policy, cfg)
        assert np.array_equal(first[inp['order']].reshape(-1, 4)[:500],
                              other[inp['order']].reshape(-1, 4)[:500])
        assert stats['calls'] == first.sum()


def test_utility_priority_changes_only_competing_transfer_order():
    import numpy as np
    from local_pirc.local_sharing import transfer
    params={'risk_gain':0.,'slow_gain':0.,'protect':0.}
    req=np.array([True,True,False,False])
    available=np.array([0.,0.,2.,2.])
    moved,remaining,_=transfer(available,req,np.zeros(4),np.zeros(4),np.zeros(4),
        'safe_fixed',params,safe=np.zeros(4),recipient_priority=np.array([0.,1.,0.,0.]))
    assert moved==1. and remaining==0.
    assert available[1]>=1. and available[0]<1.
    assert abs(available.sum()-4.)<1e-9
